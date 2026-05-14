# type: ignore
"""vLLM-compatible Ouro model implementation.

This implementation adapts the Universal Transformer architecture of the Ouro model for vLLM compatibility.

Key adaptation choices:
1.  The Universal Transformer loop (`total_ut_steps`) is 'unrolled'. If the original model has `N` layers
    and runs for `T` steps, this implementation creates `N * T` distinct layer objects. This allows
    vLLM's KV cache to function correctly, as each pass gets a unique cache slot.
2.  Weight sharing is implemented during the weight loading process. The weights of the original `N`
    physical layers are loaded into all corresponding unrolled layers.
3.  The dynamic early-exit mechanism is omitted, as it is incompatible with vLLM's batched inference
    paradigm. The model always runs for the full number of configured steps.
"""
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata

# KVCache definition for type hinting
KVCache = Tuple[torch.Tensor, torch.Tensor]

from transformers import PretrainedConfig

class OuroConfig(PretrainedConfig):
    """
    HuggingFace-style model configuration for the Ouro model, adapted for vLLM.
    The default values are aligned with the official Ouro-7B-beta configuration.
    """
    model_type = "ouro"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None, # Ouro tokenizer might not have a pad token by default
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        total_ut_steps=4, # Crucial parameter for Ouro
        # Note: The following sliding window parameters are for config compatibility.
        # A full vLLM implementation would require logic to switch attention types per layer.
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.total_ut_steps = total_ut_steps
        #print(f"OuroConfig initialized with total_ut_steps={total_ut_steps}")
        # Store sliding window params for potential future use
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

class OuroAttention(nn.Module):
    def __init__(
        self,
        config: OuroConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        
        # CORRECT: vLLM's Attention layer handles RoPE internally.
        # We just need to pass the configuration.
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([
            self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        ], dim=-1)
        
        # CORRECT: Pass positions to the attention layer for RoPE calculation.
        attn_output = self.attn(positions, q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class OuroMLP(nn.Module):
    def __init__(
        self,
        config: OuroConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class OuroDecoderLayer(nn.Module):
    def __init__(
        self,
        config: OuroConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.self_attn = OuroAttention(config, quant_config)
        self.mlp = OuroMLP(config, quant_config)
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, kv_cache, attn_metadata)
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class OuroModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        
        self.total_layers = config.num_hidden_layers * config.total_ut_steps
        self.layers = nn.ModuleList([
            OuroDecoderLayer(config, quant_config) for _ in range(self.total_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # REMOVED: RoPE precomputation is not needed. vLLM handles it.

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        # CORRECT: Forward pass through the unrolled layers, passing necessary metadata.
        for i, layer in enumerate(self.layers):
            hidden_states = layer(positions, hidden_states, kv_caches[i], attn_metadata)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class OuroForvLLM(nn.Module):
    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = OuroModel(vllm_config)
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size)
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states, sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # --- COMPLETELY REWRITTEN WEIGHT LOADING LOGIC ---
        
        # Convert the weight iterator to a dictionary for easy lookups
        hf_weights_dict = dict(weights)
        
        # Get necessary config values for weight sharing
        num_physical_layers = self.config.num_hidden_layers

        # Iterate over the parameters of THIS vLLM model
        for vllm_name, param in self.named_parameters():
            # Skip buffers that are not loaded from checkpoints
            if "rotary_emb" in vllm_name or "freqs_cis" in vllm_name:
                continue

            # Determine the corresponding HuggingFace weight name
            if "layers" in vllm_name:
                # This is a layer weight that needs to be shared
                parts = vllm_name.split(".")
                unrolled_layer_idx = int(parts[2])
                
                # Convert the unrolled index back to the physical layer index
                physical_layer_idx = unrolled_layer_idx % num_physical_layers
                
                # Construct the base name for the HF physical layer
                hf_layer_name = f"model.layers.{physical_layer_idx}"
                
                # Find the correct HF weight(s) based on the vLLM parameter name
                if "self_attn.qkv_proj.weight" in vllm_name:
                    # Merge Q, K, V weights from the physical layer
                    q_w = hf_weights_dict[f"{hf_layer_name}.self_attn.q_proj.weight"]
                    k_w = hf_weights_dict[f"{hf_layer_name}.self_attn.k_proj.weight"]
                    v_w = hf_weights_dict[f"{hf_layer_name}.self_attn.v_proj.weight"]
                    loaded_weight = torch.cat([q_w, k_w, v_w], dim=0)
                elif "self_attn.o_proj.weight" in vllm_name:
                    loaded_weight = hf_weights_dict[f"{hf_layer_name}.self_attn.o_proj.weight"]
                elif "mlp.gate_up_proj.weight" in vllm_name:
                    # Merge Gate and Up weights
                    gate_w = hf_weights_dict[f"{hf_layer_name}.mlp.gate_proj.weight"]
                    up_w = hf_weights_dict[f"{hf_layer_name}.mlp.up_proj.weight"]
                    loaded_weight = torch.cat([gate_w, up_w], dim=0)
                elif "mlp.down_proj.weight" in vllm_name:
                    loaded_weight = hf_weights_dict[f"{hf_layer_name}.mlp.down_proj.weight"]
                # Handle the four LayerNorms
                elif "input_layernorm.weight" in vllm_name:
                    loaded_weight = hf_weights_dict[f"{hf_layer_name}.input_layernorm.weight"]
                elif "input_layernorm_2.weight" in vllm_name:
                    loaded_weight = hf_weights_dict[f"{hf_layer_name}.input_layernorm_2.weight"]
                elif "post_attention_layernorm.weight" in vllm_name:
                    loaded_weight = hf_weights_dict[f"{hf_layer_name}.post_attention_layernorm.weight"]
                elif "post_attention_layernorm_2.weight" in vllm_name:
                    loaded_weight = hf_weights_dict[f"{hf_layer_name}.post_attention_layernorm_2.weight"]
                else:
                    # This case should ideally not be hit if all layer params are covered
                    print(f"Warning: Unhandled layer parameter: {vllm_name}")
                    continue
            
            # Handle non-layer weights (embedding, final norm, lm_head)
            else:
                # These have a direct 1-to-1 mapping
                hf_name = vllm_name
                if hf_name not in hf_weights_dict:
                    print(f"Warning: Weight not found in checkpoint: {hf_name}")
                    continue
                loaded_weight = hf_weights_dict[hf_name]

            # Use the default loader to handle device placement and dtype conversion
            default_weight_loader(param, loaded_weight)

# Registration with vLLM
from vllm import ModelRegistry
ModelRegistry.register_model("OuroForCausalLM", OuroForvLLM)