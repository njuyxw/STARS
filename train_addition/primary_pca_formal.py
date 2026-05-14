import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import numpy as np
import random
from tqdm import tqdm
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap



class Config:
    num_digits = 4  # Each operand is 4 digits 
    
    # Vocabulary: 0-9, +, =, <pad>, <eos>, <space>
    vocab = [
        "0","1","2","3","4","5","6","7","8","9",
        "+","="," ",
        "Ʌ",   # PAD
        "§"    # EOS
    ]
    pad_token_id = vocab.index("Ʌ")
    eos_token_id = vocab.index("§")

    # Model architecture parameters
    d_model = 512
    n_heads = 8
    d_ff = d_model*2  # Hidden dimension in FFN
    dropout = 0.1
    max_len = 512  # Maximum sequence length (4+4=result can be up to ~15 digits)
    
    # Norm structures
    # pre_norm post_norm sandwich_branch sandwich_dual
    norm_structure = "sandwich_dual"
    # Norm types
    # LayerNorm RMSNorm DeepNorm SimpleNorm
    norm_type = "LayerNorm"

    # random loop
    # (log-normal distribution)
    # Log-normal distribution peak (mode) = e^(μ - σ²)
    random_loop = False
    random_loop_mu = 2.62    # 核心均值参数
    random_loop_sigma = 0.60 # 核心标准差参数
    random_loop_min = 1      # 最小循环次数
    random_loop_max = 40     # 最大循环次数

    # if DeepNorm:
    deepnorm_alpha = 1.5

    # prelude code
    prelude = False
    coda = False

    # Looped settings: k_layers * l_loops (effective depth)
    k_layers = 1
    l_loops = 1000   # Fixed loop count 用于测轨迹 所以设置很大
    
    # Training parameters (multi-GPU optimized)
    batch_size = 512  # Batch size per GPU, total batch size = batch_size * num_gpus
    accumulation_steps = 1  # Gradient accumulation steps
    total_steps = 10000  # Total training steps
    learning_rate = 1e-4
    warmup_steps = 2000
    # Auto-detect GPUs (ensure GPUs are available, e.g., cuda:0 and cuda:1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

config = Config()
# Print device information
print(f"Using {config.num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(config.num_gpus)]}")
def sample_random_loops(config):
    """
    Sample loop count from log-normal distribution with peak at 6
    Use log-normal distribution, then round and limit to [min, max] range
    """
    # Sample from log-normal distribution (peak around 6)
    log_normal_sample = np.random.lognormal(
        mean=config.random_loop_mu,
        sigma=config.random_loop_sigma
    )
    # Round and limit range
    num_loops = int(np.round(log_normal_sample))
    num_loops = max(config.random_loop_min, min(config.random_loop_max, num_loops))
    return num_loops

class CharTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.c2i = {c: i for i, c in enumerate(vocab)}
        self.i2c = {i: c for i, c in enumerate(vocab)}
    
    def encode(self, s):
        return [self.c2i[c] for c in s]
    
    def decode(self, indices):
        return "".join([self.i2c[i] for i in indices if i < len(self.vocab)])

tokenizer = CharTokenizer(config.vocab)

class AdditionDataset(IterableDataset):
    def __init__(self, tokenizer, config, mode='train', total_samples=1000000, data_dir='data'):
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode
        self.total_samples = total_samples
        self.data_dir = data_dir
        
        if mode == "train":
            file_path = os.path.join(data_dir, "train_4digit.jsonl")
        else:
            file_path = os.path.join(data_dir, "test_4digit.jsonl")
        
        # Build data file path list
        self.data_files = []
        if os.path.exists(file_path):
            self.data_files.append(file_path)
        else:
            print(f"Warning: Data file {file_path} not found")
        
        if len(self.data_files) == 0:
            raise ValueError(f"No data files found for mode={mode} in {data_dir}")
        
        print(f"Loading data from {len(self.data_files)} file(s): {self.data_files}")
        
        # For evaluation scenarios (fewer samples), preload to memory for faster access
        # For training scenarios (many samples), use streaming mode
        self.preload_threshold = 1000000  # Preload to memory if sample count is less than this
        self.data = None
        self.data_size = None
        if total_samples < self.preload_threshold:
            # Preload small amount of data to memory (for evaluation)
            print(f"Preloading {total_samples} samples into memory...")
            self._preload_data(total_samples)
            print(f"Preloaded {len(self.data)} samples.")
        else:
            # Training scenario: stream reading, no preloading
            print(f"Using streaming mode for {total_samples} samples (memory efficient)")
    
    def _parse_jsonl_line(self, line):
        """Parse a line from JSONL file and convert to model required format"""
        try:
            data = json.loads(line.strip())
            # New data format: num1, num2, expression, answer
            # Example: {"num1": 95822412, "num2": 24942603, "expression": "95822412 + 24942603", "answer": 120765015}
            num1 = data["num1"]
            num2 = data["num2"]
            answer = data["answer"]
            
            # Build input string: "num1+num2=" (remove spaces from expression)
            expression = data.get("expression", f"{num1} + {num2}")
            input_str = expression.replace(" ", "") + "="  # e.g., "95822412+24942603="
            
            # Reverse target digit order: predict least significant digit first
            target_str = f" {str(answer)[::-1]}"
            full_text = input_str + target_str + "§"
            
            # Encode
            tokenized = self.tokenizer.encode(full_text)
            input_len = len(self.tokenizer.encode(input_str))
            
            # Labels should be shifted right by one position (GPT style)
            labels = []
            for i in range(len(tokenized)):
                if i < input_len - 1:
                    # Input part (except last position): no loss calculation
                    labels.append(-100)
                elif i < len(tokenized) - 1:
                    # Position input_len-1 and after: predict next token
                    labels.append(tokenized[i + 1])
                else:
                    # Last position (EOS): no need to predict next token
                    labels.append(-100)
            
            return {
                "input_ids": torch.tensor(tokenized, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "input_len": input_len,
                "ground_truth_num": answer
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing line: {line[:50]}... Error: {e}")
            return None
    
    def _preload_data(self, max_samples):
        """Preload data to memory (for evaluation scenarios)"""
        self.data = []
        samples_per_file = max_samples // len(self.data_files)
        remainder = max_samples % len(self.data_files)
        
        for file_idx, file_path in enumerate(self.data_files):
            current_samples = samples_per_file + (1 if file_idx < remainder else 0)
            count = 0
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if count >= current_samples:
                            break
                        sample = self._parse_jsonl_line(line)
                        if sample is not None:
                            self.data.append(sample)
                            count += 1
            except FileNotFoundError:
                print(f"Warning: File {file_path} not found, skipping")
                continue
        
        self.data_size = len(self.data)
    
    def _read_sample_from_file(self, file_path, line_offset=0):
        """Read sample from file at specified offset (for index-based access)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Skip previous lines
                for _ in range(line_offset):
                    line = f.readline()
                    if not line:
                        return None
                # Read target line
                line = f.readline()
                if not line:
                    return None
                return self._parse_jsonl_line(line)
        except (FileNotFoundError, IOError) as e:
            print(f"Error reading file {file_path} at offset {line_offset}: {e}")
            return None
    
    def _get_file_line_count(self, file_path):
        """Get file line count (for cyclic access)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except (FileNotFoundError, IOError):
            return 0
    
    def generate_sample(self, index):
        """Get sample by index (for evaluation scenarios)"""
        if self.data is not None:
            # Get from preloaded data
            return self.data[index % self.data_size]
        else:
            # Streaming mode: need to read from file
            # To support index-based access, we need to know line count of each file
            # Use simple round-robin allocation strategy
            if not hasattr(self, '_file_line_counts'):
                # Cache file line counts (calculate once to avoid repeated IO)
                self._file_line_counts = [self._get_file_line_count(f) for f in self.data_files]
                self._total_file_lines = sum(self._file_line_counts)
            
            if self._total_file_lines == 0:
                raise ValueError("All data files are empty")
            
            # Calculate which file and which line
            file_idx = 0
            line_offset = index % self._total_file_lines
            for i, line_count in enumerate(self._file_line_counts):
                if line_offset < line_count:
                    file_idx = i
                    break
                line_offset -= line_count
            
            # Read from corresponding file
            sample = self._read_sample_from_file(self.data_files[file_idx], line_offset)
            if sample is None:
                # If read fails, return first sample
                print("======Failed to read from corresponding file, returning first sample===============")
                sample = self._read_sample_from_file(self.data_files[0], 0)
            return sample

    def __iter__(self):
        """Iterator: stream data reading with cyclic support"""
        if self.data is not None:
            # Preload mode: cycle through memory data
            index = 0
            while True:
                yield self.data[index % self.data_size]
                index += 1
        else:
            # Streaming mode: cycle through file reading
            # Round-robin approach for multiple files
            file_handles = []
            file_line_counts = []
            
            # Cache file line counts (avoid repeated calculation)
            if not hasattr(self, '_cached_file_line_counts'):
                self._cached_file_line_counts = {}
            
            # Open all files
            for file_path in self.data_files:
                try:
                    f = open(file_path, 'r', encoding='utf-8')
                    file_handles.append(f)
                    # Use cached line count
                    if file_path not in self._cached_file_line_counts:
                        self._cached_file_line_counts[file_path] = self._get_file_line_count(file_path)
                    file_line_counts.append(self._cached_file_line_counts[file_path])
                except (FileNotFoundError, IOError) as e:
                    print(f"Warning: Cannot open file {file_path}: {e}")
                    file_handles.append(None)
                    file_line_counts.append(0)
            
            # Round-robin reading: read one line from each file, cycle through
            file_indices = list(range(len(file_handles)))
            current_file_idx = 0
            samples_yielded = 0
            
            try:
                while True:
                    # If total sample limit reached, restart (cycle)
                    if self.total_samples > 0 and samples_yielded >= self.total_samples:
                        # Reopen files (cyclic reading)
                        for f in file_handles:
                            if f is not None:
                                f.seek(0)
                        samples_yielded = 0
                    
                    # Try to read from current file
                    f = file_handles[current_file_idx]
                    if f is not None:
                        line = f.readline()
                        if line:
                            sample = self._parse_jsonl_line(line)
                            if sample is not None:
                                yield sample
                                samples_yielded += 1
                            # Move to next file
                            current_file_idx = (current_file_idx + 1) % len(file_handles)
                        else:
                            # File finished, restart
                            f.seek(0)
                            line = f.readline()
                            if line:
                                sample = self._parse_jsonl_line(line)
                                if sample is not None:
                                    yield sample
                                    samples_yielded += 1
                            current_file_idx = (current_file_idx + 1) % len(file_handles)
                    else:
                        # File unavailable, skip
                        current_file_idx = (current_file_idx + 1) % len(file_handles)
            finally:
                # Close all files
                for f in file_handles:
                    if f is not None:
                        f.close()
class CausalSelfAttention(nn.Module):
    """GPT-style causal self-attention (manual implementation, not using F.scaled_dot_product_attention)"""
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        # GPT style: use single linear layer to generate Q, K, V
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.size()
        # Generate Q, K, V in one pass (GPT style)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(config.d_model, dim=2)
        # Multi-head attention reshape: [B, T, C] -> [B, n_heads, T, d_head]
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Manual attention calculation (GPT style, not using F.scaled_dot_product_attention)
        # Compute attention scores: [B, n_heads, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.d_head ** 0.5))
        
        # Apply causal mask (lower triangular matrix)
        if mask is None:
            # Generate causal mask
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~causal_mask, float('-inf'))
        else:
            # If mask provided, use the provided mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            att = att.masked_fill(~mask, float('-inf'))
        
        # Softmax
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        # Apply attention to values
        y = att @ v  # [B, n_heads, T, d_head]
        
        # Concatenate multi-head results: [B, n_heads, T, d_head] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """GPT-style feedforward network"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """GPT-style Transformer block (Sandwich norm structure)"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = self.config.d_model

        # 辅助函数：统一从 nn 库获取对应的 Norm 层
        def get_norm():
            if self.config.norm_type == "LayerNorm":
                return nn.LayerNorm(d_model)
            elif self.config.norm_type == "RMSNorm":
                return nn.RMSNorm(d_model) 
            elif self.config.norm_type == "SimpleNorm":
                # SimpleNorm 通常指不带可学习参数的归一化
                return nn.LayerNorm(d_model, elementwise_affine=False)
            elif self.config.norm_type == "DeepNorm":
                # DeepNorm 结构上使用 LayerNorm，但配合特定的初始化和残差缩放
                return nn.LayerNorm(d_model)
            else:
                return nn.LayerNorm(d_model)
         # --- 初始化归一化层 ---
        
        # Pre-Norm 结构
        if self.config.norm_structure == "pre_norm":
            self.ln_1_pre = get_norm()
            self.ln_2_pre = get_norm()

        # Post-Norm 结构
        if self.config.norm_structure == "post_norm":
            self.ln_1_post = get_norm()
            self.ln_2_post = get_norm()

        # Sandwich Branch 结构 (仅在分支内部做两次 Norm)
        if self.config.norm_structure == "sandwich_branch":
            self.ln_1_pre = get_norm()
            self.ln_2_pre = get_norm()
            self.ln_1_post = get_norm()
            self.ln_2_post = get_norm()

        # Sandwich Dual 结构 (残差包在 Norm 内部)
        if self.config.norm_structure == "sandwich_dual":
            self.ln_1_pre = get_norm()
            self.ln_2_pre = get_norm()
            self.ln_1_post = get_norm()
            self.ln_2_post = get_norm()

        # 初始化组件
        self.attn = CausalSelfAttention(self.config)
        self.mlp = MLP(self.config)
        
    def forward(self, x):
        # --- 1. Pre-Norm ---
        if self.config.norm_structure == "pre_norm":
            x = x + self.attn(self.ln_1_pre(x))
            x = x + self.mlp(self.ln_2_pre(x))
        # --- 2. Post-Norm (包含 DeepNorm 特殊缩放逻辑) ---
        elif self.config.norm_structure == "post_norm":
            if self.config.norm_type == "DeepNorm":
                # DeepNorm 公式: LN(x * alpha + f(x))
                alpha = getattr(self.config, "deepnorm_alpha", 1.0)
                x = self.ln_1_post(x * alpha + self.attn(x))
                x = self.ln_2_post(x * alpha + self.mlp(x))
            else:
                x = self.ln_1_post(x + self.attn(x))
                x = self.ln_2_post(x + self.mlp(x))
        # --- 3. Sandwich Branch (分支内双重 Norm) ---
        elif self.config.norm_structure == "sandwich_branch":
            x = x + self.ln_1_post(self.attn(self.ln_1_pre(x)))
            x = x + self.ln_2_post(self.mlp(self.ln_2_pre(x)))
        # --- 4. Sandwich Dual (LN_post(x + Sublayer(LN_pre(x)))) ---
        elif self.config.norm_structure == "sandwich_dual":
            x = self.ln_1_post(x + self.attn(self.ln_1_pre(x)))
            x = self.ln_2_post(x + self.mlp(self.ln_2_pre(x)))

        return x

class LoopedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Embeddings
        self.token_embedding = nn.Embedding(len(config.vocab), config.d_model)
        self.position_embedding = nn.Embedding(config.max_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        
        # Prelude block (optional, single block)
        if getattr(config, "prelude", False):
            self.prelude = Block(config)
        else:
            self.prelude = None

        # Looped Blocks (k_layers)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.k_layers)])

        # Coda block (optional, single block)
        if getattr(config, "coda", False):
            self.coda = Block(config)
        else:
            self.coda = None

        # Output layer
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, len(config.vocab), bias=False)

        # Weight sharing (optional, does not affect multi-GPU)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize
        self.apply(self._init_weights)

        # 添加 DeepNorm 特殊初始化
        if self.config.norm_type == "DeepNorm":
            self._deepnorm_init()  # 需要定义这个方法

    def _deepnorm_init(self):
        """DeepNorm 的特殊初始化"""
        alpha = getattr(self.config, "deepnorm_alpha", 1.0)
        
        for layer in self.layers:
            # 缩放注意力输出投影
            if hasattr(layer.attn, 'c_proj'):
                gain = (0.5 * alpha) ** 0.5
                nn.init.xavier_normal_(layer.attn.c_proj.weight, gain=gain)
            
            # 缩放 MLP 输出投影
            if hasattr(layer.mlp, 'c_proj'):
                gain = (0.5 * alpha) ** 0.5
                nn.init.xavier_normal_(layer.mlp.c_proj.weight, gain=gain)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        elif isinstance(module, nn.LayerNorm):
           
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.RMSNorm):
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None, num_loops=None):
        """
        Args:
            idx: Input token ids
            targets: Target labels (provided during training)
            num_loops: Loop count (if None, randomly sampled during training, use config.l_loops during evaluation)
        """
        B, T = idx.size()
        device = idx.device
        
        # GPT-style positional encoding: generate index for each position
        # If sequence length exceeds max_len, use modulo or truncate (GPT typically truncates)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        # If exceeds max_len, use last position encoding (common GPT practice)
        pos = torch.clamp(pos, 0, self.config.max_len - 1)
        
        # GPT-style embedding: token embedding + position embedding
        tok_emb = self.token_embedding(idx)  # [B, T, d_model]
        pos_emb = self.position_embedding(pos)  # [T, d_model]
        # Expand positional encoding to match batch dimension: [T, d_model] -> [1, T, d_model] -> [B, T, d_model]
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        x = tok_emb + pos_emb
        x = self.dropout(x)
        
        # Prelude (non-looped, executed once)
        if self.prelude:
            x = self.prelude(x)

        # Determine loop count
        if num_loops is None:
            if targets is not None and config.random_loop is True:
                # During training: randomly sample loop count from log-normal distribution
                num_loops = sample_random_loops(self.config)
            else:
                # During evaluation: use fixed loop count
                num_loops = self.config.l_loops
        
        # Looped structure: pass through transformer layers multiple times (random loop count)
        for _ in range(num_loops):
            for layer in self.layers:
                x = layer(x)
        
        # Coda (non-looped, executed once)
        if self.coda:
            x = self.coda(x)

        # GPT-style output: LayerNorm + Linear head (using result from l_loops iterations)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Critical fix: average loss across GPUs to ensure scalar output
            # Note: must explicitly specify ignore_index=-100 to ensure input part loss is ignored
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,  # Ignore input part labels
                reduction='mean'  # Explicitly specify per-sample average, output scalar (default is also mean, but need to be explicit)
            )
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, num_loops=None):
        """
        Generation logic for evaluation:
        - Remove EOS stopping (EOS only used for sample marking during training, not needed for evaluation)
        - Fixed generation by max_new_tokens (true result length) to ensure complete output at once
        - Use greedy decoding only (paper does not mention sampling, avoid randomness affecting accuracy)
        - Use fixed loop count (config.l_loops) for evaluation
        """
        if num_loops is None:
            num_loops = self.config.l_loops
        idx = idx.to(config.device)
        # Strictly generate by max_new_tokens, no early termination (ensure complete output at once)
        for step in range(max_new_tokens):
            # Truncate to model maximum sequence length
            idx_cond = idx[:, -self.config.max_len:]
            # Use fixed loop count during evaluation
            logits, _ = self(idx_cond, targets=None, num_loops=num_loops)  # Ignore loss (not needed during generation)
            # Take logits of last token (causal language model)
            logits = logits[:, -1, :]
            # Debug: print logits distribution for first few steps
            if step < 3:
                probs = F.softmax(logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs, k=5, dim=-1)
                top_k_tokens = [tokenizer.i2c[idx.item()] for idx in top_k_indices[0]]
                print(f"[Debug Step {step}] Top 5 tokens: {list(zip(top_k_tokens, top_k_probs[0].cpu().tolist()))}")
            # Greedy decoding (use greedy for evaluation to avoid noise from sampling)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            # Debug: print generated token
            if step < 3:
                generated_token = tokenizer.i2c[idx_next[0, 0].item()]
                print(f"[Debug Step {step}] Generated token: '{generated_token}' (idx={idx_next[0, 0].item()})")
            # Concatenate new token, don't check EOS
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    @torch.no_grad()
    def get_embeddings_per_loop(self, idx, max_loops=None):
        """
        获取每次loop后的embedding
        Args:
            idx: 输入token ids [B, T]
            max_loops: 最大循环次数，如果为None则使用config.l_loops
        Returns:
            embeddings_list: 每次loop后的embedding列表，每个元素为 [B, T, d_model]
        """
        B, T = idx.size()
        device = idx.device
        
        # GPT-style positional encoding: generate index for each position
        # If sequence length exceeds max_len, use modulo or truncate (GPT typically truncates)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        # If exceeds max_len, use last position encoding (common GPT practice)
        pos = torch.clamp(pos, 0, self.config.max_len - 1)
        
        # GPT-style embedding: token embedding + position embedding
        tok_emb = self.token_embedding(idx)  # [B, T, d_model]
        pos_emb = self.position_embedding(pos)  # [T, d_model]
        # Expand positional encoding to match batch dimension: [T, d_model] -> [1, T, d_model] -> [B, T, d_model]
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        x = tok_emb + pos_emb
        x = self.dropout(x)
        
        # Prelude layer: single block before looped layers
        if self.prelude is not None:
            x = self.prelude(x)

        
        # 保存prelude后的embedding（loop 0）
        embeddings_list = []
        
        # 确定循环次数
        num_loops = max_loops if max_loops is not None else self.config.l_loops
        
        # 循环结构：每次loop后保存embedding
        for loop_idx in range(num_loops):
            for layer in self.layers:
                x = layer(x)
            # 保存每次loop后的embedding
            embeddings_list.append(x.clone().cpu())
        
        # Coda layer: single block after looped layers
        if self.coda is not None:
            x = self.coda(x)
        # 保存coda后的embedding（最终embedding）
            embeddings_list.append(x.clone().cpu())
        
        return embeddings_list

def visualize_embedding_changes(model, n_operands=16, sample_idx=0, max_loops=10, save_path=None):
    """
    可视化不同loop次数下embedding的变化
    Args:
        model: 训练好的模型
        n_operands: 操作数数量，默认16
        sample_idx: 样本索引，默认0
        max_loops: 最大循环次数，默认10
        save_path: 保存路径，如果为None则不保存
    """
    model.eval()
    model.to(config.device)
    
    # 加载n=16的测试样本
    print(f"Loading sample {sample_idx} for n={n_operands}...")
    ds = AdditionDataset(tokenizer, config, mode='test', specific_n=n_operands, total_samples=1000, data_dir='data')
    sample = ds.generate_sample(sample_idx)
    
    # 获取输入
    input_str_len = sample["input_len"]
    input_ids = sample["input_ids"][:input_str_len].unsqueeze(0).to(config.device)
    
    # 解码输入序列以便显示
    input_tokens = [tokenizer.i2c[idx.item()] for idx in input_ids[0]]
    input_str = tokenizer.decode(input_ids[0].tolist())
    
    print(f"Input: {input_str}")
    print(f"Input tokens: {input_tokens}")
    print(f"Sequence length: {len(input_tokens)}")
    
    # 获取每次loop后的embedding
    print(f"Extracting embeddings for {max_loops} loops...")
    embeddings_list = model.get_embeddings_per_loop(input_ids, max_loops=max_loops)
    
    print(f"Got {len(embeddings_list)} embedding snapshots (including initial)")
    
    # 转换为numpy数组
    # embeddings_list[i] shape: [1, T, d_model]
    num_loops = len(embeddings_list) - 1  # 减去初始embedding
    seq_len = embeddings_list[0].shape[1]
    d_model = embeddings_list[0].shape[2]
    
    # 方法1: 计算每个token在每个loop后的embedding的L2范数
    embedding_norms = np.zeros((num_loops + 1, seq_len))  # +1 for initial
    for loop_idx, emb in enumerate(embeddings_list):
        emb_np = emb[0].numpy()  # [T, d_model]
        embedding_norms[loop_idx] = np.linalg.norm(emb_np, axis=1)
    
    # 方法2: 计算每个token的embedding相对于初始embedding的变化（余弦相似度或L2距离）
    initial_emb = embeddings_list[0][0].numpy()  # [T, d_model]
    embedding_changes = np.zeros((num_loops, seq_len))
    for loop_idx in range(1, len(embeddings_list)):
        current_emb = embeddings_list[loop_idx][0].numpy()  # [T, d_model]
        # 计算L2距离
        diff = current_emb - initial_emb
        embedding_changes[loop_idx - 1] = np.linalg.norm(diff, axis=1)
    
    # 方法3: 计算相邻loop之间的变化
    embedding_deltas = np.zeros((num_loops, seq_len))
    for loop_idx in range(1, len(embeddings_list)):
        prev_emb = embeddings_list[loop_idx - 1][0].numpy()
        curr_emb = embeddings_list[loop_idx][0].numpy()
        diff = curr_emb - prev_emb
        embedding_deltas[loop_idx - 1] = np.linalg.norm(diff, axis=1)
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(max(16, seq_len * 0.6), 12))
    
    # 直接使用token字符作为标签
    # 如果序列太长，只显示部分标签
    if seq_len > 50:
        # 长序列：只显示部分标签
        step = max(1, seq_len // 30)
        xticklabels = [input_tokens[i] if i % step == 0 else '' for i in range(seq_len)]
    else:
        # 短序列：显示所有token
        xticklabels = input_tokens
    
    # 图1: Embedding L2范数随loop的变化
    sns.heatmap(embedding_norms, 
                xticklabels=xticklabels,
                yticklabels=[f'Loop {i}' if i > 0 else 'Initial' for i in range(num_loops + 1)],
                cmap='viridis', 
                ax=axes[0],
                cbar_kws={'label': 'L2 Norm'})
    axes[0].set_title(f'Embedding L2 Norm Across Loops (n={n_operands}, Sample {sample_idx})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Token Position', fontsize=12)
    axes[0].set_ylabel('Loop Iteration', fontsize=12)
    axes[0].tick_params(axis='x', rotation=90, labelsize=8)
    
    # 图2: 相对于初始embedding的变化
    sns.heatmap(embedding_changes,
                xticklabels=xticklabels,
                yticklabels=[f'Loop {i+1}' for i in range(num_loops)],
                cmap='plasma',
                ax=axes[1],
                cbar_kws={'label': 'L2 Distance from Initial'})
    axes[1].set_title(f'Embedding Change from Initial (n={n_operands}, Sample {sample_idx})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Token Position', fontsize=12)
    axes[1].set_ylabel('Loop Iteration', fontsize=12)
    axes[1].tick_params(axis='x', rotation=90, labelsize=8)
    
    # 图3: 相邻loop之间的变化（delta）
    sns.heatmap(embedding_deltas,
                xticklabels=xticklabels,
                yticklabels=[f'Loop {i+1}' for i in range(num_loops)],
                cmap='coolwarm',
                ax=axes[2],
                cbar_kws={'label': 'L2 Delta'})
    axes[2].set_title(f'Embedding Delta Between Adjacent Loops (n={n_operands}, Sample {sample_idx})', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Token Position', fontsize=12)
    axes[2].set_ylabel('Loop Iteration', fontsize=12)
    axes[2].tick_params(axis='x', rotation=90, labelsize=8)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.savefig(f'embedding_heatmap_n{n_operands}_sample{sample_idx}.png', dpi=300, bbox_inches='tight')
        print(f"Figure saved to: embedding_heatmap_n{n_operands}_sample{sample_idx}.png")
    
    plt.close()
    
    # 打印统计信息
    print("\n=== Embedding Statistics ===")
    print(f"Initial embedding norm - Mean: {embedding_norms[0].mean():.4f}, Std: {embedding_norms[0].std():.4f}")
    print(f"Final embedding norm - Mean: {embedding_norms[-1].mean():.4f}, Std: {embedding_norms[-1].std():.4f}")
    print(f"Total change from initial - Mean: {embedding_changes[-1].mean():.4f}, Max: {embedding_changes[-1].max():.4f}")
    print(f"Average delta per loop - Mean: {embedding_deltas.mean():.4f}, Std: {embedding_deltas.std():.4f}")
    
    return {
        'embedding_norms': embedding_norms,
        'embedding_changes': embedding_changes,
        'embedding_deltas': embedding_deltas,
        'input_tokens': input_tokens,
        'input_str': input_str
    }

def save_embedding_data(embeddings_list, save_path):
    """
    将模型生成的 embeddings_list 保存到硬盘。
    embeddings_list: list of tensors, 每个形状通常是 [1, T, d_model]
    """
    # 1. 数据清洗：转到CPU，去掉梯度，转为纯Tensor列表
    clean_list = []
    for emb in embeddings_list:
        # 确保是 CPU tensor，且没有梯度信息
        clean_emb = emb.detach().cpu()
        clean_list.append(clean_emb)
    
    # 2. 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 3. 保存
    torch.save(clean_list, save_path)
    print(f"Embeddings 已保存至: {save_path}")

def visualize_embedding_trajectories(model, n_operands=16, sample_idx=0, max_loops=10, 
                                     n_components=2, save_path=None, selected_tokens=None):
    """
    可视化embedding矩阵展开成向量后的轨迹
    将每个loop的embedding矩阵 [1, T, d_model] 展开成向量 [T*d_model]，然后可视化这个向量的轨迹
    Args:
        model: 训练好的模型
        n_operands: 操作数数量，默认16
        sample_idx: 样本索引，默认0
        max_loops: 最大循环次数，默认10
        n_components: 降维后的维度，默认2（2D可视化）
        save_path: 保存路径，如果为None则不保存
        selected_tokens: 已废弃，保留以兼容旧代码
    """
    model.eval()
    model.to(config.device)
    
    # 加载测试样本
    print(f"Loading sample {sample_idx} for n={n_operands}...")
    ds=AdditionDataset(tokenizer, config, mode='test', total_samples=1000, data_dir='data')
    # ds = AdditionDataset(tokenizer, config, mode='test', specific_n=n_operands, total_samples=1000, data_dir='data')
    sample = ds.generate_sample(sample_idx)
    
    # 获取输入
    input_str_len = sample["input_len"]
    input_ids = sample["input_ids"][:input_str_len].unsqueeze(0).to(config.device)
    
    # 解码输入序列以便显示
    input_tokens = [tokenizer.i2c[idx.item()] for idx in input_ids[0]]
    input_str = tokenizer.decode(input_ids[0].tolist())
    
    print(f"Input: {input_str}")
    print(f"Input tokens: {input_tokens}")
    print(f"Sequence length: {len(input_tokens)}")
    
    # 获取每次loop后的embedding
    print(f"Extracting embeddings for {max_loops} loops...")
    output_dir = os.path.dirname(save_path) 
    pt_save_path = os.path.join(output_dir, "emb.pt")
    embeddings_list = model.get_embeddings_per_loop(input_ids, max_loops=max_loops)
    save_embedding_data(embeddings_list, save_path=pt_save_path)
    print(f"Got {len(embeddings_list)} embedding snapshots (including initial)")
    
    # 转换为numpy数组并准备PCA
    # embeddings_list[i] shape: [1, T, d_model]
    num_loops = len(embeddings_list) - 1
    seq_len = embeddings_list[0].shape[1]
    d_model = embeddings_list[0].shape[2]
    flattened_dim = seq_len * d_model
    
    print(f"Embedding shape per loop: [1, {seq_len}, {d_model}]")
    print(f"Flattened vector dimension: {flattened_dim}")
    
    # 将每个loop的embedding矩阵展开成向量
    flattened_vectors = []
    for loop_idx, emb in enumerate(embeddings_list):
        emb_np = emb[0].numpy()  # [T, d_model]
        # 展平成向量: [T, d_model] -> [T*d_model]
        flattened = emb_np.flatten()  # [T*d_model]
        flattened_vectors.append(flattened)
    
    # 转换为numpy数组: [num_loops+1, T*d_model]
    flattened_vectors = np.array(flattened_vectors)
    print(f"Flattened vectors shape: {flattened_vectors.shape}")
    
    # 对所有展平后的向量进行PCA拟合
    print(f"Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    pca.fit(flattened_vectors)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # 对每个loop的展平向量进行降维
    trajectory = pca.transform(flattened_vectors)  # [num_loops+1, n_components]
    print(f"Trajectory shape: {trajectory.shape}")
    
    # 可视化
    if n_components == 2:
        # 2D可视化
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        x_coords = trajectory[:, 0]
        y_coords = trajectory[:, 1]
        
        # 绘制轨迹线
        ax.plot(x_coords, y_coords, 'o-', color='blue', 
               linewidth=2.5, markersize=8, alpha=0.8, 
               label='Feature Vector Trajectory')
        
        # 标记起点（初始embedding）
        ax.scatter(x_coords[0], y_coords[0], s=200, color='green', 
                  marker='s', edgecolors='black', linewidths=2, zorder=5,
                  label='Start (Loop 0)')
        
        # 标记终点（最终embedding）
        ax.scatter(x_coords[-1], y_coords[-1], s=200, color='red', 
                  marker='*', edgecolors='black', linewidths=2, zorder=5,
                  label=f'End (Loop {num_loops})')
        
        # 添加箭头指示方向（每隔几个点画一个箭头）
        arrow_step = max(1, len(trajectory) // 15)
        for i in range(0, len(trajectory) - 1, arrow_step):
            dx = x_coords[i+1] - x_coords[i]
            dy = y_coords[i+1] - y_coords[i]
            if abs(dx) > 0.001 or abs(dy) > 0.001:  # 只画有意义的箭头
                ax.arrow(x_coords[i], y_coords[i], dx*0.8, dy*0.8,
                        head_width=0.1, head_length=0.1, fc='blue', 
                        ec='blue', alpha=0.6, length_includes_head=True)
        
        # 每10个位置标记一个*
        for i in range(10, len(trajectory), 10):
            ax.scatter(x_coords[i], y_coords[i], s=200, color='orange', 
                      marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        
        # 在关键点标注loop编号
        for i in [0, len(trajectory)//2, len(trajectory)-1]:
            ax.annotate(f'L{i}', (x_coords[i], y_coords[i]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        ax.set_ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
        ax.set_title(f'Flattened Feature Vector Trajectory in PCA Space\n(n={n_operands}, Sample {sample_idx}, {max_loops} loops, Vector Dim={flattened_dim})', 
                    fontsize=14, fontweight='bold')
        ax.grid(False, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
    elif n_components == 3:
        # 3D可视化
        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            pass  # 新版本matplotlib不需要显式导入
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x_coords = trajectory[:, 0]
        y_coords = trajectory[:, 1]
        z_coords = trajectory[:, 2]
        
        # 绘制轨迹线
        ax.plot(x_coords, y_coords, z_coords, 'o-', color='blue', 
               linewidth=2.5, markersize=8, alpha=0.8,
               label='Feature Vector Trajectory')
        
        # 标记起点和终点
        ax.scatter(x_coords[0], y_coords[0], z_coords[0], s=200, color='green', 
                  marker='s', edgecolors='black', linewidths=2, zorder=5,
                  label='Start (Loop 0)')
        ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], s=200, color='red', 
                  marker='*', edgecolors='black', linewidths=2, zorder=5,
                  label=f'End (Loop {num_loops})')
        
        # 每10个位置标记一个*
        for i in range(10, len(trajectory), 10):
            ax.scatter(x_coords[i], y_coords[i], z_coords[i], s=200, color='orange', 
                      marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        
        # 在关键点标注loop编号
        for i in [0, len(trajectory)//2, len(trajectory)-1]:
            ax.text(x_coords[i], y_coords[i], z_coords[i], f'L{i}', 
                   fontsize=10, fontweight='bold')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontsize=12)
        ax.set_title(f'Flattened Feature Vector Trajectory in 3D PCA Space\n(n={n_operands}, Sample {sample_idx}, {max_loops} loops, Vector Dim={flattened_dim})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory figure saved to: {save_path}")
    else:
        plt.savefig(f'embedding_trajectories_n{n_operands}_sample{sample_idx}_loops{max_loops}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Trajectory figure saved to: embedding_trajectories_n{n_operands}_sample{sample_idx}_loops{max_loops}.png")
    
    plt.close()
    
    return {
        'trajectory': trajectory,
        'pca': pca,
        'input_tokens': input_tokens,
        'flattened_dim': flattened_dim
    }

# ==========================================
# 8. 计算雅可比谱半径
# ==========================================
def compute_jacobian_spectral_radius(model, n_operands=16, sample_idx=0, loop_idx=None):
    """
    计算最后一个loop得到的embedding相对于循环的block的雅可比谱半径
    
    雅可比矩阵：J = ∂y/∂x，其中：
    - x是某个loop开始时的embedding [B, T, d_model]
    - y是经过k_layers个Block后的embedding [B, T, d_model]
    - 一个loop = 通过k_layers个Block
    
    谱半径 = max(|λ|)，其中λ是雅可比矩阵的特征值
    
    Args:
        model: 训练好的模型
        n_operands: 操作数数量，默认16
        sample_idx: 样本索引，默认0
        loop_idx: 计算哪个loop的雅可比谱半径，如果为None则计算最后一个loop
    Returns:
        spectral_radius: 雅可比矩阵的谱半径（标量）
        max_eigenvalue: 最大特征值（复数）
        jacobian_shape: 雅可比矩阵的形状
    """
    model.eval()
    model.to(config.device)
    
    # 加载测试样本
    print(f"Loading sample {sample_idx} for n={n_operands}...")
    #ds = AdditionDataset(tokenizer, config, mode='test', specific_n=n_operands, total_samples=1000, data_dir='data')
    ds = AdditionDataset(tokenizer, config, mode='test', total_samples=1000, data_dir='data')
    sample = ds.generate_sample(sample_idx)
    
    # 获取输入
    input_str_len = sample["input_len"]
    input_ids = sample["input_ids"][:input_str_len].unsqueeze(0).to(config.device)
    
    # 解码输入序列以便显示
    input_tokens = [tokenizer.i2c[idx.item()] for idx in input_ids[0]]
    input_str = tokenizer.decode(input_ids[0].tolist())
    
    print(f"Input: {input_str}")
    print(f"Sequence length: {len(input_tokens)}")
    
    # 获取embedding列表以确定要计算的loop
    print(f"Extracting embeddings to determine loop index...")
    embeddings_list = model.get_embeddings_per_loop(input_ids, max_loops=config.l_loops)
    
    # 确定要计算的loop索引
    if loop_idx is None:
        loop_idx = len(embeddings_list) - 2  # 最后一个loop（-2因为embeddings_list包含初始embedding）
    
    if loop_idx < 0 or loop_idx >= len(embeddings_list) - 1:
        raise ValueError(f"Invalid loop_idx: {loop_idx}. Must be in [0, {len(embeddings_list)-2}]")
    
    print(f"Computing Jacobian for loop {loop_idx}...")
    
    # 获取该loop开始时的embedding（需要requires_grad=True）
    x_start = embeddings_list[loop_idx].to(config.device).clone().detach().requires_grad_(True)
    # x_start shape: [B, T, d_model]
    B, T, d_model = x_start.shape
    
    # 定义一个函数，表示一个loop的变换（通过k_layers个Block）
    def loop_transform(x):
        """一个loop的变换：通过k_layers个Block"""
        y = x
        for layer in model.layers:
            y = layer(y)
        return y
    
    # 计算雅可比矩阵
    # 将embedding展平为向量以便计算雅可比矩阵
    x_flat = x_start.view(-1)  # [B*T*d_model]
    output_flat_shape = (B, T, d_model)
    
    print(f"Computing Jacobian matrix...")
    print(f"Input shape: {x_start.shape}, Flattened size: {x_flat.shape[0]}")
    print(f"Jacobian matrix will be: [{x_flat.shape[0]}, {x_flat.shape[0]}]")
    
    # 使用torch.autograd.functional.jacobian计算雅可比矩阵
    # 注意：这可能需要大量内存，对于大矩阵可能不适用
    try:
        from torch.autograd.functional import jacobian
        
        def loop_transform_flat(x_flat_in):
            """展平版本的loop变换"""
            x_in = x_flat_in.view(B, T, d_model)
            y = loop_transform(x_in)
            return y.view(-1)
        
        # 计算雅可比矩阵
        print("Computing full Jacobian matrix (this may take a while and use significant memory)...")
        J = jacobian(loop_transform_flat, x_flat)  # [B*T*d_model, B*T*d_model]
        jacobian_shape = J.shape
        
        print(f"Jacobian matrix shape: {jacobian_shape}")
        print(f"Computing eigenvalues (this may take a while)...")
        
        # 转换为numpy以便计算特征值（PyTorch的eig可能不稳定）
        J_np = J.detach().cpu().numpy()
        
        # 计算特征值
        eigenvalues = np.linalg.eigvals(J_np)
        
        # 计算谱半径（最大特征值的绝对值）
        max_eigenvalue = eigenvalues[np.argmax(np.abs(eigenvalues))]
        spectral_radius = np.abs(max_eigenvalue)
        
    except RuntimeError as e:
        if "out of memory" in str(e) or "memory" in str(e).lower():
            print(f"Warning: Direct Jacobian computation failed due to memory constraints.")
            print(f"Using power iteration method with JVP to estimate spectral radius...")
            
            # 使用幂迭代法估计最大特征值（不需要显式构造雅可比矩阵）
            # 幂迭代：v_{k+1} = J * v_k / ||J * v_k||
            # 使用雅可比-向量乘积（JVP）高效计算 J * v
            
            from torch.autograd.functional import jvp
            
            def loop_transform_flat(x_flat_in):
                """展平版本的loop变换"""
                x_in = x_flat_in.view(B, T, d_model)
                y = loop_transform(x_in)
                return y.view(-1)
            
            # 初始化随机向量
            v = torch.randn(B * T * d_model, device=config.device, dtype=x_start.dtype)
            v = v / torch.norm(v)
            
            num_iterations = 50  # 幂迭代次数
            tolerance = 1e-6
            
            print(f"Running power iteration ({num_iterations} iterations)...")
            
            # 确保x_flat有requires_grad（用于JVP）
            x_flat_for_jvp = x_flat.clone().detach().requires_grad_(True)
            
            for i in range(num_iterations):
                # 使用JVP计算 J * v（高效，不需要显式构造雅可比矩阵）
                # jvp(func, inputs, v) 计算雅可比-向量乘积
                _, Jv = jvp(loop_transform_flat, (x_flat_for_jvp,), (v,))
                
                # 归一化
                v_norm = torch.norm(Jv)
                if v_norm < 1e-10:
                    print(f"Warning: Jv norm is too small at iteration {i+1}, stopping.")
                    break
                v_new = Jv / v_norm
                
                # 检查收敛
                diff = torch.norm(v_new - v)
                if diff < tolerance:
                    print(f"Power iteration converged at iteration {i+1}")
                    break
                
                v = v_new
                
                if (i + 1) % 10 == 0:
                    print(f"  Iteration {i+1}/{num_iterations}, diff: {diff:.6e}")
            
            # 计算最大特征值的估计：λ_max ≈ v^T * J * v
            _, Jv_final = jvp(loop_transform_flat, (x_flat_for_jvp,), (v,))
            lambda_approx = (v * Jv_final).sum().item()
            
            spectral_radius = abs(lambda_approx)
            max_eigenvalue = lambda_approx
            jacobian_shape = (B * T * d_model, B * T * d_model)
            
            print(f"Estimated spectral radius using power iteration: {spectral_radius:.6f}")
        else:
            raise e
    
    print(f"\n=== Jacobian Spectral Radius ===")
    print(f"Loop index: {loop_idx}")
    print(f"Embedding shape: {x_start.shape}")
    print(f"Jacobian matrix shape: {jacobian_shape}")
    print(f"Maximum eigenvalue: {max_eigenvalue}")
    print(f"Spectral radius: {spectral_radius:.6f}")
    
    return spectral_radius, max_eigenvalue, jacobian_shape

# ==========================================
# 9. 计算最后一次迭代的embedding的F范数
# ==========================================
def compute_final_embedding_frobenius_norm(model, n_operands=16, sample_idx=0, max_loops=None):
    """
    计算最后一次迭代的embedding的Frobenius范数（F范数）
    Args:
        model: 训练好的模型
        n_operands: 操作数数量，默认16
        sample_idx: 样本索引，默认0
        max_loops: 最大循环次数，如果为None则使用config.l_loops
    Returns:
        frobenius_norm: 最后一次迭代的embedding的F范数（标量）
        embedding_shape: embedding的形状 [B, T, d_model]
    """
    model.eval()
    model.to(config.device)
    
    # 加载测试样本
    print(f"Loading sample {sample_idx} for n={n_operands}...")
    ds = AdditionDataset(tokenizer, config, mode='test', specific_n=n_operands, total_samples=1000, data_dir='data')
    sample = ds.generate_sample(sample_idx)
    
    # 获取输入
    input_str_len = sample["input_len"]
    input_ids = sample["input_ids"][:input_str_len].unsqueeze(0).to(config.device)
    
    # 解码输入序列以便显示
    input_tokens = [tokenizer.i2c[idx.item()] for idx in input_ids[0]]
    input_str = tokenizer.decode(input_ids[0].tolist())
    
    print(f"Input: {input_str}")
    print(f"Sequence length: {len(input_tokens)}")
    
    # 获取每次loop后的embedding
    num_loops = max_loops if max_loops is not None else config.l_loops
    print(f"Extracting embeddings for {num_loops} loops...")
    embeddings_list = model.get_embeddings_per_loop(input_ids, max_loops=num_loops)
    
    # 获取最后一次迭代的embedding
    final_embedding = embeddings_list[-1]  # [B, T, d_model]
    embedding_shape = final_embedding.shape
    
    # 计算Frobenius范数
    # F范数：||A||_F = sqrt(sum_i sum_j |a_ij|^2)
    # 对于形状为[B, T, d_model]的tensor，计算整个tensor的F范数
    frobenius_norm = torch.norm(final_embedding, p='fro').item()
    
    print(f"\n=== Final Embedding Frobenius Norm ===")
    print(f"Embedding shape: {embedding_shape}")
    print(f"Frobenius norm: {frobenius_norm:.6f}")
    print(f"Number of loops: {num_loops}")
    
    return frobenius_norm, embedding_shape

# ==========================================
# 9. 主函数（加载模型并测试）
# ==========================================
if __name__ == "__main__":
    # 固定随机种子（确保复现性）
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 加载模型
    model_path = ""
    print(f"Loading model from: {model_path}")
    model = LoopedTransformer(config)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    print("Model loaded successfully.")
    
    # 可视化embedding变化（n=16的case）
    # print("\n=== Visualizing Embedding Changes ===")
    # visualize_embedding_changes(
    #     model, 
    #     n_operands=16, 
    #     sample_idx=0, 
    #     max_loops=10,
    #     save_path="post_embedding_heatmap_n16_sample0.png"
    # )
    
    #可视化embedding轨迹（降维）
    print("\n=== Visualizing Embedding Trajectories ===")
    save_dir = "/lamda12/yangxw/codes/circu_model/train_addition/newlogs/training_results_k1_L4_20260128_001900"
    n_operands = 2
    sample_idx = 0
    exp_tag = (
        f"n{n_operands}_sample{sample_idx}"
        f"{config.norm_structure}_"
        f"{config.norm_type}"
    )
    visualize_embedding_trajectories(
        model,
        n_operands=n_operands,
        sample_idx=sample_idx,
        max_loops=None,  
        n_components=2,
        save_path=os.path.join(save_dir, f"post_embedding_trajectories_{exp_tag}.png"),
        selected_tokens=None  # None表示显示所有token，也可以指定特定token索引
    )
    
    # print("\n=== Computing Final Embedding Frobenius Norm ===")
    # frobenius_norm, embedding_shape = compute_final_embedding_frobenius_norm(
    #     model,
    #     n_operands=16,
    #     sample_idx=0,
    #     max_loops=config.l_loops  # 使用config中的l_loops值
    # )

    # print("\n=== Computing Jacobian Spectral Radius ===")
    # spectral_radius, max_eigenvalue, jacobian_shape = compute_jacobian_spectral_radius(
    #     model,
    #     n_operands=2,
    #     sample_idx=0,
    #     loop_idx=600  # None表示计算最后一个loop
    # )

    
    print("\nVisualization completed.")