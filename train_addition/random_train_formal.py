import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup
import numpy as np
import random
from tqdm import tqdm
import json
import csv
import os
from datetime import datetime

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
    norm_structure = "sandwich_branch"
    # Norm types
    # LayerNorm RMSNorm DeepNorm SimpleNorm
    norm_type = "LayerNorm"

    # L2 limit
    l2_limit = False
    l2_weight = 1
    # if DeepNorm:
    deepnorm_alpha = 1

    # Looped settings: k_layers * l_loops (effective depth)
    k_layers = 1
    l_loops = 4    # Fixed loop count 

    #对数正态 泊松分布 均匀分布
    #"log-norm" "poisson" "uniform"
    random_distribution = "log_norm"

    # random loop
    # (log-normal distribution)
    # Log-normal distribution peak (mode) = e^(μ - σ²)
    random_loop = True
    random_loop_mu = 2
    random_loop_sigma = 0.7
    random_loop_min = 1      # 最小循环次数
    random_loop_max = 100  # 最大循环次数

    poisson_lambda = 5
    
    # prelude code
    prelude = False
    coda = False

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

# ==========================================
# 2. Dataset Construction (Ensure data is generated on CPU)
# ==========================================
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

def collate_fn(batch):
    # Dynamic padding (executed on CPU to avoid multi-GPU data confusion)
    max_len = max([len(x["input_ids"]) for x in batch])
    # print(max_len)
    input_ids = []
    labels = []
    
    for item in batch:
        curr_len = len(item["input_ids"])
        pad_len = max_len - curr_len
        # Concatenate PAD (right pad)
        ids = torch.cat([
            item["input_ids"],
            torch.tensor([config.pad_token_id] * pad_len, dtype=torch.long)
        ])
        input_ids.append(ids)
        
        lbs = torch.cat([
            item["labels"],
            torch.tensor([-100] * pad_len, dtype=torch.long)
        ])
        labels.append(lbs)
    
    return {
        "input_ids": torch.stack(input_ids),  # CPU tensor
        "labels": torch.stack(labels)        # CPU tensor
    }

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
def sample_poisson_loops(config):
    """
    从泊松分布中采样循环次数。

    使用泊松分布，然后将结果限制在 [min, max] 范围内。
    分布的峰值（期望值）由 config.poisson_lambda 控制。
    """
    # 从泊松分布中采样
    # lam (lambda) 是分布的期望值或峰值
    poisson_sample = np.random.poisson(
        lam=config.poisson_lambda
    )

    # 限制范围
    num_loops = max(config.random_loop_min, min(config.random_loop_max, poisson_sample))

    return num_loops

def sample_uniform_loops(config):
    """
    从均匀分布中采样循环次数。

    在 [min, max] 闭区间内，每个整数被抽取的概率都相等。
    """
    # randint 的第三个参数是 high，它是上限（不包含）
    # 所以我们需要用 config.random_loop_max + 1
    num_loops = np.random.randint(
        low=config.random_loop_min,
        high=config.random_loop_max + 1
    )
    return num_loops
# ==========================================
# 3. Model Definition (Reference GPT implementation, retain looped structure)
# ==========================================
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
            # 防御性编程：先检查再初始化
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
        if self.prelude is not None:
            x = self.prelude(x)

        # Determine loop count
        if num_loops is None:
            if targets is not None and config.random_loop is True:
                # During training: randomly sample loop count from log-normal distribution
                if config.random_distribution == "log_norm":
                    num_loops = sample_random_loops(self.config)
                elif config.random_distribution == "poisson":
                    num_loops = sample_poisson_loops(self.config)
                elif config.random_distribution == "uniform":
                    num_loops = sample_uniform_loops(self.config)
            else:
                # During evaluation: use fixed loop count
                num_loops = self.config.l_loops
        
        # Looped structure: pass through transformer layers multiple times (random loop count)
        for i in range(num_loops):
            if self.config.l2_limit:
                if i == num_loops-1:
                    old_x = x
                    for layer in self.layers:
                        x = layer(x)
                    # 计算最后一次循环前后的 L2 距离
                    l2_distance = torch.norm(x - old_x, p=2, dim=-1).mean()
                    # L2 范数损失（可以添加权重系数）
                    l2_norm_loss = self.config.l2_weight * l2_distance
                else:
                    for layer in self.layers:
                        x = layer(x)
            else:
                for layer in self.layers:
                    x = layer(x)
        
        # Coda (non-looped, executed once)
        if self.coda is not None:
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
            if self.config.l2_limit and l2_norm_loss is not None:
                    loss = loss + l2_norm_loss
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Generation logic for evaluation:
        - Remove EOS stopping (EOS only used for sample marking during training, not needed for evaluation)
        - Fixed generation by max_new_tokens (true result length) to ensure complete output at once
        - Use greedy decoding only (paper does not mention sampling, avoid randomness affecting accuracy)
        - Use fixed loop count (config.l_loops) for evaluation
        """
        idx = idx.to(config.device)
        # Strictly generate by max_new_tokens, no early termination (ensure complete output at once)
        for step in range(max_new_tokens):
            # Truncate to model maximum sequence length
            idx_cond = idx[:, -self.config.max_len:]
            # Use fixed loop count during evaluation
            logits, _ = self(idx_cond, targets=None, num_loops=self.config.l_loops)  # Ignore loss (not needed during generation)
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

# ==========================================
# 4. Training Result Saving Utility Functions
# ==========================================
def save_training_loss(step, loss, avg_loss, lr, loss_file):
    """Save training loss to CSV file"""
    file_exists = os.path.exists(loss_file)
    with open(loss_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['step', 'loss', 'avg_loss', 'learning_rate', 'timestamp'])
        writer.writerow([step, f'{loss:.6f}', f'{avg_loss:.6f}', f'{lr:.8f}', datetime.now().isoformat()])

def save_evaluation_result(step, input_str, generated_text, generated_num, gt_text, gt_num, is_correct, eval_file):
    """Save evaluation results at intermediate checkpoints"""
    result = {
        'step': step,
        'input': input_str,
        'generated_text': generated_text,
        'generated_num': generated_num,
        'ground_truth_text': gt_text,
        'ground_truth_num': gt_num,
        'is_correct': is_correct,
        'timestamp': datetime.now().isoformat()
    }
    
    # Read existing results or create new list
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            results = json.load(f)
    else:
        results = []
    
    results.append(result)
    
    # Save results
    with open(eval_file, 'w') as f:
        json.dump(results, f, indent=2)

def save_final_accuracy(accuracy, correct, total, acc_file):
    """Save final accuracy evaluation results"""
    result = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'timestamp': datetime.now().isoformat()
    }
    
    # Read existing results or create new list
    if os.path.exists(acc_file):
        with open(acc_file, 'r') as f:
            results = json.load(f)
    else:
        results = []
    
    results.append(result)
    
    # Save results
    with open(acc_file, 'w') as f:
        json.dump(results, f, indent=2)

# ==========================================
# 5. Multi-GPU Training Main Loop (Core: DataParallel wrapper)
# ==========================================
def train():
    print(f"Initializing Looped Transformer: {config.k_layers} layers x {config.l_loops} loops (Effective depth: {config.k_layers * config.l_loops})")
    
    # Create save directory and file paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"random_logs/r{config.random_loop}_training_results_k{config.k_layers}_{config.norm_structure}__{config.norm_type}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    loss_file = os.path.join(save_dir, "training_loss.csv")
    eval_file = os.path.join(save_dir, "evaluation_samples.json")
    
    print(f"Training results will be saved to: {save_dir}")
    
    # 1. Initialize model and wrap with DataParallel (multi-GPU parallel)
    model = LoopedTransformer(config)
    if config.num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(config.num_gpus))).to(config.device)
    else:
        model = model.to(config.device)
    
    # 2. Dataset and DataLoader (multi-GPU automatically splits batch)
    train_dataset = AdditionDataset(tokenizer, config, mode='train', total_samples=config.total_steps*config.batch_size, data_dir='data')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,  # Batch size per GPU
        collate_fn=collate_fn,
        pin_memory=True,  # Accelerate CPU to GPU data transfer
        num_workers=4     # Data loading threads (adjust according to CPU cores)
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,          
        betas=(0.9, 0.999), 
        eps=1e-8,          
        weight_decay=0.01,  
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_steps
    )
    
    # 4. Training loop
    model.train()
    progress_bar = tqdm(range(config.total_steps), desc="Training")
    data_iter = iter(train_loader)
    loss_window = []
    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        input_ids = batch["input_ids"].to(config.device, non_blocking=True)
        targets = batch["labels"].to(config.device, non_blocking=True)
        
        # Forward pass (DataParallel automatically splits batch to multiple GPUs)
        # During training, use random loop count (automatically sampled inside forward)
        logits, loss = model(input_ids, targets)
        
        # Critical fix 1: Ensure loss is scalar (average after multi-GPU aggregation)
        if config.num_gpus > 1:
            loss = loss.mean()  # Convert multi-GPU loss vector [loss0, loss1] to scalar (loss0+loss1)/2
        
        total_loss = loss
        
        # Debug: check if loss calculation is correct (only for first few steps, after loss converted to scalar)
        if step < 5:
            # Check non-100 positions in targets (only check first sample to avoid DataParallel complexity)
            batch_size = targets.size(0)
            first_sample_input = input_ids[0]  # Only check first sample
            first_sample_targets = targets[0]  # Only check first sample
            valid_mask = (first_sample_targets != -100)
            valid_targets = first_sample_targets[valid_mask]
            if len(valid_targets) > 0:
                # Check corresponding logits (only check first sample)
                first_sample_logits = logits[0]  # [seq_len, vocab_size]
                # Only take logits at valid positions
                valid_logits = first_sample_logits[valid_mask]
                # Calculate accuracy
                preds = torch.argmax(valid_logits, dim=-1)
                acc = (preds == valid_targets).float().mean()
                print(f"[Debug Step {step}] Loss: {loss.item():.6f}, Total Loss: {total_loss.item():.6f}, Accuracy on valid targets (first sample): {acc.item():.4f}")
                # Print label alignment verification: show input sequence and corresponding labels
                valid_indices = torch.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    print(f"[Debug Step {step}] Label alignment check (first 3 valid positions):")
                    for idx_pos in valid_indices[:3]:
                        context_end = idx_pos + 1
                        context = first_sample_input[:context_end]
                        predicted_token_id = first_sample_targets[idx_pos].item()
                        context_str = tokenizer.decode(context.tolist())
                        predicted_token = tokenizer.i2c[predicted_token_id]
                        print(f"  Position {idx_pos}: context='{context_str}' -> should predict '{predicted_token}' (id={predicted_token_id})")
                # Print first few predictions and ground truth
                if len(preds) > 0:
                    print(f"[Debug Step {step}] First 5 predictions: {[tokenizer.i2c[p.item()] for p in preds[:5]]}")
                    print(f"[Debug Step {step}] First 5 targets: {[tokenizer.i2c[t.item()] for t in valid_targets[:5]]}")
        
        # Critical fix 2: Gradient accumulation scaling (based on global batch average loss)
        total_loss = total_loss / config.accumulation_steps
        total_loss.backward()
        
        # Update parameters after accumulating to specified steps
        if (step + 1) % config.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Record loss (restore to global loss value)
        loss_window.append(total_loss.item() * config.accumulation_steps)
        if len(loss_window) > 100:
            loss_window.pop(0)
        
        # Print progress and save loss
        if step % 1 == 0:
            avg_loss = np.mean(loss_window)
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({"Avg Loss": f"{avg_loss:.6f}"})
            # Save loss (every 10 steps to reduce IO overhead)
            if step % 10 == 0:
                save_training_loss(step, total_loss.item() * config.accumulation_steps, avg_loss, current_lr, loss_file)
        
        # Evaluate generation quality every 500 steps (only use main GPU)
        if step % 500 == 0 and step > 0:
            eval_result = evaluate_sample(model.module if config.num_gpus > 1 else model, step, eval_file)
            model.train()
    
    # 5. Save model (only save main GPU parameters to avoid redundancy)
    model_save_path = os.path.join(save_dir, f"looped_transformer_k{config.k_layers}_L{config.l_loops}_{config.norm_structure}_{config.norm_type}_4digit_addition.pt")
    print(f"\nSaving model to {model_save_path}...")
    # After DataParallel wrapper, model parameters are in model.module
    torch.save(
        model.module.state_dict() if config.num_gpus > 1 else model.state_dict(),
        model_save_path
    )
    print("Model saved successfully.")
    
    # Save training configuration information
    config_info = {
        'k_layers': config.k_layers,
        'l_loops': config.l_loops,
        'norm_structure': config.norm_structure,
        'norm_type': config.norm_type,
        'd_model': config.d_model,
        'n_heads': config.n_heads,
        'd_ff': config.d_ff,
        'l2_limit': config.l2_limit,
        'l2_weight': config.l2_weight,
        "random_distribution": config.random_distribution,
        'poisson': config.poisson_lambda,
        'random_loop_mu': config.random_loop_mu,
        'random_loop_sigma': config.random_loop_sigma,
        'random_loop_min': config.random_loop_min,
        'random_loop_max': config.random_loop_max,
        'prelude': config.prelude,
        'coda': config.coda,
        'random_loop': config.random_loop,
        'dropout': config.dropout,
        'batch_size': config.batch_size,
        'total_steps': config.total_steps,
        'learning_rate': config.learning_rate,
        'warmup_steps': config.warmup_steps,
        'num_digits': config.num_digits,
        'num_gpus': config.num_gpus,
        'timestamp': timestamp
    }
    config_file = os.path.join(save_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config_info, f, indent=2)
    
    return model, save_dir
def extract_number(text):
    """
    Extract pure digits from generated text and convert to integer (handle non-digit characters and conversion failures)
    Note: Since target order is reversed (predict least significant digit first), need to reverse extracted digit string back
    Args:
        text: Model generated text (e.g., " 4031", "4031§", "abc123", etc., digit order is reversed)
    Returns:
        int or None: Extracted integer (reversed back to normal order), None if extraction fails
    """
    # 1. Filter non-digit characters (only keep 0-9)
    digits = [c for c in text if c.isdigit()]
    if not digits:  # No digits, extraction failed
        return None
    # 2. Concatenate to digit string, reverse and convert to integer (because target order is reversed during training)
    try:
        digit_str = "".join(digits)
        reversed_digit_str = digit_str[::-1]  # Reverse back to normal order
        return int(reversed_digit_str)
    except ValueError:  # Extreme case (e.g., empty string), return None
        return None
# ==========================================
# 6. Generation Evaluation (only use main GPU to avoid multi-GPU synchronization)
# ==========================================
def evaluate_sample(model, step=None, eval_file=None):
    model.eval()
    # Generate test samples for 8-digit addition
    ds = AdditionDataset(tokenizer, config, mode='test', total_samples=1000, data_dir='data')
    sample = ds.generate_sample(0)
    
    # Key: get ground truth numerical value (for integer comparison)
    input_str_len = sample["input_len"]
    ground_truth_ids = sample["input_ids"][input_str_len:]
    # Filter EOS from ground truth, decode to text
    ground_truth_ids = ground_truth_ids[ground_truth_ids != config.eos_token_id]
    gt_text = tokenizer.decode(ground_truth_ids.tolist())
    # Convert ground truth to integer
    gt_num = extract_number(gt_text)
    
    # Prepare input: only "operands+equals"
    input_ids = sample["input_ids"][:input_str_len].unsqueeze(0).to(config.device)
    max_new_tokens = len(ground_truth_ids)  # Generate by true result length
    
    # Generate result
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    # Extract generated result and decode
    generated_result_ids = generated_ids[0, input_str_len:]
    generated_result_ids = generated_result_ids[generated_result_ids != config.eos_token_id]
    generated_text = tokenizer.decode(generated_result_ids.tolist())
    # Convert generated result to integer
    generated_num = extract_number(generated_text)
    
    # Determine correctness
    is_correct = generated_num == gt_num if (generated_num is not None and gt_num is not None) else False
    
    # Print information (with text and integer for clear comparison)
    input_str = tokenizer.decode(input_ids[0].tolist())
    print(f"\n[Step Check] Input: {input_str}")
    print(f"[Step Check] Model Output Text: {generated_text} → Extracted Number: {generated_num}")
    print(f"[Step Check] Ground Truth Text: {gt_text} → Extracted Number: {gt_num}")
    print(f"[Step Check] Result: {'CORRECT' if is_correct else 'INCORRECT'}")
    print("-" * 60)
    
    # Save evaluation results
    if step is not None and eval_file is not None:
        save_evaluation_result(step, input_str, generated_text, generated_num, gt_text, gt_num, is_correct, eval_file)
    
    return {
        'input': input_str,
        'generated_text': generated_text,
        'generated_num': generated_num,
        'gt_text': gt_text,
        'gt_num': gt_num,
        'is_correct': is_correct
    }
# ==========================================
# 7. Test Accuracy (only use main GPU to avoid multi-GPU overhead)
# ==========================================
def evaluate_accuracy(model, n_samples=1000, acc_file=None):
    # Extract main GPU model (single GPU for evaluation after multi-GPU training)
    model = model.module if (config.num_gpus > 1 and hasattr(model, 'module')) else model
    model.eval()
    model.to(config.device)
    correct = 0
    ds = AdditionDataset(tokenizer, config, mode='test', total_samples=n_samples, data_dir='data')
    
    print(f"Evaluating on 4-digit addition (Total {n_samples} samples)...")
    with torch.no_grad():
        for idx in tqdm(range(n_samples)):
            sample = ds.generate_sample(idx)
            input_str_len = sample["input_len"]
            ground_truth_ids = sample["input_ids"][input_str_len:]
            
            # 1. Process ground truth: decode → extract integer
            ground_truth_ids = ground_truth_ids[ground_truth_ids != config.eos_token_id]
            gt_text = tokenizer.decode(ground_truth_ids.tolist())
            gt_num = extract_number(gt_text)
            if gt_num is None:  # Extreme case (generation logic error), skip this sample
                continue
            
            # 2. Generate model output
            prompt = sample["input_ids"][:input_str_len].unsqueeze(0).to(config.device)
            max_new_tokens = len(ground_truth_ids)
            generated_ids = model.generate(prompt, max_new_tokens=max_new_tokens)
            
            # 3. Process generated result: decode → extract integer
            generated_result_ids = generated_ids[0, input_str_len:]
            generated_result_ids = generated_result_ids[generated_result_ids != config.eos_token_id]
            generated_text = tokenizer.decode(generated_result_ids.tolist())
            generated_num = extract_number(generated_text)
            if generated_num is None:  # Extraction failed, considered as error
                continue
            # 4. Integer comparison: count as correct if match
            if generated_num == gt_num:
                correct += 1
    
    # Calculate and print accuracy
    acc = correct / n_samples if n_samples > 0 else 0.0
    print(f"[Final Accuracy] 4-digit addition: {acc*100:.2f}% (Correct: {correct}/{n_samples})")
    
    # Save accuracy results
    if acc_file is not None:
        save_final_accuracy(acc, correct, n_samples, acc_file)
    
    return acc

# ==========================================
# 8. Main Function (Start Training and Evaluation)
# ==========================================
if __name__ == "__main__":
    # Fix random seed (ensure reproducibility)
    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print("Starting Training...")
    model, save_dir = train()
    # model=LoopedTransformer(config)
    # model_path = ""
    # model_path=""
    # model.load_state_dict(torch.load(model_path))
    # After training, evaluate
    print("\n=== Final Evaluation ===")
    acc_file = os.path.join(save_dir, "final_accuracy.json")
    evaluate_accuracy(model, n_samples=100, acc_file=acc_file)
    print(f"\nAll results saved to: {save_dir}")