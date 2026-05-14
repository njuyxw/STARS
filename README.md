# LoopedModel_STARS

This repository contains the implementation and experiments for our paper on stabilizing and improving looped language models. It includes two experimental settings:

- **Addition task** (`train_addition/`): explores the effects of normalization structures, random loop-count training, and Jacobian regularization on a synthetic addition task.
- **Math reasoning** (`ouro_experiment/`): fine-tunes [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B) on math datasets with random loop sampling and Jacobian regularization.

## Directory Structure

```
LoopedModel_STARS/
├── README.md
├── ICML-looped-llm-cmr-v2.pdf   # Paper
├── environment.yml              # Conda environment specification
├── train_addition/              # Addition task experiments
│   ├── data/                    # Training/testing data (4-digit addition)
│   ├── primary_train_formal.py  # Train: norm structure & type study (Sec 4.1)
│   ├── primary_test_formal.py   # Test: norm structure & type evaluation
│   ├── primary_pca_formal.py    # PCA visualization of hidden state trajectories
│   ├── random_train_formal.py   # Train: random loop sampling + model structure (Sec 4.2)
│   ├── random_test_formal.py    # Test: random loop models evaluation
│   ├── train.py                 # Train: combined random loop + Jacobian regularization
│   ├── formal_logs/             # Logs for primary experiments
│   ├── random_logs/             # Logs for random loop experiments
│   ├── logs/                    # Additional logs
│   └── reglogs/                 # Logs for regularization experiments
└── ouro_experiment/             # Math reasoning experiments
    ├── Ouro-1.4B/               # Model weights & config (download from HuggingFace)
    │   ├── modeling_ouro.py     # Modified: reg_loss + random loop support
    │   ├── configuration_ouro.py
    │   ├── config.json
    │   └── ...
    ├── train_large.py           # Training script
    ├── run.sh                   # Launch training via torchrun
    ├── run_eval.sh              # Evaluation with vLLM + lm-eval-harness
    ├── ouro_vllm.py             # vLLM adapter (for older vLLM without Ouro support)
    └── outputs/                 # Training checkpoints
```

## Environment

A reference conda environment is provided in `environment.yml`. To create it:

```bash
conda env create -f environment.yml
conda activate ouro
```

For vLLM-based evaluation, additionally install vLLM (the exact version depends on your CUDA setup; we tested with `vllm>=0.8.0`):

```bash
pip install vllm
```

## Addition Task (`train_addition/`)

Synthetic 4-digit addition experiments on a small looped transformer. Refer to the paper for full experimental details and analysis.

### Key Files

**`primary_train_formal.py`** — Studies the effect of normalization placement and type on hidden state trajectory (Section 4.1). Configurable via `Config` class:

- `norm_structure`: `pre_norm`, `post_norm`, `sandwich_branch`, `sandwich_dual`
- `norm_type`: `LayerNorm`, `RMSNorm`, `DeepNorm`, `SimpleNorm`
- Fixed loop count (`l_loops`), single shared block (`k_layers=1`)

**`primary_test_formal.py`** — Evaluates models trained by `primary_train_formal.py`.

**`primary_pca_formal.py`** — PCA-based visualization of hidden state trajectories across loop iterations.

**`random_train_formal.py`** — Studies random loop-count training and model structure variants (Section 4.2). Key `Config` options:

- `random_distribution`: `"log_norm"`, `"poisson"`, `"uniform"` — loop count distribution
- `random_loop_mu` / `random_loop_sigma` — log-normal parameters
- `prelude` / `coda`: toggles non-looped blocks before/after the recurrent module, enabling comparison of **only-recurrent**, **with-prelude**, and **with-coda** architectures
- `l2_limit`: optionally adds an L2 constraint on the final loop step

**`random_test_formal.py`** — Evaluates models trained by `random_train_formal.py`.

**`train.py`** — Combines random loop sampling with Jacobian regularization (`reg_loss`), the proposed method in the paper. Uses power iteration to estimate the Jacobian Frobenius norm of the loop dynamics, penalizing unstable recurrent transformations. Key parameter: `reg_weight`.

### Training

All scripts are self-contained; modify the `Config` class in each file and run directly:

```bash
cd train_addition
python primary_train_formal.py
python random_train_formal.py
python train.py
```

Each script auto-detects available GPUs. Outputs (checkpoints, logs, loss curves) are saved to timestamped directories under `formal_logs/`, `random_formal_logs/`, or `newlogs/`.

## Math Reasoning (`ouro_experiment/`)

Fine-tunes Ouro-1.4B on math datasets with loop regularization and random loop sampling.

### Setup

**1. Create environment and download the model:**

```bash
conda env create -f environment.yml
conda activate ouro

pip install huggingface_hub
huggingface-cli download ByteDance/Ouro-1.4B --local-dir ouro_experiment/Ouro-1.4B
```

**2. Install vLLM (for evaluation):**

```bash
pip install vllm
```

### Key Modifications to `modeling_ouro.py`

- **Jacobian regularization**: During training, computes the Frobenius norm of the loop dynamics Jacobian via power iteration on the final hidden states, added as a regularization term (controlled by `reg_weight`).
- **Dynamic `total_ut_steps`**: The number of universal transformer iterations can be set per forward pass, enabling random loop-count training.

### Configuration

Edit the `CLISettings` dataclass in `train_large.py` for revising parameters.

Random loop parameters and regularization weight are set in `tightly_scoped_fwd_bwd`:

```python
total_ut_steps = sample_random_loops(random_loop_mu=1.7, random_loop_sigma=0.4,
                                     random_loop_min=1, random_loop_max=16)
total_loss = (1 - reg_weight) * loss + reg_weight * reg_loss   # reg_weight default: 0.1
```

### Training

```bash
cd ouro_experiment
bash run.sh
```

Uses `torchrun` on 8 GPUs. Edit `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` in `run.sh` as needed. Checkpoints are saved to `outputs/<run_name>/`.

### Evaluation

**With native vLLM Ouro support:**

```bash
cd ouro_experiment
bash run_eval.sh
```

Edit `MODEL_PATH`, `total_ut_steps` in `hf_overrides`, and `--tasks` in the script.

**With older vLLM (no native Ouro support):**

Place `ouro_vllm.py` in the same directory as `run_eval.sh`. This adapter unrolls the recurrent layers into `N * T` distinct layer objects for KV-cache compatibility and disables dynamic early-exit for batched inference.

## Citation
