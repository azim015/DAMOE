# DA-MoE: Diversity-Aware Mixture-of-Experts for Time-Series Forecasting

> **Paper:** *Efficient Diversity-aware Mixture-of-Experts Model Selection For Time-Series Prediction* — Anonymous Authors, Under review at ICML.

---

## Overview

DA-MoE is a general-purpose time-series forecasting framework built on a **Mixture-of-Experts (MoE)** architecture. Unlike static ensemble methods, DA-MoE dynamically selects, routes, and fuses specialized forecasting models on a per-sample basis — adapting to the temporal regime of each input window.

### Key Contributions

| # | Contribution |
|---|---|
| 1 | **Diversity-aware expert pool construction** via pairwise error correlation minimisation and regime-specialised experts |
| 2 | **Hybrid multi-scale meta-feature extractor** combining statistical descriptors, self-supervised temporal embeddings, and graph-aware multivariate encodings |
| 3 | **Uncertainty-aware fusor** with entropy regularisation, calibration loss, stability regularisation, and a residual nonlinear mixer |
| 4 | **Reliability-driven feedback mechanism** — a bounded score `sk ∈ [−1, 1]` per expert updated online after each prediction |

---

## Architecture

```
Input Window X  (B, T_in, d_vars)
        │
        ▼
┌───────────────────────────────────────────────┐
│        Hybrid Meta-Feature Extractor          │
│  ┌──────────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Multi-Scale  │ │Temporal  │ │  Graph   │  │
│  │ Statistical  │ │Encoder   │ │ Encoder  │  │
│  │ Descriptors  │ │ (SSL)    │ │ (GNN)    │  │
│  └──────┬───────┘ └────┬─────┘ └────┬─────┘  │
│         └──────────────┴────────────┘         │
│                   x* (meta-feature vector)    │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│        Expert Pool  F = {f1, ..., fK}         │
│  PatchTST │ iTransformer │ TimeMixer           │
│  DLinear  │ FEDformer    │ TimesNet            │
│  PAttn    │ TimeXer                            │
│  (shared backbone + LoRA adapters)             │
└───────────────────────────────────────────────┘
        │  predictions + uncertainty
        ▼
┌───────────────────────────────────────────────┐
│       Uncertainty-Aware Fusor                 │
│  • Nonlinear gating  G(x̃; Θ) / τ             │
│  • Score-adjusted weights  w̃k                 │
│  • Residual nonlinear mixer  H(·; Ω)          │
│  • Multi-objective loss                       │
│    L = Lforecast + λ1·Lentropy                │
│          + λ2·Lcalibration + λ3·Lstability    │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│     Reliability Feedback Mechanism            │
│  sk,t = clip((1−β)·sk,t−1 + β·tanh(γ·ak,t)) │
└───────────────────────────────────────────────┘
        │
        ▼
   Forecast Ŷ  (B, T_out, d_vars)
```

---

## Project Structure

```
da_moe/
├── README.md
├── requirements.txt
├── setup.py
├── run_experiments.py          # Main CLI entry point
│
├── models/
│   ├── __init__.py
│   ├── experts.py              # 8 expert backbones + shared LoRA backbone
│   ├── meta_features.py        # Hybrid multi-scale meta-feature extractor
│   ├── diversity_selection.py  # Error-correlation selection + reliability tracker
│   ├── fusor.py                # Gating network, mixer, fusion loss
│   └── da_moe.py               # Full DA-MoE model (Algorithm 1)
│
├── data/
│   ├── __init__.py
│   └── dataset.py              # Dataset class, scalers, synthetic generators
│
└── experiments/
    ├── __init__.py
    ├── trainer.py              # 2-phase training pipeline + evaluation
    └── ablation.py             # Ablation variants + top-k sensitivity sweep
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/da-moe.git
cd da-moe
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Single dataset, single horizon

```bash
python run_experiments.py \
    --dataset ETTh1 \
    --pred_len 96 \
    --input_len 96 \
    --epochs 30 \
    --device cpu
```

### With ablation study and top-k sensitivity analysis

```bash
python run_experiments.py \
    --dataset ETTh1 \
    --pred_len 96 \
    --epochs 30 \
    --ablation \
    --sensitivity \
    --device cuda
```

### Multi-horizon benchmark (96 / 192 / 336 / 720)

```bash
python run_experiments.py \
    --dataset Traffic \
    --multi_horizon \
    --epochs 30 \
    --device cuda
```

### Full benchmark across all 8 datasets

```bash
python run_experiments.py \
    --all_datasets \
    --epochs 30 \
    --device cuda
```

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `ETTh1` | Dataset name (`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Traffic`, `Electricity`, `Weather`, `Solar`) |
| `--pred_len` | `96` | Forecast horizon |
| `--input_len` | `96` | Look-back window length |
| `--d_vars` | `7` | Number of input variables |
| `--epochs` | `20` | Training epochs |
| `--ssl_epochs` | `3` | Self-supervised pretraining epochs |
| `--lr` | `1e-3` | Learning rate |
| `--batch_size` | `32` | Batch size |
| `--patience` | `5` | Early stopping patience |
| `--device` | `cpu` | Device (`cpu`, `cuda`, `auto`) |
| `--ablation` | `False` | Run ablation study |
| `--sensitivity` | `False` | Run top-k sensitivity sweep |
| `--multi_horizon` | `False` | Run horizons {96, 192, 336, 720} |
| `--all_datasets` | `False` | Run all 8 benchmark datasets |

---

## Using DA-MoE Programmatically

```python
import torch
from models import build_da_moe
from data import make_loaders

# Build model
model = build_da_moe({
    "input_len":     96,
    "pred_len":      96,
    "d_vars":        7,
    "d_model":       64,
    "d_meta_embed":  128,
    "graph_d":       32,
    "hidden_fusor":  128,
    "max_pool_size": 5,
    "top_k":         5,
    "temperature":   1.0,
    "eta":           2.0,
    "dropout":       0.1,
}, device="cpu")

# Inference
x = torch.randn(4, 96, 7)          # (batch, time, variables)
y_hat, info = model(x)             # y_hat: (4, 96, 7)

print(y_hat.shape)                  # torch.Size([4, 96, 7])
print(info["selected_experts"])     # ['patchtst', 'itransformer', ...]
print(info["gate_weights"].shape)   # torch.Size([4, 5])

# Training step
loss, loss_info = model.compute_loss(x, target=torch.randn(4, 96, 7))

# DataLoaders (built-in synthetic benchmarks)
train_loader, val_loader, test_loader, scaler = make_loaders(
    "ETTh1", input_len=96, pred_len=96, batch_size=32
)
```

### Full training pipeline

```python
from experiments.trainer import train

results = train(
    model,
    train_loader, val_loader, test_loader,
    device="cpu",
    epochs=30,
    lr=1e-3,
    patience=5,
    ssl_epochs=3,
)

print(results["final"])
# {'MSE': 0.421, 'MAE': 0.429, 'RMSE': 0.649, 'MAPE': 0.083}
```

---

## Expert Models

DA-MoE maintains a pool of 8 diverse expert architectures, all sharing a backbone encoder with lightweight **LoRA-style adapters** for parameter efficiency:

| Expert | Architecture Style | Key Mechanism |
|---|---|---|
| `PatchTSTExpert` | Transformer | Patch tokenisation + attention |
| `iTransformerExpert` | Transformer | Variable-axis (inverted) attention |
| `TimeMixerExpert` | MLP-Mixer | Multi-scale decomposition mixing |
| `DLinearExpert` | Linear | Trend + residual linear decomposition |
| `FEDformerExpert` | Spectral | Low-frequency Fourier mode mixing |
| `TimesNetExpert` | CNN | 1-D → 2-D reshape + 2-D convolution |
| `PAttnExpert` | Attention | Pure multi-head self-attention |
| `TimeXerExpert` | Cross-Attention | Encoder–decoder cross-attention |

---

## Meta-Feature Extractor

The extractor builds a unified meta-feature vector `x*` from three branches:

### Branch 1 — Multi-Scale Statistical Descriptors
Computes mean, variance, lag-k autocorrelation, and entropy at scales `{1, 2, 4, 8}` over non-overlapping temporal segments. Output: `(B, L × Fs × d_vars × 2)`.

### Branch 2 — Self-Supervised Temporal Encoder
A Transformer encoder pretrained with masked reconstruction. Maps the input to a compact embedding `zlearned ∈ R^d_embed`.

### Branch 3 — Graph-Aware Multivariate Encoder
Builds a lag-aware Pearson correlation graph over variables. A one-hop GNN produces a structural embedding `zgraph ∈ R^dg` encoding cross-variable dependency structure.

---

## Reliability Score Tracker

Each expert maintains a score `sk ∈ [−1, 1]` updated after every prediction batch:

```
a_k,t  = (b_t − ℓ_k,t) / (b_t + ε)          # normalised advantage
s_k,t  = clip((1−β)·s_k,t−1 + β·tanh(γ·a_k,t), −1, 1)
```

where `b_t = median_j(ℓ_j,t)` is the cross-model baseline loss. Scores feed into the composite utility function used for expert ranking:

```
U_k = λ1·Acc_k + λ2·Div_k − λ3·Unc_k + λ4·s_k
```

---

## Training Protocol

DA-MoE uses a two-phase training pipeline:

**Phase 1 — SSL Pretraining**
The temporal encoder is pretrained offline on unlabelled windows using a masked reconstruction objective, ensuring generalisation before label-supervised training.

**Phase 2a — Expert Pool Selection**
All candidate experts are evaluated on the validation set. Pairwise error correlations are computed and a greedy selection procedure builds the final pool by minimising redundancy.

**Phase 2b — Supervised Fusor Training**
The fusor and meta-extractor are trained end-to-end with the multi-objective loss:

```
L = L_forecast + λ1·L_entropy + λ2·L_calibration + λ3·L_stability
```

Early stopping monitors validation MSE. The fusor supports continual adaptation via online gradient updates after deployment.

---

## Ablation Study

Run with `--ablation` to reproduce Table 4 from the paper:

| Variant | Description |
|---|---|
| **Full DA-MoE** | Complete framework |
| w/o Diversity Selection | Random expert pool (no correlation-based selection) |
| w/o Reliability Scoring | All `sk = 0` (no feedback mechanism) |
| w/o Learned Meta-Features | Statistical features only (temporal + graph encoders zeroed) |
| w/o Nonlinear Fusion | Linear averaging only (residual mixer disabled) |

---

## Top-k Sensitivity Analysis

Run with `--sensitivity` to reproduce Table 5 from the paper, sweeping `k ∈ {2, 3, 4, 5, 6, 8}`.

---

## Benchmarks

DA-MoE is evaluated across three forecasting settings:

**Long-term forecasting** (horizons: 96 / 192 / 336 / 720):
`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Weather`, `Solar-Energy`, `Electricity`, `Traffic`

**Short-term spatio-temporal forecasting:**
`PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`

**Electricity price forecasting:**
`NP`, `PJM`, `BE`, `FR`, `DE`

---

## Configuration Reference

All model hyperparameters can be passed to `build_da_moe()` as a dict:

| Parameter | Default | Description |
|---|---|---|
| `input_len` | `96` | Look-back window T_in |
| `pred_len` | `96` | Forecast horizon T_out |
| `d_vars` | `7` | Number of variables |
| `d_model` | `64` | Shared backbone hidden size |
| `d_meta_embed` | `128` | Temporal encoder output dim |
| `graph_d` | `32` | Graph encoder output dim |
| `hidden_fusor` | `128` | Fusor MLP hidden size |
| `max_pool_size` | `5` | Max experts after diversity selection |
| `top_k` | `5` | Experts activated per forward pass |
| `scales` | `(1,2,4,8)` | Temporal scales for stat extractor |
| `lags` | `(1,2,4)` | Autocorrelation lags |
| `temperature` | `1.0` | Gating softmax temperature τ |
| `eta` | `2.0` | Score-adjustment exponent η |
| `mc_samples` | `0` | MC-Dropout samples (0 = disabled) |
| `dropout` | `0.1` | Dropout rate |

---

## Citation

```bibtex
@inproceedings{damoe2025,
  title     = {Efficient Diversity-aware Mixture-of-Experts Model Selection
               For Time-Series Prediction},
  author    = {Anonymous Authors},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025},
  note      = {Under review}
}
```

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.
