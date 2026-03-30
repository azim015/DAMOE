"""
Ablation study experiments (Section 4.2 of DA-MoE paper).

Evaluates contribution of each component:
  - Full DA-MoE
  - w/o Diversity Selection (Random pool)
  - w/o Reliability Scoring (sk removed)
  - w/o Learned Meta-Features (statistical features only)
  - w/o Nonlinear Fusion (linear averaging)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader


def _quick_eval(model, loader, device) -> Dict[str, float]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_hat, _ = model(x)
            preds.append(y_hat.cpu())
            targets.append(y.cpu())
    P = torch.cat(preds)
    T = torch.cat(targets)
    return {
        "MSE": F.mse_loss(P, T).item(),
        "MAE": F.l1_loss(P, T).item(),
    }


def _quick_train(model, train_loader, val_loader, device,
                 epochs: int = 10, lr: float = 1e-3):
    params = (list(model.fusor.parameters()) +
              list(model.meta_extractor.parameters()))
    opt = torch.optim.Adam(params, lr=lr)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss, _ = model.compute_loss(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


# ---------------------------------------------------------------------------
# Variant builders
# ---------------------------------------------------------------------------

def build_variant_random_pool(base_model):
    """w/o Diversity Selection: random expert pool selection."""
    m = copy.deepcopy(base_model)
    import random
    pool = random.sample(m.expert_names, min(m.top_k, len(m.expert_names)))
    m.selected_pool = pool
    return m


def build_variant_no_reliability(base_model):
    """w/o Reliability Scoring: force all scores to 0 (uniform weighting)."""
    m = copy.deepcopy(base_model)
    # Monkey-patch _get_reliability to always return zeros
    def _zero_reliability(names):
        return torch.zeros(len(names))
    m._get_reliability = _zero_reliability
    return m


def build_variant_no_learned_features(base_model):
    """w/o Learned Meta-Features: zero-out temporal encoder and graph encoder."""
    m = copy.deepcopy(base_model)
    # Replace temporal and graph encoders with zero-output modules
    d_embed = m.meta_extractor.temporal_enc.output_dim
    graph_d = m.meta_extractor.graph_enc.output_dim

    class ZeroOut(nn.Module):
        def __init__(self, dim): super().__init__(); self.dim = dim
        def forward(self, *args, **kwargs):
            # return zeros with the right batch dimension
            x = args[0]
            return torch.zeros(x.size(0), self.dim, device=x.device)

    m.meta_extractor.temporal_enc.forward = \
        lambda x, **kw: torch.zeros(x.size(0), d_embed, device=x.device)
    m.meta_extractor.graph_enc.forward = \
        lambda x: torch.zeros(x.size(0), graph_d, device=x.device)
    return m


def build_variant_linear_fusion(base_model):
    """w/o Nonlinear Fusion: disable residual mixer (zero output)."""
    m = copy.deepcopy(base_model)
    # Zero-out the mixer weights so it outputs 0
    for p in m.fusor.mixer.parameters():
        p.data.zero_()
        p.requires_grad = False
    return m


# ---------------------------------------------------------------------------
# Run ablation table
# ---------------------------------------------------------------------------

def run_ablation(
    base_model,
    train_loader,
    val_loader,
    test_loader,
    device:  str = "cpu",
    epochs:  int = 10,
    lr:      float = 1e-3,
) -> Dict[str, Dict[str, float]]:
    """
    Runs all ablation variants.
    Returns dict: variant_name -> {MSE, MAE}
    """

    variants = {
        "Full DA-MoE":                 copy.deepcopy(base_model),
        "w/o Diversity Selection":     build_variant_random_pool(base_model),
        "w/o Reliability Scoring":     build_variant_no_reliability(base_model),
        "w/o Learned Meta-Features":   build_variant_no_learned_features(base_model),
        "w/o Nonlinear Fusion":        build_variant_linear_fusion(base_model),
    }

    results = {}
    for name, model in variants.items():
        print(f"\n[Ablation] Training variant: {name}")
        model = model.to(device)
        _quick_train(model, train_loader, val_loader, device, epochs, lr)
        metrics = _quick_eval(model, test_loader, device)
        results[name] = metrics
        print(f"  MSE={metrics['MSE']:.4f}  MAE={metrics['MAE']:.4f}")

    print("\n" + "=" * 60)
    print("Ablation Results Summary")
    print("=" * 60)
    print(f"{'Variant':<35} {'MSE':>8} {'MAE':>8}")
    print("-" * 55)
    for name, m in results.items():
        print(f"{name:<35} {m['MSE']:>8.4f} {m['MAE']:>8.4f}")

    return results


# ---------------------------------------------------------------------------
# Sensitivity analysis: number of experts k
# ---------------------------------------------------------------------------

def sensitivity_analysis_top_k(
    base_model,
    train_loader,
    val_loader,
    test_loader,
    k_values: List[int] = [2, 3, 4, 5, 6, 8],
    device:   str = "cpu",
    epochs:   int = 8,
    lr:       float = 1e-3,
) -> Dict[int, Dict[str, float]]:
    """Sweeps top-k and reports MSE/MAE for each setting."""
    results = {}
    for k in k_values:
        k_eff = min(k, len(base_model.expert_names))
        print(f"\n[Sensitivity] top-k = {k_eff}")
        m = copy.deepcopy(base_model)
        m.top_k         = k_eff
        m.selected_pool = m.expert_names[:k_eff]
        m = m.to(device)
        _quick_train(m, train_loader, val_loader, device, epochs, lr)
        metrics = _quick_eval(m, test_loader, device)
        results[k] = metrics
        print(f"  k={k_eff}: MSE={metrics['MSE']:.4f}  MAE={metrics['MAE']:.4f}")

    print("\n" + "=" * 50)
    print("Top-k Sensitivity Analysis")
    print("=" * 50)
    print(f"{'k':>5} {'MSE':>10} {'MAE':>10}")
    print("-" * 30)
    for k, m in results.items():
        print(f"{k:>5} {m['MSE']:>10.4f} {m['MAE']:>10.4f}")

    return results
