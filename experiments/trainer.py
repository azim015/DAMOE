"""
Training and evaluation pipeline for DA-MoE.

Implements the two-phase training protocol from the paper:
  Phase 1: Offline SSL pretraining of temporal encoder on unlabelled windows.
  Phase 2: End-to-end supervised training of fusor + online expert scoring.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(pred, target).item()

def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(pred, target).item()

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(F.mse_loss(pred, target)).item()

def mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    return (((pred - target).abs() / (target.abs() + eps)).mean()).item()

def evaluate(model, loader, device: str) -> Dict[str, float]:
    """Run full evaluation loop. Returns MSE, MAE, RMSE, MAPE."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_hat, _ = model(x)
            preds.append(y_hat.cpu())
            targets.append(y.cpu())
    P = torch.cat(preds,   dim=0)
    T = torch.cat(targets, dim=0)
    return {
        "MSE":  mse(P, T),
        "MAE":  mae(P, T),
        "RMSE": rmse(P, T),
        "MAPE": mape(P, T),
    }


# ---------------------------------------------------------------------------
# Phase 1: SSL pretraining of temporal encoder
# ---------------------------------------------------------------------------

def pretrain_temporal_encoder(
    model,
    loader:      DataLoader,
    device:      str   = "cpu",
    epochs:      int   = 5,
    lr:          float = 1e-3,
    log_every:   int   = 10,
) -> List[float]:
    """SSL masked-reconstruction pretraining of meta_extractor.temporal_enc."""
    enc       = model.meta_extractor.temporal_enc
    optimizer = torch.optim.Adam(enc.parameters(), lr=lr)
    losses    = []

    enc.train()
    for epoch in range(epochs):
        epoch_loss = []
        for step, (x, _) in enumerate(loader):
            x = x.to(device)
            optimizer.zero_grad()
            loss = enc.ssl_loss(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(loss.item())
            if step % log_every == 0:
                print(f"  [SSL Pretrain] Epoch {epoch+1}/{epochs} "
                      f"step {step}: loss={loss.item():.4f}")
        losses.append(np.mean(epoch_loss))
        print(f"[SSL Pretrain] Epoch {epoch+1} avg loss: {losses[-1]:.4f}")
    return losses


# ---------------------------------------------------------------------------
# Phase 2: Main supervised training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    str,
    scheduler  = None,
) -> Dict[str, float]:
    model.train()
    total_loss = []
    info_acc   = {}

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss, info = model.compute_loss(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss.append(info["total"])
        for k, v in info.items():
            info_acc.setdefault(k, []).append(v)

    if scheduler is not None:
        scheduler.step()

    return {k: float(np.mean(v)) for k, v in info_acc.items()}


def train(
    model,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    device:       str   = "cpu",
    epochs:       int   = 20,
    lr:           float = 1e-3,
    patience:     int   = 5,
    ssl_epochs:   int   = 3,
    log_every:    int   = 1,
) -> Dict:
    """
    Full DA-MoE training pipeline.

    1. SSL pretrain temporal encoder.
    2. Build expert pool from validation set.
    3. End-to-end supervised training with multi-objective loss.
    4. Early stopping on validation MSE.
    """

    print("=" * 60)
    print("DA-MoE Training Pipeline")
    print("=" * 60)

    # ---- Phase 1: SSL pretraining ------------------------------------
    print("\n[Phase 1] Self-Supervised Pretraining of Temporal Encoder")
    pretrain_temporal_encoder(model, train_loader, device, epochs=ssl_epochs)

    # ---- Pool selection from validation errors -----------------------
    print("\n[Phase 2a] Building Expert Pool from Validation Errors")
    model.build_pool_from_validation(val_loader, device=device)

    # Save fusor reference params (for stability regularisation)
    model.fusor.save_reference_params()

    # ---- Phase 2b: Supervised training -------------------------------
    print("\n[Phase 2b] Supervised End-to-End Training")
    # Train only fusor + meta-extractor; backbone is shared
    fusor_params = (list(model.fusor.parameters()) +
                    list(model.meta_extractor.parameters()))
    optimizer = torch.optim.AdamW(fusor_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mse  = float("inf")
    best_state    = None
    patience_ctr  = 0
    history       = {"train": [], "val": [], "test": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_info = train_one_epoch(model, train_loader, optimizer, device, scheduler)
        val_metrics  = evaluate(model, val_loader,  device)
        test_metrics = evaluate(model, test_loader, device)

        history["train"].append(train_info)
        history["val"].append(val_metrics)
        history["test"].append(test_metrics)

        elapsed = time.time() - t0
        if epoch % log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_info['total']:.4f} | "
                  f"Val MSE: {val_metrics['MSE']:.4f} "
                  f"MAE: {val_metrics['MAE']:.4f} | "
                  f"Test MSE: {test_metrics['MSE']:.4f} "
                  f"MAE: {test_metrics['MAE']:.4f} | "
                  f"Time: {elapsed:.1f}s")

        # Early stopping
        if val_metrics["MSE"] < best_val_mse:
            best_val_mse = val_metrics["MSE"]
            best_state   = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n[Final Evaluation]")
    final = evaluate(model, test_loader, device)
    print(f"  MSE : {final['MSE']:.4f}")
    print(f"  MAE : {final['MAE']:.4f}")
    print(f"  RMSE: {final['RMSE']:.4f}")
    print(f"  MAPE: {final['MAPE']:.4f}")

    return {"history": history, "final": final, "best_val_mse": best_val_mse}
