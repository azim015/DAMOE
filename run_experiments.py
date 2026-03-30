#!/usr/bin/env python3
"""
DA-MoE: Full experimental run script.

Runs training, evaluation, ablation, and sensitivity analysis
across multiple datasets, producing a results summary table
matching the paper's experimental setup.

Usage:
    python run_experiments.py
    python run_experiments.py --dataset ETTh1 --pred_len 96 --epochs 20
"""

import argparse
import sys
import os
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from models  import DAMoE, build_da_moe
from data    import make_loaders
from experiments.trainer  import train, evaluate
from experiments.ablation import run_ablation, sensitivity_analysis_top_k


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "d_model":       64,
    "d_meta_embed":  128,
    "graph_d":       32,
    "hidden_fusor":  128,
    "max_pool_size": 5,
    "top_k":         5,
    "scales":        (1, 2, 4),
    "lags":          (1, 2, 4),
    "temperature":   1.0,
    "eta":           2.0,
    "mc_samples":    0,
    "dropout":       0.1,
}


def run_single_experiment(
    dataset_name: str,
    pred_len:     int,
    input_len:    int = 96,
    d_vars:       int = 7,
    batch_size:   int = 32,
    epochs:       int = 20,
    ssl_epochs:   int = 3,
    lr:           float = 1e-3,
    patience:     int = 5,
    device:       str = "cpu",
    run_ablation_flag: bool = False,
    run_sensitivity:   bool = False,
) -> dict:

    print(f"\n{'='*65}")
    print(f"Dataset: {dataset_name}  |  pred_len: {pred_len}  |  input_len: {input_len}")
    print(f"{'='*65}")

    train_loader, val_loader, test_loader, scaler = make_loaders(
        dataset_name, input_len, pred_len, batch_size
    )

    config = {**DEFAULT_CONFIG, "input_len": input_len,
              "pred_len": pred_len, "d_vars": d_vars}
    model  = build_da_moe(config, device=device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {param_count:,}")

    results = train(
        model, train_loader, val_loader, test_loader,
        device=device, epochs=epochs, lr=lr, patience=patience,
        ssl_epochs=ssl_epochs, log_every=1
    )

    if run_ablation_flag:
        print("\n--- Ablation Study ---")
        abl = run_ablation(model, train_loader, val_loader, test_loader,
                           device=device, epochs=max(epochs // 3, 5), lr=lr)
        results["ablation"] = abl

    if run_sensitivity:
        print("\n--- Top-k Sensitivity ---")
        sens = sensitivity_analysis_top_k(
            model, train_loader, val_loader, test_loader,
            k_values=[2, 3, 4, 5, 6, 8],
            device=device, epochs=max(epochs // 3, 5), lr=lr
        )
        results["sensitivity"] = sens

    return results


def run_multi_horizon(
    dataset_name: str,
    horizons: list  = [96, 192, 336, 720],
    **kwargs
) -> dict:
    """Run experiments across multiple forecast horizons and average results."""
    all_results = {}
    for h in horizons:
        r = run_single_experiment(dataset_name, pred_len=h, **kwargs)
        all_results[h] = r["final"]

    avg = {k: np.mean([all_results[h][k] for h in horizons])
           for k in ["MSE", "MAE"]}

    print(f"\n{'='*50}")
    print(f"Multi-horizon Summary: {dataset_name}")
    print(f"{'='*50}")
    print(f"{'Horizon':>10} {'MSE':>10} {'MAE':>10}")
    for h, m in all_results.items():
        print(f"{h:>10} {m['MSE']:>10.4f} {m['MAE']:>10.4f}")
    print(f"{'Average':>10} {avg['MSE']:>10.4f} {avg['MAE']:>10.4f}")

    return {"per_horizon": all_results, "average": avg}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DA-MoE Experiments")
    parser.add_argument("--dataset",    type=str,   default="ETTh1")
    parser.add_argument("--pred_len",   type=int,   default=96)
    parser.add_argument("--input_len",  type=int,   default=96)
    parser.add_argument("--d_vars",     type=int,   default=7)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--ssl_epochs", type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--patience",   type=int,   default=5)
    parser.add_argument("--device",     type=str,   default="cpu")
    parser.add_argument("--ablation",   action="store_true",
                        help="Run ablation study")
    parser.add_argument("--sensitivity",action="store_true",
                        help="Run top-k sensitivity analysis")
    parser.add_argument("--multi_horizon", action="store_true",
                        help="Run across horizons {96,192,336,720}")
    parser.add_argument("--all_datasets", action="store_true",
                        help="Run across all synthetic benchmark datasets")
    args = parser.parse_args()

    device = (args.device if args.device != "auto"
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.all_datasets:
        datasets   = ["ETTh1", "ETTh2", "ETTm1", "ETTm2",
                      "Traffic", "Electricity", "Weather", "Solar"]
        summary    = {}
        for ds in datasets:
            r = run_multi_horizon(
                ds, horizons=[96, 192, 336, 720],
                input_len=args.input_len, d_vars=args.d_vars,
                batch_size=args.batch_size, epochs=args.epochs,
                ssl_epochs=args.ssl_epochs, lr=args.lr,
                patience=args.patience, device=device
            )
            summary[ds] = r["average"]

        print("\n" + "=" * 60)
        print("FULL BENCHMARK SUMMARY (averaged across horizons)")
        print("=" * 60)
        print(f"{'Dataset':<20} {'MSE':>10} {'MAE':>10}")
        print("-" * 44)
        for ds, m in summary.items():
            print(f"{ds:<20} {m['MSE']:>10.4f} {m['MAE']:>10.4f}")

    elif args.multi_horizon:
        run_multi_horizon(
            args.dataset, horizons=[96, 192, 336, 720],
            input_len=args.input_len, d_vars=args.d_vars,
            batch_size=args.batch_size, epochs=args.epochs,
            ssl_epochs=args.ssl_epochs, lr=args.lr,
            patience=args.patience, device=device
        )
    else:
        run_single_experiment(
            args.dataset, args.pred_len, args.input_len, args.d_vars,
            args.batch_size, args.epochs, args.ssl_epochs,
            args.lr, args.patience, device,
            run_ablation_flag=args.ablation,
            run_sensitivity=args.sensitivity,
        )


if __name__ == "__main__":
    main()
