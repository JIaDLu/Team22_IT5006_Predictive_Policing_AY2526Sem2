#!/usr/bin/env python
"""
Run baseline experiments (temporal-only, no GNN).

Usage:
    python scripts/run_baseline.py --temporal lstm
    python scripts/run_baseline.py --temporal gru
    python scripts/run_baseline.py --temporal mha
    python scripts/run_baseline.py --temporal all   # run all three
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import (
    CHECKPOINT_DIR,
    DEVICE,
    TEMPORAL_HIDDEN_DIM,
    WANDB_PROJECT,
)
from src.evaluate import evaluate, print_metrics
from src.models.baseline import BaselineModel
from src.train import train_model
from src.utils.experiment import prepare_data, save_run_results

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def run_single_baseline(temporal_type: str, train_loader, val_loader, test_loader,
                         adj_tensor, scaler, use_wandb: bool = False):
    """Run one baseline experiment."""
    run_name = f"Baseline-{temporal_type.upper()}"
    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"{'='*60}")

    # W&B init
    if use_wandb and HAS_WANDB:
        wandb.init(
            project=WANDB_PROJECT.split("/")[-1],
            entity=WANDB_PROJECT.split("/")[0],
            name=run_name,
            config={
                "model_type": "baseline",
                "temporal": temporal_type,
                "hidden_dim": TEMPORAL_HIDDEN_DIM,
            },
            reinit=True,
        )

    model = BaselineModel(temporal_type=temporal_type)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    save_path = f"{CHECKPOINT_DIR}/{run_name}.pt"
    history = train_model(
        model, train_loader, val_loader, adj_tensor,
        save_path=save_path, use_wandb=use_wandb,
    )

    val_metrics = evaluate(model, val_loader, adj_tensor, scaler)
    print_metrics(val_metrics, f"{run_name} Validation")

    test_metrics = evaluate(model, test_loader, adj_tensor, scaler)
    print_metrics(test_metrics, f"{run_name} Test")

    # W&B log final metrics
    if use_wandb and HAS_WANDB:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()
                    if isinstance(v, (int, float))})
        wandb.finish()

    save_run_results(run_name, history, val_metrics, test_metrics)
    return run_name, history, val_metrics, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument(
        "--temporal", type=str, default="all",
        choices=["lstm", "gru", "mha", "all"],
        help="Temporal encoder type (default: all)",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    # Prepare data (shared across all baseline runs)
    train_loader, val_loader, test_loader, adj_tensor, scaler = prepare_data()

    types = ["lstm", "gru", "mha"] if args.temporal == "all" else [args.temporal]
    for t in types:
        run_single_baseline(
            t, train_loader, val_loader, test_loader,
            adj_tensor, scaler, use_wandb=args.wandb,
        )

    print("\n✓ All baseline experiments complete.")


if __name__ == "__main__":
    main()