#!/usr/bin/env python
"""
Generate comparison plots and summary tables from saved experiment results.

Usage:
    # Compare baselines only
    python scripts/compare.py --runs Baseline-LSTM Baseline-GRU Baseline-MHA \
                              --tag baselines

    # Compare best baseline vs all ST-GNN models
    python scripts/compare.py --runs Baseline-LSTM STGNN-LSTM STGNN-GRU STGNN-MHA \
                              --tag stgnn_vs_baseline

    # Compare all 6 models
    python scripts/compare.py --runs Baseline-LSTM Baseline-GRU Baseline-MHA \
                                     STGNN-LSTM STGNN-GRU STGNN-MHA \
                              --tag all_models
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FIGURES_DIR, RESULTS_DIR
from src.utils.experiment import load_run_results
from src.utils.visualization import (
    build_summary_table,
    plot_per_region_mae,
    plot_radar,
    plot_temporal_mae,
    plot_training_curves,
)


def main():
    parser = argparse.ArgumentParser(description="Generate comparison outputs")
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="Run names to compare (e.g. Baseline-LSTM STGNN-LSTM)",
    )
    parser.add_argument(
        "--tag", type=str, default="comparison",
        help="Tag for output filenames (default: comparison)",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="Which split's metrics to visualize (default: test)",
    )
    args = parser.parse_args()

    out_dir = Path(FIGURES_DIR) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    # ----------------------------------------------------------------
    # Load results
    # ----------------------------------------------------------------
    all_results = {}
    for name in args.runs:
        try:
            all_results[name] = load_run_results(name)
            print(f"  ✓ Loaded {name}")
        except FileNotFoundError:
            print(f"  ✗ {name} not found in {RESULTS_DIR}/, skipping")

    if len(all_results) < 2:
        print("Need at least 2 runs to compare. Exiting.")
        return

    split = args.split

    # ----------------------------------------------------------------
    # 1. Summary table (CSV)
    # ----------------------------------------------------------------
    scalar_results = {
        name: data["metrics"][split] for name, data in all_results.items()
    }
    df = build_summary_table(
        scalar_results,
        metrics=["MAE", "RMSE", "SMAPE"],
        save_path=str(out_dir / "summary.csv"),
    )
    print(f"\n{df.to_string()}\n")

    # ----------------------------------------------------------------
    # 2. Radar chart
    # ----------------------------------------------------------------
    plot_radar(
        scalar_results,
        metrics=["MAE", "RMSE", "SMAPE"],
        title=f"Model Comparison — {split.capitalize()} Set ({args.tag})",
        save_path=str(out_dir / "radar.png"),
    )

    # ----------------------------------------------------------------
    # 3. Per-region MAE (spatial)
    # ----------------------------------------------------------------
    region_maes = {
        name: data[f"{split}_per_region_mae"]
        for name, data in all_results.items()
    }
    plot_per_region_mae(
        region_maes,
        title=f"Per-Region MAE — {split.capitalize()} Set",
        save_path=str(out_dir / "per_region_mae.png"),
    )

    # ----------------------------------------------------------------
    # 4. Temporal MAE
    # ----------------------------------------------------------------
    day_maes = {
        name: data[f"{split}_per_day_mae"]
        for name, data in all_results.items()
    }
    plot_temporal_mae(
        day_maes,
        title=f"Temporal MAE — {split.capitalize()} Set",
        save_path=str(out_dir / "temporal_mae.png"),
    )

    # ----------------------------------------------------------------
    # 5. Training curves
    # ----------------------------------------------------------------
    histories = {
        name: data["history"] for name, data in all_results.items()
    }
    plot_training_curves(
        histories,
        title=f"Training & Validation Loss ({args.tag})",
        save_path=str(out_dir / "training_curves.png"),
    )

    print(f"\n✓ All plots saved to {out_dir}/")


if __name__ == "__main__":
    main()