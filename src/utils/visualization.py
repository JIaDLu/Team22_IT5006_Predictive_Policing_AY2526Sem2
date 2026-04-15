"""
Visualization utilities for experiment comparison.

Academic-style plots with consistent palette and clear annotations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# Academic color palette (colorblind-friendly)
# ============================================================
PALETTE = {
    "Baseline-LSTM":  "#4C72B0",
    "Baseline-GRU":   "#55A868",
    "Baseline-MHA":   "#C44E52",
    "STGNN-LSTM":     "#8172B2",
    "STGNN-GRU":      "#CCB974",
    "STGNN-MHA":      "#64B5CD",
}

# Fallback ordered colors when model names don't match palette
ORDERED_COLORS = list(PALETTE.values())


def _get_color(name: str, idx: int = 0) -> str:
    return PALETTE.get(name, ORDERED_COLORS[idx % len(ORDERED_COLORS)])


def _setup_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


# ============================================================
# 1. Radar chart — multi-metric comparison
# ============================================================
def plot_radar(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
):
    """
    Radar chart comparing multiple models on multiple metrics.

    Parameters
    ----------
    results : {model_name: {metric_name: value}}
    metrics : list of metric names to plot
    """
    _setup_style()
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # Normalize each metric to [0, 1] range for fair radar comparison
    all_vals = {m: [] for m in metrics}
    for model_results in results.values():
        for m in metrics:
            all_vals[m].append(model_results[m])

    max_vals = {m: max(v) if max(v) > 0 else 1.0 for m, v in all_vals.items()}
    min_vals = {m: min(v) for m, v in all_vals.items()}

    for idx, (name, model_results) in enumerate(results.items()):
        # For error metrics: lower is better → invert for radar (higher = better)
        values = []
        for m in metrics:
            rng = max_vals[m] - min_vals[m]
            if rng > 0:
                normalized = (model_results[m] - min_vals[m]) / rng
                values.append(1.0 - normalized)  # invert: lower error → higher score
            else:
                values.append(1.0)
        values += values[:1]

        color = _get_color(name, idx)
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[plot] saved → {save_path}")
    plt.close(fig)


# ============================================================
# 2. Per-region MAE — sorted bar chart + boxplot
# ============================================================
def plot_per_region_mae(
    region_maes: Dict[str, np.ndarray],
    title: str = "Per-Region MAE Comparison",
    save_path: Optional[str] = None,
):
    """
    Top: sorted bar chart for each model.
    Bottom: boxplot comparison.
    """
    _setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1.5]})

    # --- Bar chart (sorted by first model's MAE) ---
    ax = axes[0]
    first_key = list(region_maes.keys())[0]
    sort_idx = np.argsort(region_maes[first_key])[::-1]
    x = np.arange(len(sort_idx))
    width = 0.8 / len(region_maes)

    for i, (name, mae_arr) in enumerate(region_maes.items()):
        color = _get_color(name, i)
        ax.bar(x + i * width, mae_arr[sort_idx], width, label=name, color=color, alpha=0.85)

    ax.set_xlabel("Region (sorted by MAE)")
    ax.set_ylabel("MAE")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.set_xticks([])

    # --- Boxplot ---
    ax2 = axes[1]
    box_data = []
    box_labels = []
    box_colors = []
    for i, (name, mae_arr) in enumerate(region_maes.items()):
        box_data.append(mae_arr)
        box_labels.append(name)
        box_colors.append(_get_color(name, i))

    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel("MAE Distribution")
    ax2.set_title("Per-Region MAE Distribution", fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[plot] saved → {save_path}")
    plt.close(fig)


# ============================================================
# 3. Temporal MAE — line chart with rolling average
# ============================================================
def plot_temporal_mae(
    day_maes: Dict[str, np.ndarray],
    rolling_window: int = 7,
    title: str = "Temporal MAE Over Time",
    save_path: Optional[str] = None,
):
    """
    Line chart of per-day MAE with rolling smoothed overlay.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    for idx, (name, mae_arr) in enumerate(day_maes.items()):
        color = _get_color(name, idx)
        days = np.arange(len(mae_arr))

        # Raw (faint)
        ax.plot(days, mae_arr, alpha=0.25, color=color, linewidth=0.8)

        # Rolling average (bold)
        if len(mae_arr) >= rolling_window:
            rolling = pd.Series(mae_arr).rolling(rolling_window, center=True).mean().values
            ax.plot(days, rolling, color=color, linewidth=2.0, label=f"{name} (rolling {rolling_window}d)")
        else:
            ax.plot(days, mae_arr, color=color, linewidth=2.0, label=name)

    ax.set_xlabel("Day Index")
    ax.set_ylabel("MAE")
    ax.set_title(title, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[plot] saved → {save_path}")
    plt.close(fig)


# ============================================================
# 4. Training curves
# ============================================================
def plot_training_curves(
    histories: Dict[str, dict],
    title: str = "Training & Validation Loss",
    save_path: Optional[str] = None,
):
    """Plot train/val loss curves for multiple models."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, hist) in enumerate(histories.items()):
        color = _get_color(name, idx)
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], color=color, linewidth=1.5, label=name)
        axes[1].plot(epochs, hist["val_loss"], color=color, linewidth=1.5, label=name)

    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber Loss")
    axes[0].legend()

    axes[1].set_title("Validation Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Huber Loss")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[plot] saved → {save_path}")
    plt.close(fig)


# ============================================================
# 5. Summary table
# ============================================================
def build_summary_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ("MAE", "RMSE", "SMAPE"),
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build comparison summary table and optionally save as CSV.
    """
    rows = []
    for name, r in results.items():
        row = {"Model": name}
        for m in metrics:
            row[m] = r[m]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path)
        print(f"[table] saved → {save_path}")

    return df