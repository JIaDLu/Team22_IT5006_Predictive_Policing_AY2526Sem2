"""
Comprehensive evaluation module.

Metrics:
  - Global: MAE, RMSE, SMAPE
  - Spatial: per-region MAE  (shape: num_regions,)
  - Temporal: per-day MAE    (shape: num_samples,)

All metrics are computed in original (unscaled) crime_count space.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import DEVICE, TARGET_IDX


# ------------------------------------------------------------------
# Inverse scaling helper
# ------------------------------------------------------------------
def inverse_scale_target(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Inverse-transform the crime_count column from standardized space."""
    mean = scaler.mean_[TARGET_IDX]
    std = scaler.scale_[TARGET_IDX]
    return values * std + mean


# ------------------------------------------------------------------
# Metric functions (operate on original-scale arrays)
# ------------------------------------------------------------------
def compute_mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def compute_smape(pred: np.ndarray, true: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (0-200%)."""
    denom = np.abs(true) + np.abs(pred)
    mask = denom > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(2.0 * np.abs(true[mask] - pred[mask]) / denom[mask]) * 100)


def per_region_mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """MAE per region. pred/true: (S, N). Returns: (N,)."""
    return np.mean(np.abs(pred - true), axis=0)


def per_day_mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """MAE per day (sample). pred/true: (S, N). Returns: (S,)."""
    return np.mean(np.abs(pred - true), axis=1)


# ------------------------------------------------------------------
# Main evaluation entry point (requires torch)
# ------------------------------------------------------------------
def evaluate(
    model,
    loader,
    adj,
    scaler: StandardScaler,
    device: str = DEVICE,
) -> Dict[str, object]:
    """
    Run model on a DataLoader, compute all metrics in original scale.

    Returns dict with keys:
      'MAE', 'RMSE', 'SMAPE',
      'per_region_mae' (ndarray, shape N),
      'per_day_mae' (ndarray, shape S),
      'preds' (ndarray, shape (S,N)),
      'labels' (ndarray, shape (S,N)),
    """
    import torch  # deferred import

    model = model.to(device)
    model.eval()
    adj = adj.to(device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_hat = model(X_batch, adj)
            all_preds.append(Y_hat.cpu().numpy())
            all_labels.append(Y_batch.numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Inverse scale
    preds_orig = inverse_scale_target(preds, scaler)
    labels_orig = inverse_scale_target(labels, scaler)

    return {
        "MAE": compute_mae(preds_orig, labels_orig),
        "RMSE": compute_rmse(preds_orig, labels_orig),
        "SMAPE": compute_smape(preds_orig, labels_orig),
        "per_region_mae": per_region_mae(preds_orig, labels_orig),
        "per_day_mae": per_day_mae(preds_orig, labels_orig),
        "preds": preds_orig,
        "labels": labels_orig,
    }


def print_metrics(metrics: Dict[str, object], split_name: str = "Test") -> None:
    print(f"\n{'─' * 40}")
    print(f"  {split_name} Metrics (original scale)")
    print(f"{'─' * 40}")
    for k in ("MAE", "RMSE", "SMAPE"):
        print(f"  {k:>10s}: {metrics[k]:.4f}")
    print(f"{'─' * 40}\n")