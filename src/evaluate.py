"""
Model evaluation with inverse-scaled metrics.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.config import DEVICE, TARGET_IDX
from src.models.stgnn import STGNN


def evaluate(
    model: STGNN,
    loader: DataLoader,
    adj: torch.Tensor,
    scaler: StandardScaler,
    device: str = DEVICE,
) -> Dict[str, float]:
    """
    Evaluate model on a DataLoader and return metrics in original scale.

    Metrics: MAE, RMSE, MAPE (masked where true > 0).
    """
    model = model.to(device)
    model.eval()
    adj = adj.to(device)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_hat = model(X_batch, adj)  # (B, N)
            all_preds.append(Y_hat.cpu().numpy())
            all_labels.append(Y_batch.numpy())

    preds = np.concatenate(all_preds, axis=0)   # (S, N)
    labels = np.concatenate(all_labels, axis=0)  # (S, N)

    # Inverse-transform crime_count (target_idx) from scaled space
    preds_orig = _inverse_transform_target(preds, scaler)
    labels_orig = _inverse_transform_target(labels, scaler)

    # Flatten for global metrics
    p = preds_orig.flatten()
    l = labels_orig.flatten()

    mae = np.mean(np.abs(p - l))
    rmse = np.sqrt(np.mean((p - l) ** 2))

    # MAPE — only where label > 0 to avoid division by zero
    mask = l > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((l[mask] - p[mask]) / l[mask])) * 100
    else:
        mape = float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape}


def _inverse_transform_target(
    values: np.ndarray, scaler: StandardScaler
) -> np.ndarray:
    """
    Inverse-transform only the target column (crime_count) from scaled space.

    values : (S, N) — scaled predictions or labels for the target feature.
    """
    mean = scaler.mean_[TARGET_IDX]
    std = scaler.scale_[TARGET_IDX]
    return values * std + mean


def print_metrics(metrics: Dict[str, float], split_name: str = "Test") -> None:
    print(f"\n{'─' * 40}")
    print(f"  {split_name} Metrics (original scale)")
    print(f"{'─' * 40}")
    for k, v in metrics.items():
        print(f"  {k:>10s}: {v:.4f}")
    print(f"{'─' * 40}\n")