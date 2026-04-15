"""
Shared experiment utilities.

Handles data loading, W&B init, and result persistence
so that run_baseline.py and run_stgnn.py stay lean.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.config import (
    ADJACENCY_PATH,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    RAW_DATA_PATH,
    REGION_IDS,
    RESULTS_DIR,
    SCALER_PATH,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)
from src.utils.adjacency import build_adjacency, load_adjacency, save_adjacency
from src.utils.data_pipeline import (
    clean_data,
    load_raw_data,
    preprocess_split,
    save_scaler,
    load_scaler,
)
from src.utils.dataset import create_dataloader


def prepare_data(force_rebuild: bool = False):
    """
    Load or rebuild preprocessed data and adjacency matrix.

    Returns
    -------
    train_loader, val_loader, test_loader, adj_tensor, scaler
    """
    scaler_exists = Path(SCALER_PATH).exists()
    adj_exists = Path(ADJACENCY_PATH).exists()

    if not force_rebuild and scaler_exists and adj_exists:
        print("[data] Loading cached scaler and adjacency ...")
        scaler = load_scaler(SCALER_PATH)
        adj_np = load_adjacency(ADJACENCY_PATH)
    else:
        print("[data] Building from scratch ...")
        scaler = None
        adj_np = None

    # Always rebuild X/Y from raw CSV (safe and deterministic)
    raw_df = load_raw_data(RAW_DATA_PATH)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"], format="mixed")

    d = raw_df["Date"].dt.date
    train_raw = raw_df[d <= date.fromisoformat(TRAIN_END)].copy()
    val_raw = raw_df[(d >= date.fromisoformat(VAL_START)) & (d <= date.fromisoformat(VAL_END))].copy()
    test_raw = raw_df[d >= date.fromisoformat(TEST_START)].copy()
    print(f"[split] train={len(train_raw)}  val={len(val_raw)}  test={len(test_raw)}")

    # Adjacency
    if adj_np is None:
        train_cleaned = clean_data(train_raw.copy())
        adj_np = build_adjacency(train_cleaned)
        save_adjacency(adj_np, ADJACENCY_PATH)

    # Preprocessing pipeline
    if scaler is None:
        X_train, Y_train, scaler = preprocess_split(
            train_raw, TRAIN_START, TRAIN_END, fit_mode=True
        )
        save_scaler(scaler, SCALER_PATH)
    else:
        X_train, Y_train, _ = preprocess_split(
            train_raw, TRAIN_START, TRAIN_END, scaler=scaler, fit_mode=False
        )

    X_val, Y_val, _ = preprocess_split(
        val_raw, VAL_START, VAL_END, scaler=scaler, fit_mode=False
    )
    X_test, Y_test, _ = preprocess_split(
        test_raw, TEST_START, TEST_END, scaler=scaler, fit_mode=False
    )

    # DataLoaders
    train_loader = create_dataloader(X_train, Y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(X_val, Y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(X_test, Y_test, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[shapes] X_train={X_train.shape}  X_val={X_val.shape}  X_test={X_test.shape}")

    adj_tensor = torch.tensor(adj_np, dtype=torch.float32)
    return train_loader, val_loader, test_loader, adj_tensor, scaler


def save_run_results(
    run_name: str,
    history: dict,
    val_metrics: dict,
    test_metrics: dict,
) -> Path:
    """
    Save metrics and per-region/per-day arrays to results/<run_name>/.
    """
    out_dir = Path(RESULTS_DIR) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scalar metrics
    scalars = {
        "val": {k: v for k, v in val_metrics.items() if isinstance(v, (int, float))},
        "test": {k: v for k, v in test_metrics.items() if isinstance(v, (int, float))},
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(scalars, f, indent=2)

    # Arrays
    np.save(out_dir / "val_per_region_mae.npy", val_metrics["per_region_mae"])
    np.save(out_dir / "val_per_day_mae.npy", val_metrics["per_day_mae"])
    np.save(out_dir / "test_per_region_mae.npy", test_metrics["per_region_mae"])
    np.save(out_dir / "test_per_day_mae.npy", test_metrics["per_day_mae"])

    # History
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f)

    print(f"[results] saved → {out_dir}")
    return out_dir


def load_run_results(run_name: str) -> dict:
    """Load saved results for a given run name."""
    d = Path(RESULTS_DIR) / run_name
    with open(d / "metrics.json") as f:
        metrics = json.load(f)
    with open(d / "history.json") as f:
        history = json.load(f)
    return {
        "metrics": metrics,
        "history": history,
        "test_per_region_mae": np.load(d / "test_per_region_mae.npy"),
        "test_per_day_mae": np.load(d / "test_per_day_mae.npy"),
        "val_per_region_mae": np.load(d / "val_per_region_mae.npy"),
        "val_per_day_mae": np.load(d / "val_per_day_mae.npy"),
    }