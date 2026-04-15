"""
Model inference utilities.

Loads a trained Baseline-GRU checkpoint and runs predictions
on new data (Chicago test set or NIBRS generalization data).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.config import (
    CHECKPOINT_DIR,
    DEVICE,
    NUM_FEATURES,
    NUM_REGIONS,
    SCALER_PATH,
    TARGET_IDX,
    TEMPORAL_HIDDEN_DIM,
    WINDOW_SIZE,
)
from src.evaluate import inverse_scale_target
from src.models.baseline import BaselineModel
from src.utils.data_pipeline import load_scaler


def load_baseline_gru(
    checkpoint_path: str = f"{CHECKPOINT_DIR}/Baseline-GRU.pt",
    num_regions: int = NUM_REGIONS,
    num_features: int = NUM_FEATURES,
    device: str = DEVICE,
) -> BaselineModel:
    """
    Load a trained Baseline-GRU model from checkpoint.
    """
    model = BaselineModel(
        temporal_type="gru",
        num_features=num_features,
        num_regions=num_regions,
        temporal_hidden=TEMPORAL_HIDDEN_DIM,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[model] Loaded Baseline-GRU from {checkpoint_path}")
    return model


def predict(
    model: BaselineModel,
    X: np.ndarray,
    scaler: StandardScaler,
    device: str = DEVICE,
) -> np.ndarray:
    """
    Run inference and return predictions in original (unscaled) space.

    Parameters
    ----------
    X : ndarray (B, T, N, F) — scaled input
    scaler : StandardScaler fitted on training data

    Returns
    -------
    preds : ndarray (B, N) — predicted crime_count in original scale
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        Y_hat = model(X_tensor)  # (B, N)

    preds_scaled = Y_hat.cpu().numpy()
    preds_orig = inverse_scale_target(preds_scaled, scaler)
    return preds_orig


def run_nibrs_inference(
    dataset_dir: str,
    target_date: str,
    checkpoint_path: str = f"{CHECKPOINT_DIR}/Baseline-GRU.pt",
    scaler_path: str = SCALER_PATH,
) -> Dict:
    """
    End-to-end NIBRS inference for a single target date.

    Returns
    -------
    dict with keys:
        'predictions' : list of float (per-region predicted crime_count)
        'actuals'     : list of float (per-region actual crime_count, unscaled)
        'agencies'    : list of dict with agency metadata
        'target_date' : str
    """
    from src.utils.nibrs_pipeline import (
        load_nibrs_tables,
        join_nibrs_tables,
        align_to_chicago_format,
        geocode_agencies,
        build_inference_sample,
    )

    # Load tables and align
    tables = load_nibrs_tables(dataset_dir)
    events = join_nibrs_tables(tables)
    aligned, agency_meta = align_to_chicago_format(events, max_regions=NUM_REGIONS)

    # Geocode agencies
    agency_meta = geocode_agencies(agency_meta)

    # Determine actual region_ids present
    num_agencies = len(agency_meta)
    region_ids = list(range(1, num_agencies + 1))

    # Pad to 77 if fewer agencies
    if num_agencies < NUM_REGIONS:
        region_ids = list(range(1, NUM_REGIONS + 1))

    # Load scaler + model
    scaler = load_scaler(scaler_path)
    model = load_baseline_gru(checkpoint_path, num_regions=NUM_REGIONS)

    # Build sample
    X, Y = build_inference_sample(aligned, target_date, region_ids, scaler)

    # Predict
    preds = predict(model, X, scaler)  # (1, N)

    # Map results to agencies
    results = {
        "target_date": target_date,
        "predictions": preds[0].tolist(),
        "actuals": inverse_scale_target(Y, scaler)[0].tolist() if Y is not None else None,
        "agencies": [],
    }

    for _, row in agency_meta.iterrows():
        rid = int(row["region_id"]) - 1  # 0-indexed
        results["agencies"].append({
            "region_id": int(row["region_id"]),
            "agency_name": row["pub_agency_name"],
            "state": row.get("state_name", ""),
            "latitude": float(row.get("latitude", 0)),
            "longitude": float(row.get("longitude", 0)),
            "predicted": float(preds[0, rid]) if rid < preds.shape[1] else 0.0,
            "actual": float(inverse_scale_target(Y, scaler)[0, rid])
            if Y is not None and rid < Y.shape[1] else None,
        })

    return results