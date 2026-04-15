"""
Unified preprocessing pipeline for spatiotemporal crime prediction.

Core design:
  - All steps are reusable across train / val / test splits.
  - Scaler is fit ONLY on training data; val/test call transform only.
  - Each split is processed independently (no cross-split leakage).
"""
from __future__ import annotations

import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    COUNT_FEATURES,
    CRIME_TYPE_MAP,
    FEATURE_COLS,
    NUM_REGIONS,
    REGION_IDS,
    TIME_FEATURES,
    WINDOW_SIZE,
)


# ------------------------------------------------------------------
# 1. Load & split
# ------------------------------------------------------------------
def load_raw_data(path: str) -> pd.DataFrame:
    """Read the raw Chicago crime CSV."""
    df = pd.read_csv(path)
    print(f"[load] Raw records: {len(df)}")
    return df


def split_by_date(
    df: pd.DataFrame,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal split into train / val / test before any preprocessing."""
    d = df["Date"].dt.date
    train_df = df[d <= date.fromisoformat(train_end)].copy()
    val_df = df[(d >= date.fromisoformat(val_start)) & (d <= date.fromisoformat(val_end))].copy()
    test_df = df[d >= date.fromisoformat(test_start)].copy()
    print(f"[split] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return train_df, val_df, test_df


# ------------------------------------------------------------------
# 2. Cleaning
# ------------------------------------------------------------------
REQUIRED_COLS = [
    "ID", "Date", "Community Area", "Primary Type",
    "Latitude", "Longitude",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Keep only required columns
    - Deduplicate on ID
    - Drop rows with missing Community Area or Date
    - Parse Date → datetime, sort ascending
    """
    df = df[REQUIRED_COLS].copy()
    n0 = len(df)

    # Deduplicate
    df.drop_duplicates(subset=["ID"], inplace=True)
    n1 = len(df)

    # Drop missing critical fields
    df.dropna(subset=["Date", "Community Area"], inplace=True)
    n2 = len(df)

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[clean] {n0} → dedup {n1} → drop_na {n2}")
    return df


# ------------------------------------------------------------------
# 3. Feature aggregation
# ------------------------------------------------------------------
def build_daily_features(
    df: pd.DataFrame,
    crime_type_map: Dict[str, str] = CRIME_TYPE_MAP,
) -> pd.DataFrame:
    """
    Aggregate raw records → (region_id, date) level features.
    Returns DataFrame with columns:
        region_id, date, crime_count, theft_count, battery_count
    """
    df = df.copy()
    df["date"] = df["Date"].dt.date
    df["region_id"] = df["Community Area"].astype(int)

    # One-hot flags for crime types
    for raw_label, col_name in crime_type_map.items():
        df[col_name] = (df["Primary Type"] == raw_label).astype(int)

    agg_dict = {"ID": "count"}  # total crime count
    for col_name in crime_type_map.values():
        agg_dict[col_name] = "sum"

    agg = df.groupby(["region_id", "date"]).agg(agg_dict).reset_index()
    agg.rename(columns={"ID": "crime_count"}, inplace=True)

    return agg


# ------------------------------------------------------------------
# 4. Full spatiotemporal grid
# ------------------------------------------------------------------
def build_full_grid(
    agg_df: pd.DataFrame,
    region_ids: List[int],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Cartesian product of regions × dates; left-join with aggregated data.
    Missing crime counts → 0.  Time features computed from date.
    """
    all_dates = pd.date_range(start_date, end_date, freq="D").date
    idx = pd.MultiIndex.from_product(
        [region_ids, all_dates], names=["region_id", "date"]
    )
    grid = pd.DataFrame(index=idx).reset_index()

    # Ensure matching dtypes for merge
    agg_df = agg_df.copy()
    agg_df["region_id"] = agg_df["region_id"].astype(int)
    agg_df["date"] = pd.to_datetime(agg_df["date"]).dt.date

    grid = grid.merge(agg_df, on=["region_id", "date"], how="left")

    # Fill missing count features with 0
    for col in COUNT_FEATURES:
        grid[col] = grid[col].fillna(0).astype(float)

    # Compute time features from date
    dt_series = pd.to_datetime(grid["date"])
    grid["day_of_week"] = dt_series.dt.dayofweek.astype(float)
    grid["is_weekend"] = (grid["day_of_week"] >= 5).astype(float)
    grid["month"] = dt_series.dt.month.astype(float)

    grid.sort_values(["date", "region_id"], inplace=True)
    grid.reset_index(drop=True, inplace=True)

    num_dates = len(all_dates)
    assert len(grid) == num_dates * len(region_ids), (
        f"Grid size mismatch: {len(grid)} != {num_dates}*{len(region_ids)}"
    )
    print(f"[grid] {len(region_ids)} regions × {num_dates} dates = {len(grid)} rows")
    return grid


# ------------------------------------------------------------------
# 5. Scaling
# ------------------------------------------------------------------
def fit_scaler(grid_df: pd.DataFrame, feature_cols: List[str] = FEATURE_COLS) -> StandardScaler:
    """Fit StandardScaler on training grid data."""
    scaler = StandardScaler()
    scaler.fit(grid_df[feature_cols].values)
    print(f"[scaler] fit on {len(grid_df)} rows, {len(feature_cols)} features")
    return scaler


def apply_scaler(
    grid_df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: List[str] = FEATURE_COLS,
) -> pd.DataFrame:
    """Transform features in-place using a pre-fit scaler."""
    grid_df = grid_df.copy()
    grid_df[feature_cols] = scaler.transform(grid_df[feature_cols].values)
    return grid_df


def save_scaler(scaler: StandardScaler, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[scaler] saved → {path}")


def load_scaler(path: str) -> StandardScaler:
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------
# 6. Sliding window sample construction
# ------------------------------------------------------------------
def build_samples(
    grid_df: pd.DataFrame,
    region_ids: List[int],
    feature_cols: List[str] = FEATURE_COLS,
    window_size: int = WINDOW_SIZE,
    target_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window construction on the full grid.

    Returns
    -------
    X : np.ndarray, shape (num_samples, window_size, num_regions, num_features)
    Y : np.ndarray, shape (num_samples, num_regions)
        Target = scaled crime_count at the label day.
    """
    num_regions = len(region_ids)
    dates = sorted(grid_df["date"].unique())
    num_dates = len(dates)

    # Reshape grid → (num_dates, num_regions, num_features)
    values = grid_df.sort_values(["date", "region_id"])[feature_cols].values
    data_cube = values.reshape(num_dates, num_regions, len(feature_cols))

    X_list, Y_list = [], []
    for t in range(window_size, num_dates):
        X_list.append(data_cube[t - window_size : t])  # (W, N, F)
        Y_list.append(data_cube[t, :, target_idx])      # (N,)

    X = np.array(X_list, dtype=np.float32)  # (S, W, N, F)
    Y = np.array(Y_list, dtype=np.float32)  # (S, N)
    print(f"[samples] X={X.shape}  Y={Y.shape}")
    return X, Y


# ------------------------------------------------------------------
# 7. Unified pipeline entry point
# ------------------------------------------------------------------
def preprocess_split(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    scaler: Optional[StandardScaler] = None,
    fit_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    End-to-end pipeline for a single temporal split.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (already temporally sliced) crime records.
    start_date, end_date : str
        Date range for the full grid (inclusive).
    scaler : StandardScaler or None
        Pre-fit scaler for transform-only mode.
    fit_mode : bool
        If True, fit a new scaler on this split (training only).

    Returns
    -------
    X, Y, scaler
    """
    df = clean_data(df)
    agg = build_daily_features(df)
    grid = build_full_grid(
        agg,
        region_ids=REGION_IDS,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
    )

    if fit_mode:
        scaler = fit_scaler(grid)

    assert scaler is not None, "Scaler must be provided or fit_mode=True"
    grid = apply_scaler(grid, scaler)

    X, Y = build_samples(grid, region_ids=REGION_IDS)
    return X, Y, scaler