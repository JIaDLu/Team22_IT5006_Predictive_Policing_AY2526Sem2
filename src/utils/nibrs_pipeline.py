"""
NIBRS data processing pipeline.

Reads the multi-table NIBRS download (incident, offense, offense_type, agencies),
joins them into event-level records, and aligns the result to the same schema
used by the Chicago crime pipeline so that the same preprocessing + model
can be reused directly.

Output schema (mirrors Chicago):
    ID | Date | Community Area | Primary Type | Latitude | Longitude
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import COUNT_FEATURES, FEATURE_COLS, WINDOW_SIZE

# ------------------------------------------------------------------
# US state centroids (for fallback geo-positioning)
# ------------------------------------------------------------------
STATE_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "AL": (32.81, -86.68), "AK": (64.24, -152.49), "AZ": (34.05, -111.09),
    "AR": (34.80, -92.20), "CA": (36.78, -119.42), "CO": (39.55, -105.78),
    "CT": (41.60, -72.76), "DE": (39.00, -75.50), "FL": (28.63, -82.45),
    "GA": (33.04, -83.64), "HI": (20.75, -156.50), "ID": (44.06, -114.74),
    "IL": (40.00, -89.00), "IN": (39.91, -86.28), "IA": (42.01, -93.21),
    "KS": (38.50, -98.00), "KY": (37.84, -84.27), "LA": (30.93, -91.70),
    "ME": (45.37, -69.24), "MD": (39.05, -76.64), "MA": (42.23, -71.53),
    "MI": (43.33, -84.54), "MN": (46.28, -94.31), "MS": (32.74, -89.68),
    "MO": (38.46, -92.30), "MT": (46.80, -110.36), "NE": (41.50, -99.90),
    "NV": (38.50, -117.02), "NH": (43.68, -71.58), "NJ": (40.06, -74.78),
    "NM": (34.31, -106.02), "NY": (42.17, -74.95), "NC": (35.63, -79.81),
    "ND": (47.53, -100.47), "OH": (40.42, -82.91), "OK": (35.47, -97.52),
    "OR": (43.80, -120.55), "PA": (40.80, -77.21), "RI": (41.68, -71.51),
    "SC": (33.84, -80.95), "SD": (44.30, -100.22), "TN": (35.86, -86.35),
    "TX": (31.05, -97.56), "UT": (39.32, -111.09), "VT": (44.07, -72.67),
    "VA": (37.77, -78.17), "WA": (47.40, -120.74), "WV": (38.60, -80.95),
    "WI": (44.50, -89.50), "WY": (42.76, -107.30), "DC": (38.90, -77.04),
}

# NIBRS offense_category_name → Chicago Primary Type
NIBRS_OFFENSE_MAP = {
    "Larceny/Theft Offenses": "THEFT",
    "Assault Offenses": "BATTERY",
}

# ------------------------------------------------------------------
# 1. Dataset discovery
# ------------------------------------------------------------------
NIBRS_BASE_DIR = "data/NIBRS"
GEOCODE_CACHE_PATH = "data/NIBRS/geocode_cache.json"


def discover_datasets(base_dir: str = NIBRS_BASE_DIR) -> List[str]:
    """
    Scan data/NIBRS/ for available dataset folders.
    Returns list of folder names like ['CA-2024', 'TX-2023'].
    """
    p = Path(base_dir)
    if not p.exists():
        return []
    datasets = sorted([
        d.name for d in p.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    return datasets


def parse_dataset_name(name: str) -> Tuple[str, str]:
    """Parse 'CA-2024' → ('CA', '2024')."""
    parts = name.split("-")
    if len(parts) == 2:
        return parts[0].upper(), parts[1]
    return name, ""


# ------------------------------------------------------------------
# 2. Load NIBRS tables
# ------------------------------------------------------------------
def load_nibrs_tables(dataset_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load the 4 required NIBRS CSVs from a dataset directory.
    """
    d = Path(dataset_dir)
    tables = {}

    file_map = {
        "incident": "NIBRS_incident.csv",
        "offense": "NIBRS_OFFENSE.csv",
        "offense_type": "NIBRS_OFFENSE_TYPE.csv",
        "agencies": "agencies.csv",
    }

    for key, filename in file_map.items():
        path = d / filename
        if not path.exists():
            # Try case-insensitive search
            candidates = list(d.glob(f"*{filename.lower().replace('.csv', '')}*"))
            if candidates:
                path = candidates[0]
            else:
                raise FileNotFoundError(f"Missing {filename} in {dataset_dir}")
        tables[key] = pd.read_csv(path, low_memory=False)
        print(f"  [{key}] {len(tables[key])} rows from {path.name}")

    return tables


# ------------------------------------------------------------------
# 3. Join tables → event-level records
# ------------------------------------------------------------------
def join_nibrs_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join offense + incident + offense_type + agencies → event-level DataFrame.
    """
    offense = tables["offense"][["offense_id", "incident_id", "offense_code"]].copy()
    incident = tables["incident"][["incident_id", "agency_id", "incident_date"]].copy()
    offense_type = tables["offense_type"][["offense_code", "offense_category_name"]].copy()
    agencies = tables["agencies"][[
        "agency_id", "pub_agency_name", "state_name", "state_abbr",
    ]].drop_duplicates(subset=["agency_id"]).copy()

    # Join offense → incident (get date + agency)
    df = offense.merge(incident, on="incident_id", how="inner")

    # Join → offense_type (get category name)
    df = df.merge(offense_type, on="offense_code", how="left")

    # Join → agencies (get agency name, state)
    df = df.merge(agencies, on="agency_id", how="left")

    print(f"[join] {len(df)} event-level records")
    return df


# ------------------------------------------------------------------
# 4. Align to Chicago schema
# ------------------------------------------------------------------
def align_to_chicago_format(
    df: pd.DataFrame,
    max_regions: int = 77,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform NIBRS event data into the Chicago-compatible schema.

    Steps:
    1. Map offense_category_name → Primary Type (THEFT / BATTERY / OTHER)
    2. Parse incident_date → Date
    3. Select top-N agencies (by incident count) → region_id 1..N
    4. Pad to max_regions if fewer agencies exist

    Returns
    -------
    aligned_df : DataFrame with columns [ID, Date, Community Area, Primary Type]
    agency_meta : DataFrame with [region_id, agency_id, pub_agency_name, state_name, state_abbr]
    """
    df = df.copy()

    # Map Primary Type
    df["Primary Type"] = df["offense_category_name"].map(NIBRS_OFFENSE_MAP).fillna("OTHER")

    # Parse date
    df["Date"] = pd.to_datetime(df["incident_date"], format="mixed", errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # Drop records without agency
    df.dropna(subset=["agency_id", "pub_agency_name"], inplace=True)
    df["agency_id"] = df["agency_id"].astype(int)

    # Select top-N agencies by incident count
    agency_counts = df.groupby("agency_id").size().reset_index(name="n")
    agency_counts.sort_values("n", ascending=False, inplace=True)
    top_agencies = agency_counts.head(max_regions)["agency_id"].tolist()

    # Filter to top agencies
    df = df[df["agency_id"].isin(top_agencies)].copy()

    # Create region_id mapping (1-based, sorted by descending count)
    agency_to_region = {aid: i + 1 for i, aid in enumerate(top_agencies)}
    df["Community Area"] = df["agency_id"].map(agency_to_region)

    # Build agency metadata
    agency_info = (
        df[["agency_id", "pub_agency_name", "state_name", "state_abbr"]]
        .drop_duplicates(subset=["agency_id"])
        .copy()
    )
    agency_info["region_id"] = agency_info["agency_id"].map(agency_to_region)
    agency_info.sort_values("region_id", inplace=True)
    agency_info.reset_index(drop=True, inplace=True)

    # Build aligned DataFrame
    aligned = pd.DataFrame({
        "ID": df["offense_id"].values,
        "Date": df["Date"].values,
        "Community Area": df["Community Area"].values,
        "Primary Type": df["Primary Type"].values,
    })

    num_agencies = len(top_agencies)
    print(f"[align] {len(aligned)} records, {num_agencies} agencies "
          f"(mapped to region 1-{num_agencies})")

    return aligned, agency_info


# ------------------------------------------------------------------
# 5. Date utilities
# ------------------------------------------------------------------
def get_available_dates(
    aligned_df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
) -> List[str]:
    """
    Return available prediction target dates (require window_size history).
    """
    aligned_df["_date"] = pd.to_datetime(aligned_df["Date"]).dt.date
    all_dates = sorted(aligned_df["_date"].unique())

    if len(all_dates) <= window_size:
        return []

    # Dates with enough history for a full window
    available = all_dates[window_size:]
    return [str(d) for d in available]


# ------------------------------------------------------------------
# 6. Build inference sample for a single date
# ------------------------------------------------------------------
def build_inference_sample(
    aligned_df: pd.DataFrame,
    target_date: str,
    region_ids: List[int],
    scaler,
    window_size: int = WINDOW_SIZE,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Build a single input sample X for inference at target_date.

    Parameters
    ----------
    aligned_df : Chicago-format DataFrame
    target_date : 'YYYY-MM-DD'
    region_ids : list of region_ids (1..N)
    scaler : pre-fit StandardScaler from training

    Returns
    -------
    X : ndarray (1, window_size, num_regions, num_features)
    Y : ndarray (1, num_regions) or None if target_date has no data
    """
    from src.utils.data_pipeline import build_daily_features, build_full_grid, apply_scaler

    t_date = date.fromisoformat(target_date)
    start = t_date - pd.Timedelta(days=window_size)
    end = t_date

    # Filter to date range
    df = aligned_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    mask = (df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)
    df = df[mask].copy()

    # Build features → grid → scale
    agg = build_daily_features(df)
    grid = build_full_grid(agg, region_ids, start, end)
    grid = apply_scaler(grid, scaler)

    # Reshape → (num_dates, num_regions, num_features)
    dates = sorted(grid["date"].unique())
    num_dates = len(dates)
    num_regions = len(region_ids)
    num_features = len(FEATURE_COLS)

    values = grid.sort_values(["date", "region_id"])[FEATURE_COLS].values
    cube = values.reshape(num_dates, num_regions, num_features)

    if num_dates < window_size + 1:
        # Not enough history — pad front with zeros
        pad_size = window_size + 1 - num_dates
        pad = np.zeros((pad_size, num_regions, num_features), dtype=np.float32)
        cube = np.concatenate([pad, cube], axis=0)
        num_dates = cube.shape[0]

    # X = last window_size days, Y = target day
    X = cube[-(window_size + 1):-1][np.newaxis]  # (1, W, N, F)
    Y = cube[-1, :, 0][np.newaxis]  # (1, N) — crime_count

    return X.astype(np.float32), Y.astype(np.float32)


# ------------------------------------------------------------------
# 7. Geocode cache utilities
# ------------------------------------------------------------------
def load_geocode_cache(path: str = GEOCODE_CACHE_PATH) -> Dict[str, Tuple[float, float]]:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_geocode_cache(cache: dict, path: str = GEOCODE_CACHE_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def geocode_agencies(
    agency_meta: pd.DataFrame,
    cache_path: str = GEOCODE_CACHE_PATH,
) -> pd.DataFrame:
    """
    Add latitude/longitude to agency metadata.

    Strategy:
    1. Check local JSON cache
    2. Try geopy Nominatim (if installed)
    3. Fall back to deterministic distribution around state centroid
    """
    cache = load_geocode_cache(cache_path)
    meta = agency_meta.copy()
    lats, lons = [], []
    updated = False

    for _, row in meta.iterrows():
        key = f"{row['pub_agency_name']}|{row.get('state_name', '')}"
        if key in cache:
            lats.append(cache[key][0])
            lons.append(cache[key][1])
            continue

        # Try geopy
        coord = _try_geopy(row["pub_agency_name"], row.get("state_name", ""))
        if coord:
            cache[key] = coord
            lats.append(coord[0])
            lons.append(coord[1])
            updated = True
            continue

        # Fallback: distribute around state centroid
        abbr = row.get("state_abbr", "")
        center = STATE_CENTROIDS.get(abbr, (39.83, -98.58))  # US center fallback
        offset_lat = (hash(key) % 1000 - 500) / 500 * 2.0
        offset_lon = (hash(key + "lon") % 1000 - 500) / 500 * 3.0
        coord = (center[0] + offset_lat, center[1] + offset_lon)
        cache[key] = coord
        lats.append(coord[0])
        lons.append(coord[1])
        updated = True

    meta["latitude"] = lats
    meta["longitude"] = lons

    if updated:
        save_geocode_cache(cache, cache_path)

    return meta


def _try_geopy(name: str, state: str) -> Optional[Tuple[float, float]]:
    """Attempt geocoding via geopy. Returns None if unavailable."""
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut
        geo = Nominatim(user_agent="crime_pred_mvp", timeout=5)
        query = f"{name}, {state}" if state else name
        loc = geo.geocode(query)
        if loc:
            return (loc.latitude, loc.longitude)
    except Exception:
        pass
    return None