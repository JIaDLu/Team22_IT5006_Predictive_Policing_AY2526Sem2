#!/usr/bin/env python
"""
Pre-geocode NIBRS agencies and cache results.

Usage:
    python scripts/geocode_agencies.py --dataset CA-2024

Requires: pip install geopy
Without geopy, agencies will use fallback positioning.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import NUM_REGIONS
from src.utils.nibrs_pipeline import (
    NIBRS_BASE_DIR,
    align_to_chicago_format,
    discover_datasets,
    geocode_agencies,
    join_nibrs_tables,
    load_nibrs_tables,
)


def main():
    parser = argparse.ArgumentParser(description="Geocode NIBRS agencies")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name (e.g. CA-2024). Empty = all datasets.")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else discover_datasets()
    if not datasets:
        print("No NIBRS datasets found.")
        return

    for name in datasets:
        print(f"\n{'='*50}")
        print(f"  Geocoding: {name}")
        print(f"{'='*50}")

        dataset_dir = str(Path(NIBRS_BASE_DIR) / name)
        tables = load_nibrs_tables(dataset_dir)
        events = join_nibrs_tables(tables)
        _, agency_meta = align_to_chicago_format(events, max_regions=NUM_REGIONS)

        meta = geocode_agencies(agency_meta)
        geocoded = meta[meta["latitude"] != 0].shape[0]
        print(f"  Geocoded: {geocoded}/{len(meta)} agencies")

    print("\n✓ Done. Cache saved to data/NIBRS/geocode_cache.json")


if __name__ == "__main__":
    main()