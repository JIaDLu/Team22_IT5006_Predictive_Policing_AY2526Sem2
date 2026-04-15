"""
Flask API server for the crime prediction MVP.

Endpoints:
    GET  /                     → serve frontend
    GET  /api/datasets         → list available NIBRS datasets
    GET  /api/dates?dataset=X  → available prediction dates for a dataset
    POST /api/predict          → run inference, return predictions + agency info
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, jsonify, render_template, request

from src.config import (
    CHECKPOINT_DIR,
    NUM_REGIONS,
    SCALER_PATH,
    WINDOW_SIZE,
)
from src.utils.nibrs_pipeline import (
    NIBRS_BASE_DIR,
    STATE_CENTROIDS,
    align_to_chicago_format,
    build_inference_sample,
    discover_datasets,
    geocode_agencies,
    get_available_dates,
    join_nibrs_tables,
    load_nibrs_tables,
    parse_dataset_name,
)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

# ------------------------------------------------------------------
# Cache: avoid re-loading tables on every request
# ------------------------------------------------------------------
_dataset_cache: dict = {}


def _get_dataset(name: str):
    """Load and cache a NIBRS dataset."""
    if name not in _dataset_cache:
        dataset_dir = str(Path(NIBRS_BASE_DIR) / name)
        tables = load_nibrs_tables(dataset_dir)
        events = join_nibrs_tables(tables)
        aligned, agency_meta = align_to_chicago_format(events, max_regions=NUM_REGIONS)
        agency_meta = geocode_agencies(agency_meta)
        _dataset_cache[name] = {
            "aligned": aligned,
            "agency_meta": agency_meta,
            "state_abbr": agency_meta["state_abbr"].iloc[0]
            if "state_abbr" in agency_meta.columns and len(agency_meta) > 0
            else parse_dataset_name(name)[0],
        }
    return _dataset_cache[name]


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/datasets")
def api_datasets():
    datasets = discover_datasets()
    return jsonify({"datasets": datasets})


@app.route("/api/dates")
def api_dates():
    dataset = request.args.get("dataset", "")
    if not dataset:
        return jsonify({"error": "Missing dataset parameter"}), 400
    try:
        data = _get_dataset(dataset)
        dates = get_available_dates(data["aligned"])
        abbr = data["state_abbr"]
        center = STATE_CENTROIDS.get(abbr, (39.83, -98.58))
        return jsonify({
            "dates": dates,
            "state_abbr": abbr,
            "center": {"lat": center[0], "lon": center[1]},
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True)
    dataset = body.get("dataset", "")
    target_date = body.get("date", "")

    if not dataset or not target_date:
        return jsonify({"error": "Missing dataset or date"}), 400

    try:
        import torch
        from src.inference import load_baseline_gru, predict
        from src.utils.data_pipeline import load_scaler
        from src.evaluate import inverse_scale_target

        data = _get_dataset(dataset)
        aligned = data["aligned"]
        agency_meta = data["agency_meta"]

        num_agencies = len(agency_meta)
        region_ids = list(range(1, max(num_agencies, NUM_REGIONS) + 1))

        scaler = load_scaler(SCALER_PATH)
        model = load_baseline_gru(
            f"{CHECKPOINT_DIR}/Baseline-GRU.pt",
            num_regions=NUM_REGIONS,
        )

        X, Y = build_inference_sample(aligned, target_date, region_ids, scaler)
        preds = predict(model, X, scaler)  # (1, 77)

        # Build response
        agencies_out = []
        for _, row in agency_meta.iterrows():
            rid = int(row["region_id"]) - 1
            entry = {
                "region_id": int(row["region_id"]),
                "agency_name": str(row["pub_agency_name"]),
                "state": str(row.get("state_name", "")),
                "latitude": float(row.get("latitude", 0)),
                "longitude": float(row.get("longitude", 0)),
                "predicted": round(float(preds[0, rid]), 2) if rid < preds.shape[1] else 0,
            }
            if Y is not None and rid < Y.shape[1]:
                entry["actual"] = round(float(inverse_scale_target(Y, scaler)[0, rid]), 2)
            agencies_out.append(entry)

        return jsonify({
            "target_date": target_date,
            "agencies": agencies_out,
            "total_predicted": round(float(preds[0, :num_agencies].sum()), 2),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crime prediction MVP server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Starting server on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)