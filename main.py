"""
main.py — Full pipeline: data preprocessing → model training → evaluation.

Usage:
    python main.py
"""
import pandas as pd
import torch

from src.config import (
    ADJACENCY_PATH,
    BATCH_SIZE,
    DEVICE,
    RAW_DATA_PATH,
    SCALER_PATH,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)
from src.evaluate import evaluate, print_metrics
from src.models.stgnn import STGNN
from src.train import train_model
from src.utils.adjacency import build_adjacency, save_adjacency
from src.utils.data_pipeline import (
    clean_data,
    load_raw_data,
    preprocess_split,
    save_scaler,
    split_by_date,
)
from src.utils.dataset import create_dataloader, print_data_summary


def main():
    # ==========================================================
    # 1. Load raw data & temporal split (BEFORE any preprocessing)
    # ==========================================================
    raw_df = load_raw_data(RAW_DATA_PATH)

    # Parse Date early so we can split
    raw_df["Date"] = pd.to_datetime(raw_df["Date"], format="mixed")
    train_raw, val_raw, test_raw = split_by_date(
        raw_df,
        train_end=TRAIN_END,
        val_start=VAL_START,
        val_end=VAL_END,
        test_start=TEST_START,
    )

    # ==========================================================
    # 2. Build adjacency matrix (from training data only)
    # ==========================================================
    train_cleaned = clean_data(train_raw.copy())
    adj_np = build_adjacency(train_cleaned)
    save_adjacency(adj_np, ADJACENCY_PATH)
    adj_tensor = torch.tensor(adj_np, dtype=torch.float32)

    # ==========================================================
    # 3. Preprocess each split through unified pipeline
    # ==========================================================
    # Training — fit scaler
    X_train, Y_train, scaler = preprocess_split(
        train_raw, TRAIN_START, TRAIN_END, fit_mode=True
    )
    save_scaler(scaler, SCALER_PATH)

    # Validation — transform only
    X_val, Y_val, _ = preprocess_split(
        val_raw, VAL_START, VAL_END, scaler=scaler, fit_mode=False
    )

    # Test — transform only
    X_test, Y_test, _ = preprocess_split(
        test_raw, TEST_START, TEST_END, scaler=scaler, fit_mode=False
    )

    # ==========================================================
    # 4. Shape verification
    # ==========================================================
    print_data_summary(X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_np)

    # ==========================================================
    # 5. DataLoaders
    # ==========================================================
    train_loader = create_dataloader(X_train, Y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(X_val, Y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(X_test, Y_test, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================================
    # 6. Model
    # ==========================================================
    model = STGNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ==========================================================
    # 7. Training
    # ==========================================================
    history = train_model(model, train_loader, val_loader, adj_tensor)

    # ==========================================================
    # 8. Evaluation
    # ==========================================================
    val_metrics = evaluate(model, val_loader, adj_tensor, scaler)
    print_metrics(val_metrics, "Validation")

    test_metrics = evaluate(model, test_loader, adj_tensor, scaler)
    print_metrics(test_metrics, "Test")


if __name__ == "__main__":
    main()