"""
main.py — Quick single-model run for development/debugging.

For full experiments, use:
    python scripts/run_baseline.py --temporal all
    python scripts/run_stgnn.py --temporal all
    python scripts/compare.py --runs ... --tag ...
"""
import pandas as pd
import torch

from src.config import DEVICE
from src.evaluate import evaluate, print_metrics
from src.models.stgnn import STGNN
from src.train import train_model
from src.utils.experiment import prepare_data, save_run_results


def main():
    train_loader, val_loader, test_loader, adj_tensor, scaler = prepare_data()

    model = STGNN(temporal_type="lstm")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    history = train_model(model, train_loader, val_loader, adj_tensor)

    val_metrics = evaluate(model, val_loader, adj_tensor, scaler)
    print_metrics(val_metrics, "Validation")

    test_metrics = evaluate(model, test_loader, adj_tensor, scaler)
    print_metrics(test_metrics, "Test")


if __name__ == "__main__":
    main()