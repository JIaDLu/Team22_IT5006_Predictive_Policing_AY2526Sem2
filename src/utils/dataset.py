"""
PyTorch Dataset and DataLoader construction for spatiotemporal crime data.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.config import BATCH_SIZE


class CrimeDataset(Dataset):
    """
    Spatiotemporal crime prediction dataset.

    Parameters
    ----------
    X : np.ndarray, shape (num_samples, window_size, num_regions, num_features)
    Y : np.ndarray, shape (num_samples, num_regions)
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def create_dataloader(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
) -> DataLoader:
    """Wrap numpy arrays into a DataLoader."""
    dataset = CrimeDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def print_data_summary(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_val: np.ndarray, Y_val: np.ndarray,
    X_test: np.ndarray, Y_test: np.ndarray,
    adj: np.ndarray,
) -> None:
    """Print shapes of all data tensors for verification."""
    print("\n" + "=" * 55)
    print("  Data Summary")
    print("=" * 55)
    print(f"  Train  X: {X_train.shape}  Y: {Y_train.shape}")
    print(f"  Val    X: {X_val.shape}    Y: {Y_val.shape}")
    print(f"  Test   X: {X_test.shape}   Y: {Y_test.shape}")
    print(f"  Adjacency: {adj.shape}")
    print("=" * 55 + "\n")