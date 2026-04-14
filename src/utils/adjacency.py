"""
Adjacency matrix construction for spatiotemporal GNN.

Pipeline:
  1. Compute region centroids from lat/lon of crime records
  2. Build KNN-based adjacency
  3. Symmetrize → add self-loops → GCN-normalize
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.config import KNN_K, NUM_REGIONS, REGION_IDS


def compute_region_centroids(
    df: pd.DataFrame,
    region_ids: List[int] = REGION_IDS,
) -> np.ndarray:
    """
    Compute the centroid (mean lat, mean lon) per Community Area.

    Parameters
    ----------
    df : pd.DataFrame with columns [Community Area, Latitude, Longitude]

    Returns
    -------
    centroids : np.ndarray, shape (num_regions, 2)  — ordered by region_ids
    """
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    df["region_id"] = df["Community Area"].astype(int)

    grouped = df.groupby("region_id")[["Latitude", "Longitude"]].mean()

    centroids = np.zeros((len(region_ids), 2))
    for i, rid in enumerate(region_ids):
        if rid in grouped.index:
            centroids[i] = grouped.loc[rid].values
        else:
            # Fallback: use overall mean (should not happen for Chicago 1–77)
            centroids[i] = grouped.mean().values
            print(f"[warn] region {rid} has no lat/lon data, using global mean")

    print(f"[centroids] computed for {len(region_ids)} regions")
    return centroids


def build_knn_adjacency(centroids: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """
    Build adjacency matrix via K-nearest-neighbors on centroid distances.

    Returns
    -------
    A : np.ndarray, shape (N, N), binary, asymmetric
    """
    N = centroids.shape[0]
    dist_matrix = cdist(centroids, centroids, metric="euclidean")

    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        # Exclude self (distance 0) by sorting and taking indices 1..k+1
        neighbors = np.argsort(dist_matrix[i])[1 : k + 1]
        A[i, neighbors] = 1.0

    return A


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Make adjacency undirected: A_sym = max(A, A^T)."""
    return np.maximum(A, A.T)


def add_self_loops(A: np.ndarray) -> np.ndarray:
    """Add identity to adjacency (self-connections)."""
    return A + np.eye(A.shape[0], dtype=A.dtype)


def gcn_normalize(A: np.ndarray) -> np.ndarray:
    """
    GCN normalization: D^{-1/2} A D^{-1/2}.
    Assumes A already contains self-loops.
    """
    D = np.sum(A, axis=1)  # degree vector
    D_inv_sqrt = np.where(D > 0, np.power(D, -0.5), 0.0)
    D_mat = np.diag(D_inv_sqrt)
    return D_mat @ A @ D_mat


def build_adjacency(
    df: pd.DataFrame,
    k: int = KNN_K,
    region_ids: List[int] = REGION_IDS,
) -> np.ndarray:
    """
    Full adjacency construction pipeline.

    Parameters
    ----------
    df : pd.DataFrame — raw training data (with Latitude, Longitude, Community Area)
    k  : int — number of nearest neighbors

    Returns
    -------
    A_norm : np.ndarray, shape (num_regions, num_regions), GCN-normalized
    """
    centroids = compute_region_centroids(df, region_ids)
    A = build_knn_adjacency(centroids, k)
    A = symmetrize(A)
    A = add_self_loops(A)
    A = gcn_normalize(A)
    print(f"[adjacency] shape={A.shape}, non-zero={np.count_nonzero(A)}")
    return A


def save_adjacency(A: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, A)
    print(f"[adjacency] saved → {path}")


def load_adjacency(path: str) -> np.ndarray:
    return np.load(path)