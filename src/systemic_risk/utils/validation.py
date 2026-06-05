from __future__ import annotations

import numpy as np


def ensure_binary_samples(samples: np.ndarray, n_features: int | None = None) -> np.ndarray:
    array = np.asarray(samples, dtype=int)
    if array.ndim != 2:
        raise ValueError("samples must be a 2D array")
    if n_features is not None and array.shape[1] != n_features:
        raise ValueError(f"samples must have {n_features} columns")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("samples must contain only 0/1 values")
    return array


def nearest_psd_correlation(matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix to a usable positive semidefinite correlation matrix."""
    corr = np.asarray(matrix, dtype=float)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, epsilon)
    projected = (eigvecs * eigvals) @ eigvecs.T
    scale = np.sqrt(np.diag(projected))
    projected = projected / np.outer(scale, scale)
    np.fill_diagonal(projected, 1.0)
    return np.clip(projected, -0.999, 0.999)
