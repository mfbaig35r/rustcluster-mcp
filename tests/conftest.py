"""Shared fixtures for rustcluster-mcp tests."""

import numpy as np
import pytest


@pytest.fixture
def tmp_data_2d(tmp_path):
    """Small synthetic 2D array (100, 50) saved as .npy."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 50))
    path = tmp_path / "data_2d.npy"
    np.save(path, X)
    return str(path)


@pytest.fixture
def tmp_data_l2_normalized(tmp_path):
    """L2-normalized embeddings (200, 768) saved as .npy."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 768))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    path = tmp_path / "data_l2.npy"
    np.save(path, X)
    return str(path)


@pytest.fixture
def tmp_data_1d(tmp_path):
    """1D array for validation error tests."""
    X = np.arange(100, dtype=np.float64)
    path = tmp_path / "data_1d.npy"
    np.save(path, X)
    return str(path)


@pytest.fixture
def tmp_data_with_nan(tmp_path):
    """2D array with NaN values."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 50))
    X[0, 0] = np.nan
    X[10, 5] = np.nan
    path = tmp_path / "data_nan.npy"
    np.save(path, X)
    return str(path)


@pytest.fixture
def tmp_data_npz(tmp_path):
    """Data saved as .npz for format test."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 50))
    path = tmp_path / "data.npz"
    np.savez(path, embeddings=X)
    return str(path)


@pytest.fixture
def tmp_labels(tmp_path):
    """Label array matching tmp_data_2d (100 samples, 3 clusters)."""
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 3, size=100).astype(np.int64)
    path = tmp_path / "labels.npy"
    np.save(path, labels)
    return str(path)
