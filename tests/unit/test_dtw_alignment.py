"""Unit tests for Dynamic Time Warping (DTW) feature alignment."""
import numpy as np
import pytest
import torch

from accent_converter.models.backends.accent.dtw import (
    align_hubert_features,
    compute_cosine_distance_matrix,
    dtw_alignment_dp,
)


def test_cosine_distance_matrix_properties():
    # Identity test: distance between identical vectors must be 0
    t_src = 20
    d = 768
    x = torch.randn(t_src, d)
    dist = compute_cosine_distance_matrix(x, x)

    assert dist.shape == (t_src, t_src)
    # Diagonal should be near 0
    diag = torch.diagonal(dist)
    assert torch.all(diag < 1e-4)
    # Values should be non-negative
    assert torch.all(dist >= 0.0)


def test_dtw_alignment_dp_monotonicity():
    # Synthetic distance matrix
    dist = np.array([
        [0.1, 0.9, 0.9],
        [0.8, 0.2, 0.9],
        [0.9, 0.9, 0.1],
    ], dtype=np.float32)

    path_x, path_y = dtw_alignment_dp(dist)

    # Must start at (0, 0) and end at (2, 2)
    assert path_x[0] == 0 and path_y[0] == 0
    assert path_x[-1] == 2 and path_y[-1] == 2

    # Steps must be monotonic: delta >= 0
    diff_x = np.diff(path_x)
    diff_y = np.diff(path_y)
    assert np.all(diff_x >= 0)
    assert np.all(diff_y >= 0)
    # No step where both remain stationary
    assert np.all((diff_x + diff_y) > 0)


def test_align_hubert_features_shapes():
    t_src = 50
    t_tgt = 65
    d = 768

    h_src = torch.randn(t_src, d)
    h_tgt = torch.randn(t_tgt, d)

    aligned_src, aligned_tgt, cost = align_hubert_features(h_src, h_tgt, use_fastdtw=False)

    assert aligned_src.shape[0] == aligned_tgt.shape[0]
    assert aligned_src.shape[1] == d
    assert aligned_tgt.shape[1] == d
    assert aligned_src.shape[0] >= max(t_src, t_tgt)
    assert cost >= 0.0


def test_align_hubert_features_fastdtw_fallback():
    # Test that use_fastdtw=True works cleanly
    t_src = 30
    t_tgt = 40
    d = 768

    h_src = torch.randn(t_src, d)
    h_tgt = torch.randn(t_tgt, d)

    aligned_src, aligned_tgt, cost = align_hubert_features(h_src, h_tgt, use_fastdtw=True)

    assert aligned_src.shape[0] == aligned_tgt.shape[0]
    assert aligned_src.shape[1] == d
    assert aligned_tgt.shape[1] == d


def test_source_length_target_shape_and_mean():
    from accent_converter.models.backends.accent.dtw import align_hubert_with_path, source_length_target
    h_src = torch.randn(30, 16)
    h_tgt = torch.randn(45, 16)
    px, py, _ = align_hubert_with_path(h_src, h_tgt, use_fastdtw=False)
    out = source_length_target(h_tgt, px, py, 30)
    assert out.shape == (30, 16)
    i = 0
    expected = h_tgt[py[px == i]].mean(0)
    assert torch.allclose(out[i], expected, atol=1e-5)

