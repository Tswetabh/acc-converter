"""Dynamic Time Warping (DTW) alignment for linguistic content representations.

Aligns variable-length HuBERT feature sequences from parallel utterances
spoken at different tempos by different speakers.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_cosine_distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute cosine distance matrix between two feature sequences.

    Parameters
    ----------
    x: torch.Tensor
        Source features of shape (T_x, D).
    y: torch.Tensor
        Target features of shape (T_y, D).

    Returns
    -------
    torch.Tensor
        Pairwise distance matrix of shape (T_x, T_y) with values in [0, 2].
    """
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    sim = torch.mm(x_norm, y_norm.transpose(0, 1))
    dist = 1.0 - sim
    return torch.clamp(dist, min=0.0, max=2.0)


def dtw_alignment_dp(dist_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute monotonic DTW warping path using dynamic programming.

    Parameters
    ----------
    dist_matrix: np.ndarray
        Distance matrix of shape (T_x, T_y).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Alignment indices (path_x, path_y) of equal length.
    """
    N, M = dist_matrix.shape
    cost = np.full((N + 1, M + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0

    # Accumulated cost
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            d = dist_matrix[i - 1, j - 1]
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Backtracking from (N, M) to (0, 0)
    i, j = N, M
    path_x = []
    path_y = []

    while i > 0 and j > 0:
        path_x.append(i - 1)
        path_y.append(j - 1)
        # Check predecessors
        steps = [
            (cost[i - 1, j - 1], i - 1, j - 1),  # diagonal
            (cost[i - 1, j], i - 1, j),          # left
            (cost[i, j - 1], i, j - 1),          # up
        ]
        steps.sort(key=lambda s: s[0])
        _, i, j = steps[0]

    path_x.reverse()
    path_y.reverse()
    return np.array(path_x, dtype=np.int64), np.array(path_y, dtype=np.int64)


def align_hubert_with_path(
    h_src: torch.Tensor,
    h_tgt: torch.Tensor,
    use_fastdtw: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Align two HuBERT continuous representation sequences.

    Parameters
    ----------
    h_src: torch.Tensor
        Source HuBERT features of shape (T_src, 768).
    h_tgt: torch.Tensor
        Target HuBERT features of shape (T_tgt, 768).
    use_fastdtw: bool
        If True and fastdtw is available, use fastdtw approximation.
        Otherwise falls back to exact dynamic programming.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        path_src, path_tgt (equal length index arrays), average alignment cost.
    """
    dist_matrix = compute_cosine_distance_matrix(h_src, h_tgt).detach().cpu().numpy()

    if use_fastdtw:
        try:
            from fastdtw import fastdtw
            from scipy.spatial.distance import cosine

            # Normalize for fastdtw
            src_np = F.normalize(h_src, p=2, dim=-1).detach().cpu().numpy()
            tgt_np = F.normalize(h_tgt, p=2, dim=-1).detach().cpu().numpy()

            distance, path = fastdtw(src_np, tgt_np, dist=cosine)
            path_x = np.array([p[0] for p in path], dtype=np.int64)
            path_y = np.array([p[1] for p in path], dtype=np.int64)
            avg_cost = float(distance) / max(1, len(path))
        except Exception:
            path_x, path_y = dtw_alignment_dp(dist_matrix)
            avg_cost = float(dist_matrix[path_x, path_y].mean())
    else:
        path_x, path_y = dtw_alignment_dp(dist_matrix)
        avg_cost = float(dist_matrix[path_x, path_y].mean())

    return path_x, path_y, avg_cost


def align_hubert_features(
    h_src: torch.Tensor,
    h_tgt: torch.Tensor,
    use_fastdtw: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Align two sequences; returns frame-repeated aligned_src, aligned_tgt, avg cost."""
    path_x, path_y, avg_cost = align_hubert_with_path(h_src, h_tgt, use_fastdtw)
    device = h_src.device
    return h_src[path_x].to(device), h_tgt[path_y].to(device), avg_cost


def source_length_target(
    h_tgt: torch.Tensor, path_x: np.ndarray, path_y: np.ndarray, t_src: int
) -> torch.Tensor:
    """Target features at the *source* frame rate.

    For each source frame i, average the target frames aligned to it by the
    DTW path. Result has shape (t_src, D) so no frames are repeated.
    """
    d = h_tgt.shape[1]
    out = torch.zeros(t_src, d, dtype=h_tgt.dtype, device=h_tgt.device)
    counts = torch.zeros(t_src, dtype=h_tgt.dtype, device=h_tgt.device)
    px = torch.from_numpy(path_x).to(h_tgt.device)
    py = torch.from_numpy(path_y).to(h_tgt.device)
    out.index_add_(0, px, h_tgt[py])
    counts.index_add_(0, px, torch.ones_like(px, dtype=h_tgt.dtype))
    return out / counts.clamp(min=1).unsqueeze(1)
