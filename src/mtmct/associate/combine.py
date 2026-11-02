"""Fusion of cost matrices into one distance + cannot-link matrix (spec §8.7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mtmct.components import CostMatrix


def same_camera_overlap(portraits: pd.DataFrame) -> np.ndarray:
    """`(T, T)` bool: two tracklets of the same camera that overlap in time (one person cannot be both)."""
    cam = portraits["camera_id"].to_numpy()
    t0, t1 = portraits["t_start"].to_numpy(), portraits["t_end"].to_numpy()
    m = (cam[:, None] == cam[None, :]) & (t0[:, None] <= t1[None, :]) & (t0[None, :] <= t1[:, None])
    np.fill_diagonal(m, False)
    return m


def combine(parts: list[tuple[CostMatrix, float]], same_camera_overlap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean over the costs that have information (non-NaN); none → 1.0. Cannot-links are OR-ed."""
    n = same_camera_overlap.shape[0]
    num, den = np.zeros((n, n)), np.zeros((n, n))
    cannot = same_camera_overlap.astype(bool)
    for cm, w in parts:
        known = ~np.isnan(cm.dist)
        num += w * np.where(known, cm.dist, 0.0)
        den += w * known
        cannot = cannot | cm.cannot_link
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.where(den > 0, num / den, 1.0)
    np.fill_diagonal(dist, 0.0)
    np.fill_diagonal(cannot, False)
    return dist, cannot
