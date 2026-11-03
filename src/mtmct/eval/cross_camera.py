"""Cross-camera evaluation helper: cheap IDF1 re-scoring for relabelled tracklets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from mtmct.eval.mot import Step


@dataclass
class TrackletCounts:
    """Sufficient statistics of the Identity metric at tracklet level; any tracklet → id relabelling is exact."""

    gt_count: np.ndarray  # (P,) GT boxes per person
    tracklets: np.ndarray  # (K,) sorted unique tracklet ids
    trk_count: np.ndarray  # (K,) predicted boxes per tracklet
    potential: np.ndarray  # (P, K) GT/pred box pairs with IoU ≥ threshold


def tracklet_counts(steps: list[Step], gt_ids: np.ndarray, tracklet_ids: np.ndarray, iou: float) -> TrackletCounts:
    g_rows = np.concatenate([s.gt for s in steps]) if steps else np.zeros(0, np.int64)
    p_rows = np.concatenate([s.pred for s in steps]) if steps else np.zeros(0, np.int64)
    persons, g_code = np.unique(gt_ids[g_rows], return_inverse=True)
    tracklets, p_code = np.unique(tracklet_ids[p_rows], return_inverse=True)
    g_code_of = np.zeros(len(gt_ids), np.int64)
    g_code_of[g_rows] = g_code
    p_code_of = np.zeros(len(tracklet_ids), np.int64)
    p_code_of[p_rows] = p_code
    pg, pp = [], []
    for s in steps:
        r, c = np.nonzero(s.sim >= iou)  # same rule as trackeval Identity (no eps)
        pg.append(g_code_of[s.gt[r]])
        pp.append(p_code_of[s.pred[c]])
    flat = np.concatenate(pg) * len(tracklets) + np.concatenate(pp) if steps else np.zeros(0, np.int64)
    potential = np.bincount(flat, minlength=len(persons) * len(tracklets)).reshape(len(persons), len(tracklets))
    return TrackletCounts(
        np.bincount(g_code, minlength=len(persons)), tracklets, np.bincount(p_code, minlength=len(tracklets)), potential
    )


def relabel(tracklets: np.ndarray, labels: pd.Series) -> np.ndarray:
    """Tracklet ids → global ids from `labels` (tracklet_id → global_id); unlabelled tracklets become singletons."""
    out = labels.reindex(tracklets).to_numpy(dtype=float, copy=True)
    missing = np.isnan(out)
    if missing.any():
        base = np.nanmax(out) + 1 if (~missing).any() else 0.0
        out[missing] = base + np.unique(tracklets[missing], return_inverse=True)[1]
    return out.astype(np.int64)


def relabelled_identity(counts: TrackletCounts, labels: np.ndarray) -> dict[str, float]:
    """IDF1/IDP/IDR when tracklet k gets global id `labels[k]` (same formulas as trackeval Identity)."""
    uniq, inv = np.unique(labels, return_inverse=True)
    pot_t = np.zeros((len(uniq), len(counts.gt_count)))
    np.add.at(pot_t, inv, counts.potential.T)
    idtp = 0.0
    if pot_t.size:
        r, c = linear_sum_assignment(pot_t, maximize=True)
        idtp = float(pot_t[r, c].sum())
    n_gt, n_pred = float(counts.gt_count.sum()), float(counts.trk_count.sum())
    return {
        "IDF1": idtp / max(1.0, 0.5 * (n_gt + n_pred)),
        "IDP": idtp / max(1.0, n_pred),
        "IDR": idtp / max(1.0, n_gt),
    }
