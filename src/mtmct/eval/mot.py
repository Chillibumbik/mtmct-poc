"""Box matching and MOT metrics (HOTA, CLEAR, Identity) via vendored TrackEval fed with in-memory data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import trackeval
from scipy.optimize import linear_sum_assignment

from mtmct.data.base import SequenceInfo

BOX = ["x", "y", "w", "h"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`(n, 4)`, `(m, 4)` x, y, w, h boxes → `(n, m)` IoU."""
    ax0, ay0, ax1, ay1 = a[:, 0, None], a[:, 1, None], (a[:, 0] + a[:, 2])[:, None], (a[:, 1] + a[:, 3])[:, None]
    bx0, by0, bx1, by1 = b[None, :, 0], b[None, :, 1], (b[:, 0] + b[:, 2])[None], (b[:, 1] + b[:, 3])[None]
    inter = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None) * np.clip(
        np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None
    )
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.asarray(inter / np.maximum(union, np.finfo(float).eps), dtype=float)


@dataclass
class Step:
    """One scored frame of one camera: positional rows of gt/pred and their IoU matrix."""

    camera_id: str
    frame_id: int
    gt: np.ndarray
    pred: np.ndarray
    sim: np.ndarray


def frame_steps(gt: pd.DataFrame, pred: pd.DataFrame, frames: dict[str, np.ndarray]) -> list[Step]:
    """Steps in dict order of cameras, then frame order; rows outside `frames` are ignored."""
    gt_cam, gt_f, gt_box = gt["camera_id"].to_numpy(), gt["frame_id"].to_numpy(), gt[BOX].to_numpy(float)
    pr_cam, pr_f, pr_box = pred["camera_id"].to_numpy(), pred["frame_id"].to_numpy(), pred[BOX].to_numpy(float)
    steps = []
    for cam, cam_frames in frames.items():
        fr = np.unique(np.asarray(cam_frames))
        gi = np.flatnonzero(gt_cam == cam)
        gi = gi[np.argsort(gt_f[gi], kind="stable")]
        pi = np.flatnonzero(pr_cam == cam)
        pi = pi[np.argsort(pr_f[pi], kind="stable")]
        g_lo, g_hi = np.searchsorted(gt_f[gi], fr, "left"), np.searchsorted(gt_f[gi], fr, "right")
        p_lo, p_hi = np.searchsorted(pr_f[pi], fr, "left"), np.searchsorted(pr_f[pi], fr, "right")
        for k, f in enumerate(fr):
            g, p = gi[g_lo[k] : g_hi[k]], pi[p_lo[k] : p_hi[k]]
            steps.append(Step(cam, int(f), g, p, iou_matrix(gt_box[g], pr_box[p])))
    return steps


def step_matches(steps: list[Step], iou: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-step Hungarian on IoU (pairs below `iou` forbidden) → matched positional gt rows, pred rows."""
    gs, ps = [], []
    for s in steps:
        if s.sim.size == 0:
            continue
        score = np.where(s.sim >= iou - np.finfo(float).eps, s.sim, 0.0)
        r, c = linear_sum_assignment(score, maximize=True)
        ok = score[r, c] > 0
        gs.append(s.gt[r[ok]])
        ps.append(s.pred[c[ok]])
    empty = np.zeros(0, dtype=np.int64)
    return (np.concatenate(gs), np.concatenate(ps)) if gs else (empty, empty)


def _ts(df: pd.DataFrame, seq: SequenceInfo) -> np.ndarray:
    fps = df["camera_id"].map({c.camera_id: c.fps for c in seq.cameras}).to_numpy(float)
    offset = df["camera_id"].map({c.camera_id: c.frame_offset for c in seq.cameras}).to_numpy(float)
    return np.asarray((df["frame_id"].to_numpy(float) + offset) / fps)


def matches_table(
    gt: pd.DataFrame, pred: pd.DataFrame, g: np.ndarray, p: np.ndarray, seq: SequenceInfo
) -> pd.DataFrame:
    """Matched positional rows → one row per matched GT box (`ts` on the common clock)."""
    gm = gt.iloc[g]
    out = pd.DataFrame(
        {
            "gt_row": gt.index.to_numpy()[g],
            "pred_row": pred.index.to_numpy()[p],
            "person_id": gm["person_id"].to_numpy(np.int64),
            "global_id": pred["global_id"].to_numpy(np.int64)[p],
            "tracklet_id": pred["tracklet_id"].to_numpy(np.int64)[p],
            "camera_id": gm["camera_id"].to_numpy(),
            "frame_id": gm["frame_id"].to_numpy(np.int64),
        }
    )
    out["ts"] = _ts(out, seq)
    return out


def _remap(ids: np.ndarray, steps_rows: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """Raw ids of each step's rows → contiguous 0..n-1 over the whole sequence (TrackEval requirement)."""
    if not steps_rows:
        return [], 0
    flat = ids[np.concatenate(steps_rows)]
    uniq, inv = np.unique(flat, return_inverse=True)
    return np.split(inv.astype(np.int64), np.cumsum([len(r) for r in steps_rows])[:-1]), len(uniq)


def mot_metrics(steps: list[Step], gt_ids: np.ndarray, pred_ids: np.ndarray, iou: float) -> dict[str, Any]:
    """HOTA (mean over alphas), DetA, AssA, LocA, CLEAR and Identity metrics of `steps` taken as one sequence."""
    g_ids, n_g = _remap(gt_ids, [s.gt for s in steps])
    p_ids, n_p = _remap(pred_ids, [s.pred for s in steps])
    data = {
        "num_timesteps": len(steps),
        "num_gt_ids": n_g,
        "num_tracker_ids": n_p,
        "num_gt_dets": sum(len(s.gt) for s in steps),
        "num_tracker_dets": sum(len(s.pred) for s in steps),
        "gt_ids": g_ids,
        "tracker_ids": p_ids,
        "similarity_scores": [s.sim for s in steps],
    }
    cfg = {"THRESHOLD": iou, "PRINT_CONFIG": False}
    hota = trackeval.metrics.HOTA().eval_sequence(data)
    clear = trackeval.metrics.CLEAR(cfg).eval_sequence(data)
    ident = trackeval.metrics.Identity(cfg).eval_sequence(data)
    out = {k: float(np.mean(hota[k])) for k in ("HOTA", "DetA", "AssA", "LocA")}
    out |= {k: float(ident[k]) for k in ("IDF1", "IDP", "IDR")}
    out |= {k: float(clear[k]) for k in ("MOTA", "MOTP")}
    out |= {"IDSW": int(clear["IDSW"]), "Frag": int(clear["Frag"]), "FP": int(clear["CLR_FP"])}
    out["FN"] = int(clear["CLR_FN"])
    return out
