"""`evaluate_run`: the full metric set of one run in the spec §9.7 `metrics.json` shape (JSON-safe)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from mtmct.data.base import SequenceInfo
from mtmct.eval.cross_camera import relabel, relabelled_identity, tracklet_counts
from mtmct.eval.identity import cluster_scores, identity_metrics
from mtmct.eval.mot import frame_steps, matches_table, mot_metrics, step_matches
from mtmct.eval.reid_sanity import reid_sanity, tracklet_labels


def json_safe(v: Any) -> Any:
    """numpy scalars/arrays → Python; NaN/inf → None."""
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, list | tuple | np.ndarray):
        return [json_safe(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def evaluate_run(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    seq: SequenceInfo,
    eval_frames: dict[str, np.ndarray],
    iou: float = 0.5,
    portraits: pd.DataFrame | None = None,
    portrait_emb: np.ndarray | None = None,
    threshold_labels: dict[float, pd.Series] | None = None,
) -> dict[str, Any]:
    """Score GT (GroundTruth) vs prediction (GlobalTracks) on `eval_frames[camera_id]` only (both sides restricted).

    Cross-camera metrics treat the cameras, in sequence order, as one concatenated sequence (`concat_cameras`).
    `threshold_labels` = {solver threshold: Series tracklet_id → global_id} gives the P/R/IDF1 curve.
    """
    frames = {c.camera_id: np.asarray(eval_frames[c.camera_id]) for c in seq.cameras if c.camera_id in eval_frames}
    steps = frame_steps(gt, pred, frames)
    gt_ids, pred_ids = gt["person_id"].to_numpy(np.int64), pred["global_id"].to_numpy(np.int64)

    per_camera = {cam: mot_metrics([s for s in steps if s.camera_id == cam], gt_ids, pred_ids, iou) for cam in frames}
    mean = {k: float(np.mean([m[k] for m in per_camera.values()])) for k in next(iter(per_camera.values()), {})}

    matches = matches_table(gt, pred, *step_matches(steps, iou), seq)
    g_rows = np.concatenate([s.gt for s in steps]) if steps else np.zeros(0, np.int64)
    p_rows = np.concatenate([s.pred for s in steps]) if steps else np.zeros(0, np.int64)
    identity = identity_metrics(matches, gt.iloc[g_rows], pred.iloc[p_rows])

    reid: dict[str, Any] = {"rank1": None, "rank5": None, "mAP": None, "n_queries": 0}
    if portraits is not None and portrait_emb is not None:
        labels = tracklet_labels(matches)
        keep = portraits["tracklet_id"].isin(labels.index).to_numpy()
        reid = reid_sanity(
            portrait_emb[keep],
            labels.loc[portraits["tracklet_id"][keep]].to_numpy(),
            portraits["camera_id"][keep].to_numpy(),
        )

    curve = []
    if threshold_labels:
        counts = tracklet_counts(steps, gt_ids, pred["tracklet_id"].to_numpy(np.int64), iou)
        match_k = np.searchsorted(counts.tracklets, matches["tracklet_id"].to_numpy(np.int64))
        for thr in sorted(threshold_labels):
            lab = relabel(counts.tracklets, threshold_labels[thr])
            pw = cluster_scores(matches["person_id"].to_numpy(), lab[match_k])
            curve.append(
                {"threshold": thr}
                | {k: pw[k] for k in ("pairwise_p", "pairwise_r", "pairwise_f1")}
                | relabelled_identity(counts, lab)
                | {"n_pred_ids": len(np.unique(lab))}
            )

    return json_safe(
        {
            "per_camera": per_camera,
            "per_camera_mean": mean,
            "cross_camera": mot_metrics(steps, gt_ids, pred_ids, iou),
            "identity": identity,
            "reid": reid,
            "threshold_curve": curve,
        }
    )
