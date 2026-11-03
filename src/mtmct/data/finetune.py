"""Training data for fine-tuning (ReID crops, detector images) cut from GT of the tune splits only.

Protocol: only the (dataset, sequence, split) triples in `TUNE_SOURCES` may feed training. Test sequences
(garden2, chirla seq_002) and test time ranges (WILDTRACK last 10 %, AIC24 second half) are never read.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mtmct.data.base import CameraInfo, SequenceInfo
from mtmct.video.reader import IMAGE_EXTS

# (dataset, sequence) → the split whose time range may be used for training.
# aic24/scene_041_all holds all 10 cameras of scene_041 with the same clock and splits.
TUNE_SOURCES: dict[tuple[str, str], str] = {
    ("campus", "garden1"): "tune",
    ("wildtrack", "main"): "tune",
    ("aic24", "scene_041_all"): "tune",
    ("chirla", "seq_001"): "tune",
    ("epfl_terrace", "terrace1"): "all",  # dev set, no test numbers are reported on it
}


def tune_gt(seq: SequenceInfo, split: str | None = None) -> pd.DataFrame:
    """GT rows of `seq` whose timestamp lies in the training split (same boundary rule as `iter_frames`)."""
    split = split or TUNE_SOURCES[(seq.dataset, seq.sequence)]
    if split == "test":
        raise ValueError("test splits must never be used for training")
    gt = seq.gt()
    if gt is None:
        raise FileNotFoundError(f"{seq.dataset}/{seq.sequence} has no GT")
    t0, t1 = seq.split_range(split)
    ts = np.empty(len(gt))
    for cam in seq.cameras:
        m = (gt.camera_id == cam.camera_id).to_numpy()
        ts[m] = cam.ts(gt.frame_id.to_numpy()[m])
    eps = 1e-6
    keep = (ts >= t0 - eps) & (ts < t1 - eps)
    return gt[keep].assign(ts=ts[keep]).reset_index(drop=True)


def visible_fraction(gt: pd.DataFrame, width: int, height: int) -> np.ndarray:
    """Per box: share of its area inside the image and not covered by a box whose bottom edge is lower (closer)."""
    x0, y0 = gt.x.to_numpy(), gt.y.to_numpy()
    x1, y1 = x0 + gt.w.to_numpy(), y0 + gt.h.to_numpy()
    area = gt.w.to_numpy() * gt.h.to_numpy()
    inside = (np.clip(x1, 0, width) - np.clip(x0, 0, width)) * (np.clip(y1, 0, height) - np.clip(y0, 0, height))
    out = inside / area
    for idx in gt.groupby(["camera_id", "frame_id"]).indices.values():
        if len(idx) < 2:
            continue
        ix = np.clip(np.minimum(x1[idx, None], x1[None, idx]) - np.maximum(x0[idx, None], x0[None, idx]), 0, None)
        iy = np.clip(np.minimum(y1[idx, None], y1[None, idx]) - np.maximum(y0[idx, None], y0[None, idx]), 0, None)
        inter = ix * iy
        inter[~(y1[None, idx] > y1[idx, None])] = 0  # only occluders standing closer to the camera
        out[idx] -= inter.max(axis=1) / area[idx]
    return np.clip(out, 0, 1)


def sample_reid_boxes(
    gt: pd.DataFrame, per_camera: int = 30, min_h: float = 64, min_visible: float = 0.6, sizes: dict | None = None
) -> pd.DataFrame:
    """≤ `per_camera` boxes per (person, camera), evenly spread in time, skipping small or occluded boxes.

    `sizes`: camera_id → (width, height) for the visibility check.
    """
    vis = np.ones(len(gt))
    for cam, idx in gt.groupby("camera_id").indices.items():
        w, h = (sizes or {}).get(cam, (1e9, 1e9))
        vis[idx] = visible_fraction(gt.iloc[idx], w, h)
    ok = gt[(gt.h >= min_h) & (vis >= min_visible)].sort_values(["person_id", "camera_id", "frame_id"])
    picks = []
    for _, g in ok.groupby(["person_id", "camera_id"]):
        n = min(per_camera, len(g))
        picks.append(g.iloc[np.unique(np.linspace(0, len(g) - 1, n).round().astype(int))])
    return pd.concat(picks, ignore_index=True) if picks else ok.iloc[:0]


def read_frames(
    seq: SequenceInfo, cam: CameraInfo, frame_ids: list[int] | np.ndarray
) -> Iterator[tuple[int, np.ndarray]]:
    """Decode only the requested frames of one camera (ascending order), yielding (frame_id, BGR image)."""
    wanted = sorted({int(f) for f in frame_ids})
    src = seq.root / cam.source
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for f in wanted:
            image = cv2.imread(str(files[f]), cv2.IMREAD_COLOR)
            if image is not None:
                yield f, image
        return
    cap = cv2.VideoCapture(str(src))
    try:
        pos = 0
        for f in wanted:
            while pos < f:
                if not cap.grab():
                    return
                pos += 1
            ok, image = cap.read()
            pos += 1
            if not ok:
                return
            yield f, image
    finally:
        cap.release()


def crop(image: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray | None:
    hh, ww = image.shape[:2]
    x0, y0 = max(round(x), 0), max(round(y), 0)
    x1, y1 = min(round(x + w), ww), min(round(y + h), hh)
    return image[y0:y1, x0:x1] if x1 - x0 >= 8 and y1 - y0 >= 16 else None


def write_yolo_labels(path: Path, boxes: pd.DataFrame, width: int, height: int) -> None:
    """YOLO txt (class cx cy w h, normalised), boxes clipped to the image."""
    x0 = boxes.x.clip(0, width).to_numpy()
    y0 = boxes.y.clip(0, height).to_numpy()
    x1 = (boxes.x + boxes.w).clip(0, width).to_numpy()
    y1 = (boxes.y + boxes.h).clip(0, height).to_numpy()
    keep = (x1 - x0 > 2) & (y1 - y0 > 2)
    rows = [
        f"0 {(a + c) / 2 / width:.6f} {(b + d) / 2 / height:.6f} {(c - a) / width:.6f} {(d - b) / height:.6f}"
        for a, b, c, d in zip(x0[keep], y0[keep], x1[keep], y1[keep], strict=True)
    ]
    path.write_text("\n".join(rows) + ("\n" if rows else ""))
