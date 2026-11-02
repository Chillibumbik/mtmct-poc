"""Streaming frame reader with FPS subsampling on the common clock (spec §8.1)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mtmct.data.base import CameraInfo, SequenceInfo

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Frame:
    frame_id: int  # 0-based index in the camera's own video
    ts: float  # seconds on the common clock
    image: np.ndarray  # BGR


def _keep_mask_iter(
    cam: CameraInfo, target_fps: float, t_range: tuple[float, float] | None
) -> Iterator[tuple[int, float, bool]]:
    """Yields (frame_id, ts, keep) forever; caller stops at end of stream."""
    period = 0.0 if target_fps <= 0 or target_fps >= cam.fps else 1.0 / target_fps
    t0, t1 = t_range if t_range else (float("-inf"), float("inf"))
    next_t: float | None = None
    eps = 1e-6
    frame_id = 0
    while True:
        ts = float(cam.ts(frame_id))
        keep = t0 - eps <= ts < t1 - eps
        if keep and period:
            if next_t is None:
                next_t = ts
            keep = ts >= next_t - eps
            if keep:
                next_t += period
                while next_t <= ts + eps:  # catch up if the stream has gaps
                    next_t += period
        yield frame_id, ts, keep
        frame_id += 1


def iter_frames(
    seq: SequenceInfo, cam: CameraInfo, target_fps: float, t_range: tuple[float, float] | None = None
) -> Iterator[Frame]:
    """Decode a camera stream lazily, yielding only frames selected by `target_fps` within `t_range` (seconds)."""
    src = seq.root / cam.source
    plan = _keep_mask_iter(cam, target_fps, t_range)
    t1 = t_range[1] if t_range else float("inf")
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for path, (frame_id, ts, keep) in zip(files, plan, strict=False):
            if ts >= t1:
                return
            if keep:
                yield Frame(frame_id, ts, _imread(path))
        return
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {src}")
    try:
        for frame_id, ts, keep in plan:
            if ts >= t1 or not cap.grab():
                return
            if keep:
                ok, image = cap.retrieve()
                if not ok:
                    return
                yield Frame(frame_id, ts, image)
    finally:
        cap.release()


def _imread(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    return image


def selected_frame_ids(
    cam: CameraInfo, n_frames: int, target_fps: float, t_range: tuple[float, float] | None = None
) -> np.ndarray:
    """Frame ids `iter_frames` would yield for a stream of `n_frames` frames (no decoding). Used by eval/oracle."""
    out = []
    for frame_id, _ts, keep in _keep_mask_iter(cam, target_fps, t_range):
        if frame_id >= n_frames:
            break
        if keep:
            out.append(frame_id)
    return np.asarray(out, dtype=np.int64)
