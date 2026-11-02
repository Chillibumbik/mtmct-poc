"""Helpers shared by the adapters: video probing, GT in Xu's text format, canonical sequence writer."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mtmct.data.base import CameraInfo, Overlap, SequenceInfo, SequenceMeta
from mtmct.io import GroundTruth, write_table

# <track_id> <xmin> <ymin> <xmax> <ymax> <frame> <lost> <occluded> <generated> <label>  (Vatic export)
_XU_COLUMNS = ["person_id", "x1", "y1", "x2", "y2", "frame_id", "lost", "occluded", "generated", "label"]


def probe_video(path: Path) -> tuple[float, int, int, int]:
    """(fps, width, height, n_frames) from the container header."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if fps <= 0 or n <= 0 or w <= 0 or h <= 0:
        raise ValueError(f"video header is incomplete (fps={fps}, frames={n}, size={w}x{h}): {path}")
    return float(fps), w, h, n


def read_xu_gt(path: Path, camera_id: str) -> pd.DataFrame:
    """Xu's per-view annotation → canonical boxes; rows flagged `lost` (outside the view) are dropped."""
    raw = pd.read_csv(path, sep=r"\s+", header=None, names=_XU_COLUMNS)
    raw = raw[(raw["lost"] == 0) & (raw["x2"] > raw["x1"]) & (raw["y2"] > raw["y1"])]
    return pd.DataFrame(
        {
            "camera_id": camera_id,
            "frame_id": raw["frame_id"].astype("int32"),
            "person_id": raw["person_id"].astype("int64"),
            "x": raw["x1"].astype("float32"),
            "y": raw["y1"].astype("float32"),
            "w": (raw["x2"] - raw["x1"]).astype("float32"),
            "h": (raw["y2"] - raw["y1"]).astype("float32"),
        }
    ).reset_index(drop=True)


def link_video(src: Path, dst: Path) -> None:
    """Relative symlink: the original file stays where it was downloaded, no re-encoding or copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))


def copy_video(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def apply_homography(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    p = np.c_[xy, np.ones(len(xy))] @ H.T
    return p[:, :2] / p[:, 2:3]


MAX_RANGE_M = 100.0  # scene extent: farther ground points are numerically meaningless (foot at the horizon)


def add_world(gt: pd.DataFrame, cameras: list[CameraInfo]) -> pd.DataFrame:
    """Fill world_x/world_y with the foot point (bottom centre of the box) through the camera's H.

    Foot points above the horizon (ray misses the ground) or farther than MAX_RANGE_M stay null.
    """
    gt = gt.assign(world_x=np.float32("nan"), world_y=np.float32("nan"))
    for cam in cameras:
        H = cam.H()
        if H is None:
            continue
        rows = gt["camera_id"] == cam.camera_id
        foot = np.c_[gt.loc[rows, "x"] + gt.loc[rows, "w"] / 2, gt.loc[rows, "y"] + gt.loc[rows, "h"]].astype(float)
        w = np.c_[foot, np.ones(len(foot))] @ H[2]
        below_horizon = w * (H @ [cam.width / 2, cam.height, 1.0])[2] > 0  # same side as the image bottom
        world = apply_homography(H, foot)
        world[~below_horizon | (np.abs(world) > MAX_RANGE_M).any(axis=1)] = np.nan
        gt.loc[rows, ["world_x", "world_y"]] = world.astype("float32")
    return gt


def build_sequence(
    data_root: Path,
    dataset: str,
    sequence: str,
    videos: dict[str, Path],
    gt: pd.DataFrame | None,
    *,
    overlap: Overlap,
    splits: dict[str, tuple[float, float]],
    homographies: dict[str, np.ndarray] | None = None,
    frame_offsets: dict[str, int] | None = None,
    place: Callable[[Path, Path], None] = link_video,
    notes: str = "",
) -> SequenceInfo:
    """Write cameras.json / meta.json / gt.parquet and put the videos under `videos/`."""
    root = data_root / dataset / sequence
    homographies = homographies or {}
    frame_offsets = frame_offsets or {}
    cameras = []
    for cam_id, src in videos.items():
        fps, w, h, n = probe_video(src)
        rel = f"videos/{cam_id}{src.suffix}"
        place(src, root / rel)
        H = homographies.get(cam_id)
        cameras.append(
            CameraInfo(
                camera_id=cam_id,
                source=rel,
                fps=fps,
                width=w,
                height=h,
                n_frames=n,
                frame_offset=frame_offsets.get(cam_id, 0),
                homography_img2world=None if H is None else H.tolist(),
            )
        )
    n_persons = None
    if gt is not None:
        by_cam = {c.camera_id: c.n_frames for c in cameras}
        if unknown := set(gt["camera_id"]) - set(by_cam):
            raise ValueError(f"GT refers to unknown cameras: {sorted(unknown)}")
        bad = gt[gt["frame_id"] >= gt["camera_id"].map(by_cam)]
        if len(bad):
            raise ValueError(f"GT has {len(bad)} rows beyond the video length (first: {bad.iloc[0].to_dict()})")
        gt = add_world(gt, cameras)
        write_table(gt, root / "gt.parquet", GroundTruth)
        n_persons = int(gt["person_id"].nunique())
    duration = max(c.n_frames / c.fps for c in cameras)
    seq = SequenceInfo(
        dataset=dataset,
        sequence=sequence,
        root=root,
        cameras=cameras,
        meta=SequenceMeta(overlap=overlap, splits=splits, duration_s=duration, n_persons=n_persons, notes=notes),
    )
    seq.save()
    return seq
