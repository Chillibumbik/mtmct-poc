"""Image → ground-plane projection of foot points (spec §8.6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mtmct.data.base import CameraInfo, SequenceInfo

MAX_RANGE_M = 100.0  # scene extent: farther ground points are numerically meaningless (foot at the horizon)


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """`(n, 2)` points → `(n, 2)` projected points; points mapped to infinity become NaN."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack([p, np.ones(len(p))]) @ np.asarray(H, dtype=np.float64).T
    w = hom[:, 2]
    w = np.where(np.abs(w) < 1e-12, np.nan, w)
    return hom[:, :2] / w[:, None]


def to_ground(cam: CameraInfo, foot: np.ndarray) -> np.ndarray:
    """Image foot points → ground plane (m) through the camera's homography (which must exist).

    Points above the horizon (the ray misses the ground) or farther than MAX_RANGE_M become NaN.
    """
    H = cam.H()
    assert H is not None, cam.camera_id
    p = np.asarray(foot, dtype=np.float64).reshape(-1, 2)
    world = apply_h(H, p)
    w = np.column_stack([p, np.ones(len(p))]) @ H[2]
    below_horizon = w * (H @ [cam.width / 2, cam.height, 1.0])[2] > 0  # same side as the image bottom
    world[~below_horizon | (np.abs(world) > MAX_RANGE_M).any(axis=1)] = np.nan
    return world


def to_world(tracklets: pd.DataFrame, seq: SequenceInfo, smooth_window: int = 5) -> pd.DataFrame:
    """Fill world_x/world_y from foot points (NaN for cameras without a homography), smoothed per tracklet in time."""
    out = tracklets.copy()
    world = np.full((len(out), 2), np.nan)
    cams = out["camera_id"].to_numpy()
    for cam in seq.cameras:
        m = cams == cam.camera_id
        if cam.H() is not None and m.any():
            world[m] = to_ground(cam, out.loc[m, ["foot_x", "foot_y"]].to_numpy())
    if smooth_window > 1 and len(out):
        order = np.lexsort((out["ts"].to_numpy(), out["tracklet_id"].to_numpy()))
        sorted_world = pd.DataFrame(world[order], columns=["x", "y"])
        sorted_world["tid"] = out["tracklet_id"].to_numpy()[order]
        rolled = sorted_world.groupby("tid")[["x", "y"]].rolling(smooth_window, center=True, min_periods=1).mean()
        world[order] = rolled.to_numpy()  # groupby(sort=True) keeps the tid-then-ts order of `order`
    out["world_x"], out["world_y"] = world[:, 0], world[:, 1]
    return out
