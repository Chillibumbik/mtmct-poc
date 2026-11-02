"""Image → ground-plane projection of foot points (spec §8.6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mtmct.data.base import SequenceInfo


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """`(n, 2)` points → `(n, 2)` projected points; points mapped to infinity become NaN."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack([p, np.ones(len(p))]) @ np.asarray(H, dtype=np.float64).T
    w = hom[:, 2]
    w = np.where(np.abs(w) < 1e-12, np.nan, w)
    return hom[:, :2] / w[:, None]


def to_world(tracklets: pd.DataFrame, seq: SequenceInfo, smooth_window: int = 5) -> pd.DataFrame:
    """Fill world_x/world_y from foot points (NaN for cameras without a homography), smoothed per tracklet in time."""
    out = tracklets.copy()
    world = np.full((len(out), 2), np.nan)
    cams = out["camera_id"].to_numpy()
    for cam in seq.cameras:
        H = cam.H()
        m = cams == cam.camera_id
        if H is not None and m.any():
            world[m] = apply_h(H, out.loc[m, ["foot_x", "foot_y"]].to_numpy())
    if smooth_window > 1 and len(out):
        order = np.lexsort((out["ts"].to_numpy(), out["tracklet_id"].to_numpy()))
        sorted_world = pd.DataFrame(world[order], columns=["x", "y"])
        sorted_world["tid"] = out["tracklet_id"].to_numpy()[order]
        rolled = sorted_world.groupby("tid")[["x", "y"]].rolling(smooth_window, center=True, min_periods=1).mean()
        world[order] = rolled.to_numpy()  # groupby(sort=True) keeps the tid-then-ts order of `order`
    out["world_x"], out["world_y"] = world[:, 0], world[:, 1]
    return out
