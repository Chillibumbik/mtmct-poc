"""Tracklet portraits: top-k quality embeddings → mean + gallery (spec §8.6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mtmct.io import Portraits


def _ends(g: pd.DataFrame) -> tuple[float, float, float, float]:
    """Entry/exit point: world coordinates when the tracklet has any, else image foot points."""
    world = g[["world_x", "world_y"]].dropna().to_numpy() if "world_x" in g else np.zeros((0, 2))
    pts = world if len(world) else g[["foot_x", "foot_y"]].to_numpy()
    return float(pts[0, 0]), float(pts[0, 1]), float(pts[-1, 0]), float(pts[-1, 1])


def build_portraits(
    tracklets: pd.DataFrame, emb: np.ndarray, top_k: int = 10
) -> tuple[pd.DataFrame, np.ndarray, list[np.ndarray]]:
    """One row per tracklet (sorted by id): portrait = L2-normalised mean of the top-k embedded boxes by quality.

    Tracklets without embedded boxes get a zero portrait and an empty gallery (costs read that as "no information").
    """
    dim = emb.shape[1]
    rows, means, galleries = [], [], []
    for tid, g in tracklets.sort_values("ts", kind="stable").groupby("tracklet_id", sort=True):
        embedded = g[g["emb_idx"] >= 0].sort_values("quality", ascending=False, kind="stable")
        idx = embedded["emb_idx"].to_numpy()[:top_k].astype(np.int64)
        gallery = emb[idx].astype(np.float32)
        mean = gallery.sum(axis=0) if len(idx) else np.zeros(dim, np.float32)
        norm = np.linalg.norm(mean)
        means.append(mean / norm if norm > 0 else mean)
        galleries.append(gallery)
        ex, ey, xx, xy = _ends(g)
        rows.append(
            {
                "tracklet_id": tid, "camera_id": g["camera_id"].iloc[0],
                "t_start": g["ts"].iloc[0], "t_end": g["ts"].iloc[-1], "n_boxes": len(g),
                "mean_quality": g["quality"].mean(), "gallery_idx": idx.tolist(),
                "entry_x": ex, "entry_y": ey, "exit_x": xx, "exit_y": xy,
            }
        )  # fmt: skip
    df = pd.DataFrame(rows, columns=list(Portraits.to_schema().columns))
    return Portraits.validate(df), np.array(means, np.float32).reshape(-1, dim), galleries
