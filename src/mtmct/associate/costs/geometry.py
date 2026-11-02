"""Geometry cost on the ground plane (spec §8.7, overlap profile)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from mtmct.components import AssocContext, CostMatrix
from mtmct.registry import register

Track = tuple[np.ndarray, np.ndarray]  # ts (n,), world xy (n, 2), sorted by ts


def _mean_distance(a: Track, b: Track) -> float:
    """Mean ground distance over the common time span, both tracks interpolated to the union of their timestamps."""
    lo, hi = max(a[0][0], b[0][0]), min(a[0][-1], b[0][-1])
    grid = np.union1d(*(t[(t >= lo) & (t <= hi)] for t, _ in (a, b)))

    def at(tr: Track) -> np.ndarray:
        return np.column_stack([np.interp(grid, tr[0], tr[1][:, 0]), np.interp(grid, tr[0], tr[1][:, 1])])

    return float(np.linalg.norm(at(a) - at(b), axis=1).mean())


@register("cost", "geometry")
class Geometry:
    """Ground-plane distance of co-visible tracklets; transition speed check for consecutive ones."""

    LICENCE = "MIT"

    class Params(BaseModel):
        gate_m: float = Field(1.5, gt=0, description="Co-visible tracklets farther apart than this cannot link")
        max_speed_mps: float = Field(3.0, gt=0, description="Max walking speed between consecutive tracklets")

    def __init__(self, params: Params) -> None:
        self.params = params

    def __call__(self, ctx: AssocContext) -> CostMatrix:
        n = len(ctx.portraits)
        dist, cl = np.full((n, n), np.nan), np.zeros((n, n), bool)
        tr = ctx.tracklets
        world = tr[["world_x", "world_y"]].notna().all(axis=1)
        if not world.any():
            ctx.warn("geometry cost disabled: no camera homography (no world coordinates)")
            return CostMatrix(dist, cl)

        pos = {tid: i for i, tid in enumerate(ctx.portraits["tracklet_id"])}
        tracks: list[Track | None] = [None] * n
        for tid, g in tr[world].sort_values("ts").groupby("tracklet_id"):
            if tid in pos:
                tracks[pos[tid]] = (g["ts"].to_numpy(), g[["world_x", "world_y"]].to_numpy())
        start, end = np.full(n, np.nan), np.full(n, np.nan)
        entry, exit_ = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
        for i, track in enumerate(tracks):
            if track is not None:
                start[i], end[i], entry[i], exit_[i] = track[0][0], track[0][-1], track[1][0], track[1][-1]

        # i ends before j starts: the jump (minus localisation slack gate_m) must be walkable in the gap.
        gap = start[None, :] - end[:, None]
        jump = np.hypot(entry[None, :, 0] - exit_[:, None, 0], entry[None, :, 1] - exit_[:, None, 1])
        too_fast = (gap > 0) & (np.maximum(jump - self.params.gate_m, 0) > self.params.max_speed_mps * gap)
        cl |= too_fast | too_fast.T

        cams = ctx.portraits["camera_id"].to_numpy()
        covisible = (gap <= 0) & (gap.T <= 0) & (cams[:, None] != cams[None, :])  # NaN (no track) compares False
        for i, j in zip(*np.nonzero(np.triu(covisible, 1)), strict=True):
            a, b = tracks[i], tracks[j]
            assert a is not None and b is not None
            d = _mean_distance(a, b)
            dist[i, j] = dist[j, i] = min(d / self.params.gate_m, 1.0)
            cl[i, j] = cl[j, i] = d > self.params.gate_m
        return CostMatrix(dist, cl)
