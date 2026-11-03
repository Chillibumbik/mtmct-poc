"""Transition-time cost between cameras from the camera topology (spec §8.7, non-overlap profile)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from mtmct.components import AssocContext, CostMatrix
from mtmct.registry import register


@register("cost", "time")
class TransitionTime:
    """Δt from leaving camera i to entering camera j checked against the topology window [lo, hi] of that pair."""

    LICENCE = "MIT"

    class Params(BaseModel):
        hard: bool = Field(True, description="Outside the window: cannot-link (true) or distance 1 (false)")
        unseen_penalty: float = Field(0.5, ge=0, le=1, description="Distance for camera pairs never seen in topology")
        overlap: Literal["ignore", "cannot_link"] = Field(
            "ignore",
            description="Pairs visible at the same time in two cameras: no information (ignore) or cannot-link "
            "(only for layouts where cameras never overlap)",
        )

    def __init__(self, params: Params) -> None:
        self.params = params

    def __call__(self, ctx: AssocContext) -> CostMatrix:
        n = len(ctx.portraits)
        if ctx.topology is None or "pairs" not in ctx.topology:
            ctx.warn("time cost disabled: no camera topology (topology.json)")
            return CostMatrix(np.full((n, n), np.nan), np.zeros((n, n), bool))

        names, cam = np.unique(ctx.portraits["camera_id"].to_numpy(), return_inverse=True)
        lo, hi = np.full((len(names),) * 2, np.nan), np.full((len(names),) * 2, np.nan)
        for a, src in enumerate(names):
            for b, dst in enumerate(names):
                if (window := ctx.topology["pairs"].get(f"{src}->{dst}")) is not None:
                    lo[a, b], hi[a, b] = window
        t0 = ctx.portraits["t_start"].to_numpy(dtype=np.float64)
        t1 = ctx.portraits["t_end"].to_numpy(dtype=np.float64)

        # Order each pair by start time (ties by index): A = earlier, Δt = B.t_start − A.t_end (< 0 when co-visible).
        idx = np.arange(n)
        first = (t0[:, None] < t0[None, :]) | ((t0[:, None] == t0[None, :]) & (idx[:, None] < idx[None, :]))
        gap = t0[None, :] - t1[:, None]
        dt = np.where(first, gap, gap.T)
        src_cam = np.where(first, cam[:, None], cam[None, :])
        dst_cam = np.where(first, cam[None, :], cam[:, None])
        w_lo, w_hi = lo[src_cam, dst_cam], hi[src_cam, dst_cam]
        seen, inside = ~np.isnan(w_lo), (dt >= w_lo) & (dt <= w_hi)
        other_cam = cam[:, None] != cam[None, :]

        dist = np.where(seen, np.where(inside, 0.0, 1.0), self.params.unseen_penalty)
        cl = seen & ~inside & self.params.hard
        covisible = (dt <= 0) & ~inside
        if self.params.overlap == "cannot_link":
            dist, cl = np.where(covisible, 1.0, dist), cl | covisible
        else:  # simultaneous views say nothing about the transition time
            dist, cl = np.where(covisible, np.nan, dist), cl & ~covisible
        dist = np.where(other_cam, dist, np.nan)
        return CostMatrix(dist, cl & other_cam)
