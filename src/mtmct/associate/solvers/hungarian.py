"""One-to-one matching between two cameras (spec §8.7 hypothesis)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment

from mtmct.associate.solvers.agglomerative import relabel
from mtmct.registry import register


@register("solver", "hungarian")
class Hungarian:
    """Hungarian matching of the tracklets of two cameras; pairs above the threshold or cannot-link stay apart."""

    LICENCE = "MIT"

    class Params(BaseModel):
        threshold: float = Field(0.45, ge=0, description="Max distance of a matched pair")

    def __init__(self, params: Params) -> None:
        self.params = params

    def solve(self, dist: np.ndarray, cannot_link: np.ndarray, cameras: np.ndarray | None = None) -> np.ndarray:
        n = len(dist)
        if cameras is None:
            raise ValueError("hungarian solver needs the camera of every tracklet")
        names = np.unique(cameras)
        if len(names) > 2:
            raise ValueError(f"hungarian solver supports at most 2 cameras, got {len(names)}; use agglomerative/anchor")
        labels = np.arange(n)
        if len(names) == 2:
            rows, cols = np.flatnonzero(cameras == names[0]), np.flatnonzero(cameras == names[1])
            cost = dist[np.ix_(rows, cols)]
            bad = cannot_link[np.ix_(rows, cols)] | (cost > self.params.threshold)
            # Matching gain (cost − threshold) < 0 only for allowed pairs, so unmatched is never worse than bad.
            r, c = linear_sum_assignment(np.where(bad, 0.0, cost - self.params.threshold))
            ok = ~bad[r, c]
            labels[cols[c[ok]]] = rows[r[ok]]
        return relabel(labels)
