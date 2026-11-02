"""Anchor-guided association (AIC23 UWIPL-ETRI style, spec §8.7 hypothesis H7)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from mtmct.associate.solvers.agglomerative import Linkage, agglomerative, relabel
from mtmct.registry import register


@register("solver", "anchor")
class Anchor:
    """Tight clustering → clusters spanning ≥ 2 cameras are anchors; other clusters join the nearest anchor."""

    LICENCE = "MIT"

    class Params(BaseModel):
        linkage: Linkage = "average"
        anchor_threshold: float = Field(0.3, ge=0, description="Tight clustering threshold that forms anchors")
        threshold: float = Field(0.45, ge=0, description="Max average distance to join an anchor")

    def __init__(self, params: Params) -> None:
        self.params = params

    def solve(self, dist: np.ndarray, cannot_link: np.ndarray, cameras: np.ndarray | None = None) -> np.ndarray:
        n = len(dist)
        tight = agglomerative(dist, cannot_link, self.params.anchor_threshold, self.params.linkage)
        k = int(tight.max()) + 1 if n else 0
        cams = np.arange(n) if cameras is None else np.asarray(cameras)
        onehot = np.zeros((n, k))
        onehot[np.arange(n), tight] = 1.0
        size = onehot.sum(axis=0)
        n_cams = np.array([len(np.unique(cams[tight == c])) for c in range(k)])
        anchors, others = np.flatnonzero(n_cams >= 2), np.flatnonzero(n_cams < 2)
        if not len(anchors) or not len(others):
            return tight

        # Cluster-level average distance and cannot-link; greedy assignment in order of increasing distance,
        # cannot-links of an anchor grow with every cluster it absorbs.
        avg = (onehot.T @ dist @ onehot) / np.outer(size, size)
        ccl = (onehot.T @ cannot_link.astype(np.float64) @ onehot) > 0
        target = np.arange(k)
        sub = avg[np.ix_(others, anchors)]
        for flat in np.argsort(sub, axis=None, kind="stable"):
            r, c = divmod(int(flat), len(anchors))
            if sub[r, c] > self.params.threshold:
                break
            u, a = others[r], anchors[c]
            if target[u] != u or ccl[u, a]:
                continue
            target[u] = a
            ccl[a] |= ccl[u]
            ccl[:, a] |= ccl[:, u]
        return relabel(target[tight])
