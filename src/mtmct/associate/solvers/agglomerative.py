"""Constrained agglomerative clustering of tracklets (spec §8.7, baseline solver)."""

from __future__ import annotations

import heapq
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from mtmct.registry import register

Linkage = Literal["average", "single", "complete"]


def relabel(labels: np.ndarray) -> np.ndarray:
    """Cluster labels → 0..k-1 in order of first appearance."""
    return pd.factorize(np.asarray(labels))[0].astype(np.int64)


def agglomerative(dist: np.ndarray, cannot_link: np.ndarray, threshold: float, linkage: Linkage) -> np.ndarray:
    """Merge the closest pair of clusters while its linkage distance ≤ threshold and no cross pair is cannot-link.

    Lance–Williams updates on a dense matrix + a heap of per-cluster nearest neighbours (lazy deletion).
    A merge never lowers a third cluster's distance to the merged pair (true for these linkages), so only clusters
    whose nearest neighbour took part in the merge need a refresh: O(T²) memory, ~O(T²) time in practice.
    """
    n = len(dist)
    d = np.array(dist, dtype=np.float64)
    cl = np.array(cannot_link, dtype=bool)
    active, size, label = np.ones(n, bool), np.ones(n, np.int64), np.arange(n)
    nn, nnd = np.zeros(n, np.int64), np.full(n, np.inf)
    heap: list[tuple[float, int]] = []

    def refresh(a: int) -> None:
        row = np.where(cl[a] | ~active, np.inf, d[a])
        row[a] = np.inf
        nn[a] = int(np.argmin(row))
        nnd[a] = row[nn[a]]
        heapq.heappush(heap, (float(nnd[a]), a))

    for a in range(n):
        refresh(a)
    while heap:
        da, a = heapq.heappop(heap)
        if not active[a] or da != nnd[a]:
            continue  # stale entry
        if da > threshold:
            break
        keep, drop = sorted((a, int(nn[a])))
        if linkage == "average":
            d[keep] = (size[keep] * d[keep] + size[drop] * d[drop]) / (size[keep] + size[drop])
        else:
            d[keep] = (np.minimum if linkage == "single" else np.maximum)(d[keep], d[drop])
        d[:, keep] = d[keep]
        cl[keep] |= cl[drop]
        cl[:, keep] = cl[keep]
        active[drop] = False
        size[keep] += size[drop]
        label[label == drop] = keep
        for c in np.flatnonzero(active & ((nn == keep) | (nn == drop))):
            refresh(int(c))
    return relabel(label)


@register("solver", "agglomerative")
class Agglomerative:
    """Agglomerative clustering; a merge is refused if any pair across the two clusters is cannot-link."""

    LICENCE = "MIT"

    class Params(BaseModel):
        linkage: Linkage = "average"
        threshold: float = Field(0.45, ge=0, description="Stop merging above this linkage distance")

    def __init__(self, params: Params) -> None:
        self.params = params

    def solve(self, dist: np.ndarray, cannot_link: np.ndarray, cameras: np.ndarray | None = None) -> np.ndarray:
        return agglomerative(dist, cannot_link, self.params.threshold, self.params.linkage)
