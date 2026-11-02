"""k-reciprocal re-ranking (Zhong et al., CVPR 2017) of a square tracklet distance matrix (spec §8.7, H6)."""

from __future__ import annotations

import numpy as np


def k_reciprocal_rerank(dist: np.ndarray, k1: int = 20, k2: int = 6, lambda_: float = 0.3) -> np.ndarray:
    """`(1 − λ)·Jaccard(k-reciprocal sets) + λ·dist`; symmetric, diagonal 0, NaN entries (no information) kept NaN."""
    d = np.array(dist, dtype=np.float64)
    n = len(d)
    missing = np.isnan(d)
    if n < 2:
        return d
    d[missing] = 1.0
    np.fill_diagonal(d, 0.0)
    k1, k2 = min(k1, n - 1), max(1, min(k2, n))
    rank = np.argsort(d, axis=1, kind="stable")

    def reciprocal(i: int, k: int) -> np.ndarray:
        fwd = rank[i, : k + 1]
        return fwd[(rank[fwd, : k + 1] == i).any(axis=1)]

    v = np.zeros((n, n))
    for i in range(n):
        rec = reciprocal(i, k1)
        expanded = [rec]
        for c in rec:
            c_rec = reciprocal(int(c), round(k1 / 2))
            if len(np.intersect1d(c_rec, rec)) > 2 / 3 * len(c_rec):
                expanded.append(c_rec)
        idx = np.unique(np.concatenate(expanded))
        w = np.exp(-d[i, idx])
        v[i, idx] = w / w.sum()
    if k2 > 1:  # local query expansion
        v = np.stack([v[rank[i, :k2]].mean(axis=0) for i in range(n)])

    holders = [np.flatnonzero(v[:, k]) for k in range(n)]
    jaccard = np.zeros((n, n))
    for i in range(n):
        overlap = np.zeros(n)
        for k in np.flatnonzero(v[i]):
            overlap[holders[k]] += np.minimum(v[i, k], v[holders[k], k])
        jaccard[i] = 1.0 - overlap / (2.0 - overlap)
    out = (1.0 - lambda_) * jaccard + lambda_ * d
    out = (out + out.T) / 2  # exact symmetry despite summation order
    out[missing] = np.nan
    np.fill_diagonal(out, 0.0)
    return np.clip(out, 0.0, 1.0)
