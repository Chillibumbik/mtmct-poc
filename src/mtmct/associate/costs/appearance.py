"""Appearance cost: cosine distance of tracklet portraits or galleries (spec §8.7)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from mtmct.components import AssocContext, CostMatrix
from mtmct.registry import register

_BLOCK_ELEMS = 1 << 22  # bounds the (rows, T, k*k) temporary of the gallery distance


def _gallery_dist(galleries: list[np.ndarray], how: Literal["median", "min"]) -> np.ndarray:
    """`d[i, j]` = median / min of `1 − cos` over all embedding pairs of galleries i and j (NaN if one is empty)."""
    t = len(galleries)
    k = max((len(g) for g in galleries), default=0)
    out = np.full((t, t), np.nan)
    if k == 0:
        return out
    dim = next(g.shape[1] for g in galleries if len(g))
    padded = np.zeros((t, k, dim), np.float32)
    mask = np.zeros((t, k), bool)
    for i, g in enumerate(galleries):
        padded[i, : len(g)], mask[i, : len(g)] = g, True
    flat = padded.reshape(t * k, dim)
    sizes = mask.sum(axis=1)
    step = max(1, _BLOCK_ELEMS // (t * k * k))
    for a in range(0, t, step):
        b = min(t, a + step)
        d = (
            np.clip(1.0 - flat[a * k : b * k] @ flat.T, 0.0, 1.0)
            .reshape(b - a, k, t, k)
            .transpose(0, 2, 1, 3)
            .reshape(b - a, t, k * k)
        )
        valid = (mask[a:b, None, :, None] & mask[None, :, None, :]).reshape(b - a, t, k * k)
        d = np.where(valid, d, np.inf)  # padding sorts last
        m = sizes[a:b, None] * sizes[None, :]  # valid pairs per (i, j)
        if how == "min":
            out[a:b] = d.min(axis=-1)
        else:  # masked median without np.nanmedian (slow): middle element(s) of the sorted valid prefix
            d.sort(axis=-1)
            lo = np.take_along_axis(d, np.maximum(m - 1, 0)[..., None] // 2, axis=-1)[..., 0]
            hi = np.take_along_axis(d, (m // 2)[..., None], axis=-1)[..., 0]
            out[a:b] = (lo + hi) / 2
        out[a:b][m == 0] = np.nan
    return out


@register("cost", "appearance")
class Appearance:
    """1 − cosine similarity of tracklet portraits (mean) or of their galleries (median / min over pairs)."""

    LICENCE = "MIT"

    class Params(BaseModel):
        agg: Literal["mean", "gallery_median", "gallery_min"] = "mean"

    def __init__(self, params: Params) -> None:
        self.params = params

    def __call__(self, ctx: AssocContext) -> CostMatrix:
        if self.params.agg == "mean":
            p = np.asarray(ctx.portrait_emb, dtype=np.float64)
            dist = 1.0 - p @ p.T
            missing = ~p.any(axis=1)
        else:
            dist = _gallery_dist(ctx.gallery_emb, "median" if self.params.agg == "gallery_median" else "min")
            missing = np.array([len(g) == 0 for g in ctx.gallery_emb], dtype=bool)
        dist = np.clip(dist, 0.0, 1.0)
        dist[missing, :] = np.nan
        dist[:, missing] = np.nan
        return CostMatrix(dist, np.zeros(dist.shape, bool))
