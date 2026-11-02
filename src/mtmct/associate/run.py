"""Cross-camera association: costs → fused matrix → solver (spec §8.7)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from mtmct import registry
from mtmct.associate.combine import combine, same_camera_overlap
from mtmct.associate.rerank import k_reciprocal_rerank
from mtmct.components import AssocContext, CostMatrix
from mtmct.config import AssociateCfg
from mtmct.data.base import SequenceInfo


def build_context(
    seq: SequenceInfo,
    tracklets: pd.DataFrame,
    portraits: pd.DataFrame,
    portrait_emb: np.ndarray,
    galleries: list[np.ndarray],
    topology: dict[str, Any] | None = None,
    warn: Callable[[str], None] | None = None,
) -> AssocContext:
    return AssocContext(
        seq=seq, tracklets=tracklets, portraits=portraits, portrait_emb=portrait_emb, gallery_emb=galleries,
        topology=topology, warn=warn or (lambda msg: None),
    )  # fmt: skip


def _fused(ctx: AssocContext, cfg: AssociateCfg) -> tuple[np.ndarray, np.ndarray]:
    parts: list[tuple[CostMatrix, float]] = []
    for name, spec in cfg.costs.items():
        cm: CostMatrix = registry.create("cost", name, spec.params)(ctx)
        if name == "appearance" and cfg.rerank is not None:
            cm = CostMatrix(k_reciprocal_rerank(cm.dist, **cfg.rerank), cm.cannot_link)
        parts.append((cm, spec.weight))
    if cfg.rerank is not None and "appearance" not in cfg.costs:
        ctx.warn("re-ranking ignored: it applies to the appearance cost, which is not enabled")
    n = len(ctx.portraits)
    same_cam = (
        same_camera_overlap(ctx.portraits)
        if cfg.constraints.get("same_camera_overlap") == "cannot_link"
        else np.zeros((n, n), bool)
    )
    return combine(parts, same_cam)


def _solve(ctx: AssocContext, name: str, params: dict[str, Any], dist: np.ndarray, cl: np.ndarray) -> pd.Series:
    solver = registry.create("solver", name, params)
    labels = solver.solve(dist, cl, cameras=ctx.portraits["camera_id"].to_numpy())
    index = pd.Index(ctx.portraits["tracklet_id"].to_numpy(), name="tracklet_id")
    return pd.Series(np.asarray(labels, dtype=np.int64), index=index, name="global_id")


def associate(ctx: AssocContext, cfg: AssociateCfg) -> pd.Series:
    """tracklet_id → global_id (0..k-1)."""
    dist, cl = _fused(ctx, cfg)
    return _solve(ctx, cfg.solver.name, cfg.solver.params, dist, cl)


def threshold_labels(ctx: AssocContext, cfg: AssociateCfg, thresholds: list[float]) -> dict[float, pd.Series]:
    """Labels per solver threshold; the cost matrices are computed once."""
    dist, cl = _fused(ctx, cfg)
    return {t: _solve(ctx, cfg.solver.name, {**cfg.solver.params, "threshold": t}, dist, cl) for t in thresholds}
