"""Component interfaces shared by all zones. Implementations register themselves via `mtmct.registry`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mtmct.data.base import SequenceInfo


class Detector(Protocol):
    def detect(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """BGR images → per image `(n, 5)` float32 array `x, y, w, h, score` in original pixels, persons only."""
        ...


class Tracker(Protocol):
    def reset(self) -> None: ...

    def update(self, dets: np.ndarray, image: np.ndarray) -> np.ndarray:
        """`(n, 5)` detections of one frame → `(m, 6)` float array `x, y, w, h, score, track_id` (track_id ≥ 0)."""
        ...


class ReID(Protocol):
    dim: int

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """BGR crops → `(n, dim)` float32, L2-normalised rows."""
        ...


class FootPoint(Protocol):
    def foot(self, boxes: np.ndarray, image: np.ndarray | None) -> np.ndarray:
        """`(n, 4)` x, y, w, h → `(n, 2)` image foot points."""
        ...


@dataclass
class CostMatrix:
    """`dist[i, j]` in [0, 1] (NaN = this cost has no information); `cannot_link[i, j]` forbids merging."""

    dist: np.ndarray
    cannot_link: np.ndarray


@dataclass
class AssocContext:
    """Everything a cost needs. Row i of `portraits`, `portrait_emb`, `gallery_emb` is the same tracklet."""

    seq: SequenceInfo
    tracklets: pd.DataFrame  # Tracklets schema (with world_x/world_y if geometry ran)
    portraits: pd.DataFrame  # Portraits schema
    portrait_emb: np.ndarray  # (T, D) L2-normalised
    gallery_emb: list[np.ndarray]  # T arrays of shape (k_i, D)
    topology: dict[str, Any] | None = None
    warn: Callable[[str], None] = field(default=lambda msg: None)


class Cost(Protocol):
    def __call__(self, ctx: AssocContext) -> CostMatrix: ...


class Solver(Protocol):
    def solve(self, dist: np.ndarray, cannot_link: np.ndarray, cameras: np.ndarray | None = None) -> np.ndarray:
        """Symmetric `(T, T)` distance + constraints (+ camera of each tracklet) → `(T,)` int64 labels from 0."""
        ...
