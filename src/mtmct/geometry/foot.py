"""Foot point of a person box (spec §8.6)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from mtmct.registry import register


@register("foot_point", "bottom_center")
class BottomCenter:
    """Bottom-centre of the box."""

    LICENCE = "MIT"

    class Params(BaseModel):
        pass

    def __init__(self, params: Params) -> None:
        self.params = params

    def foot(self, boxes: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        return np.column_stack([b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3]])
