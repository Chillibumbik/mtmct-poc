"""Shared body-ReID plumbing: crop preprocessing and L2 normalisation around one ONNX model."""

from __future__ import annotations

from typing import ClassVar, Literal

import cv2
import numpy as np
from pydantic import BaseModel

from mtmct.models import load_runner

H, W = 256, 128


class OnnxReid:
    MODEL: ClassVar[str]
    DIM: ClassVar[int]
    MEAN: ClassVar[tuple[float, float, float] | None] = None  # RGB, 0-1 scale; None = raw BGR 0-255 input
    STD: ClassVar[tuple[float, float, float]] = (1.0, 1.0, 1.0)

    class Params(BaseModel):
        backend: Literal["openvino", "onnxruntime"] = "openvino"
        device: str = "CPU"

    def __init__(self, params: Params) -> None:
        self.params = params
        self.dim = self.DIM
        self._run = load_runner(self.MODEL, params.backend, params.device, (-1, 3, H, W))

    def _prepare(self, crop: np.ndarray) -> np.ndarray:
        x = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        if self.MEAN is not None:
            x = (x[..., ::-1] / 255.0 - self.MEAN) / self.STD
        return x.transpose(2, 0, 1)

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """BGR crops (non-empty images, one model batch; the caller sizes it) → `(n, dim)` float32, L2-normalised."""
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = self._run(np.ascontiguousarray([self._prepare(c) for c in crops], dtype=np.float32)).astype(np.float32)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
