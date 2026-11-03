"""YOLOX person detector on OpenVINO / ONNX Runtime (official Megvii ONNX, pre/post-processing as in
YOLOX `demo/ONNXRuntime/onnx_inference.py`: BGR 0-255 input, letterbox pad 114, grid decode with strides 8/16/32)."""

from __future__ import annotations

from functools import cache
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from mtmct.models import load_runner
from mtmct.registry import register

STRIDES = (8, 16, 32)
PERSON = 0  # COCO class index


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """BGR image → CHW float32 `(3, size, size)` (top-left placement, pad 114) and the resize ratio."""
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[: int(h * ratio), : int(w * ratio)] = cv2.resize(
        image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR
    )
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), ratio


@cache
def _grids(size: int) -> tuple[np.ndarray, np.ndarray]:
    grids, strides = [], []
    for s in STRIDES:
        n = size // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        w = np.maximum(0.0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]) + 1)
        inter = w * h
        order = order[1:][inter / (areas[i] + areas[order[1:]] - inter) <= thr]
    return keep


def postprocess(
    raw: np.ndarray, size: int, ratio: float, image_hw: tuple[int, int], conf: float, nms_thr: float
) -> np.ndarray:
    """Raw YOLOX head output `(N, 85)` → `(n, 5)` persons `x, y, w, h, score` in original pixels, clipped to image."""
    grids, strides = _grids(size)
    centre = (raw[:, :2] + grids) * strides
    wh = np.exp(raw[:, 2:4]) * strides
    scores = raw[:, 4] * raw[:, 5 + PERSON]
    keep = scores >= conf
    centre, wh, scores = centre[keep], wh[keep], scores[keep]
    xyxy = np.concatenate([centre - wh / 2, centre + wh / 2], axis=1) / ratio
    h, w = image_hw
    xyxy = np.clip(xyxy, 0, [w, h, w, h])
    kept = nms(xyxy, scores, nms_thr) if len(scores) else []
    xyxy, scores = xyxy[kept], scores[kept]
    out = np.concatenate([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2], scores[:, None]], axis=1)
    return out[(out[:, 2] > 0) & (out[:, 3] > 0)].astype(np.float32)


class _Yolox:
    LICENCE = "Apache-2.0"
    MODEL: str

    class Params(BaseModel):
        conf: float = 0.3
        nms: float = 0.45
        input_size: int = Field(640, multiple_of=32)
        backend: Literal["openvino", "onnxruntime"] = "openvino"
        device: str = "CPU"

    def __init__(self, params: Params) -> None:
        self.params = params
        s = params.input_size
        self._run = load_runner(self.MODEL, params.backend, params.device, (1, 3, s, s))

    def detect(self, images: list[np.ndarray]) -> list[np.ndarray]:
        p = self.params
        out = []
        for image in images:
            blob, ratio = letterbox(image, p.input_size)
            raw = self._run(blob[None])[0]
            out.append(postprocess(raw, p.input_size, ratio, image.shape[:2], p.conf, p.nms))
        return out


@register("detector", "yolox_s")
class YoloxS(_Yolox):
    """YOLOX-s person detector (input 640)."""

    MODEL = "yolox_s"

    class Params(_Yolox.Params):
        input_size: int = Field(640, multiple_of=32)


@register("detector", "yolox_tiny")
class YoloxTiny(_Yolox):
    """YOLOX-tiny person detector (input 416, faster)."""

    MODEL = "yolox_tiny"

    class Params(_Yolox.Params):
        input_size: int = Field(416, multiple_of=32)


@register("detector", "yolox_s_ft")
class YoloxSFt(YoloxS):
    """YOLOX-s fine-tuned on persons of the tune splits without WILDTRACK (`scripts/finetune_yolox.py`)."""

    LICENCE = "Apache-2.0 (YOLOX, COCO init); fine-tuned on open datasets (tune splits), research use"
    MODEL = "yolox_s_ft"


@register("detector", "yolox_s_ft_wt")
class YoloxSFtWt(YoloxS):
    """`yolox_s_ft` further tuned on WILDTRACK tune frames: learns its box convention and annotated zone."""

    LICENCE = "Apache-2.0 (YOLOX, COCO init); fine-tuned on open datasets (tune splits), research use"
    MODEL = "yolox_s_ft_wt"
