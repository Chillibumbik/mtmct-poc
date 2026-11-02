"""Per-box quality in [0, 1] (spec §8.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

_WEIGHTS = {"area": 0.2, "score": 0.2, "occlusion": 0.3, "border": 0.15, "aspect": 0.15}
_BORDER_MARGIN_PX = 2.0
_BORDER_TOUCH_TERM = 0.4
_ASPECT_OK = (1.5, 3.5)  # h / w of a standing person
_FLOOR = 0.05


def _max_iou_in_frame(boxes: np.ndarray) -> np.ndarray:
    """`(n, 4)` x, y, w, h → max IoU of each box with any other box (0 for a lone box)."""
    n = len(boxes)
    if n < 2:
        return np.zeros(n)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    iw = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None)
    inter = iw * ih
    area = boxes[:, 2] * boxes[:, 3]
    iou = inter / (area[:, None] + area[None] - inter)
    np.fill_diagonal(iou, 0.0)
    return iou.max(axis=1)


def box_quality(df: pd.DataFrame, img_w: int, img_h: int) -> np.ndarray:
    """Quality of every row of `df` (columns camera_id, frame_id, x, y, w, h, score), aligned with its order.

    Weighted geometric mean `q = prod(term_i ** w_i)` of five terms in [0.05, 1], so one bad factor pulls
    the box down:
      area       0.20  percentile rank of w*h among all boxes of the same camera (bigger → more pixels for ReID)
      score      0.20  detector score
      occlusion  0.30  1 - max IoU with any other box of the same camera and frame
      border     0.15  0.4 if the box touches the image border (within 2 px), else 1
      aspect     0.15  1 while h/w is in [1.5, 3.5]; decays with |log(ratio / nearest bound)| outside
    """
    if df.empty:
        return np.zeros(0)
    area = (df["w"] * df["h"]).groupby(df["camera_id"]).rank(pct=True).to_numpy()

    occ = np.zeros(len(df))
    boxes = df[["x", "y", "w", "h"]].to_numpy(dtype=float)
    order = np.lexsort((df["frame_id"].to_numpy(), df["camera_id"].to_numpy()))
    cam, frame = df["camera_id"].to_numpy()[order], df["frame_id"].to_numpy()[order]
    cuts = np.flatnonzero((cam[1:] != cam[:-1]) | (frame[1:] != frame[:-1])) + 1
    for idx in np.split(order, cuts):
        occ[idx] = _max_iou_in_frame(boxes[idx])

    x, y, w, h = boxes.T
    touches = (x <= _BORDER_MARGIN_PX) | (y <= _BORDER_MARGIN_PX)
    touches |= (x + w >= img_w - _BORDER_MARGIN_PX) | (y + h >= img_h - _BORDER_MARGIN_PX)
    ratio = h / w
    aspect = 1.0 - np.abs(np.log(ratio / np.clip(ratio, *_ASPECT_OK)))

    terms = {
        "area": area,
        "score": df["score"].to_numpy(dtype=float),
        "occlusion": 1.0 - occ,
        "border": np.where(touches, _BORDER_TOUCH_TERM, 1.0),
        "aspect": aspect,
    }
    log_q = sum(_WEIGHTS[k] * np.log(np.clip(v, _FLOOR, 1.0)) for k, v in terms.items())
    return np.clip(np.exp(log_q), 0.0, 1.0)
