from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mtmct import registry
from mtmct.config import paths
from mtmct.detect.yolox import _grids, letterbox, postprocess
from mtmct.models import MODELS

FIXTURES = Path(__file__).parents[1] / "fixtures" / "models"
# EPFL Terrace c0 frame 525, ground-truth persons as x, y, w, h
GT_BOXES = np.array([[169, 28, 40, 135], [208, 49, 61, 191]], dtype=float)


def needs_weights(name: str):
    path = paths().models / MODELS[name].filename
    return pytest.mark.skipif(not path.exists(), reason=f"{path} missing: run `uv run python -m mtmct.models`")


def iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return iw * ih / (a[2] * a[3] + b[2] * b[3] - iw * ih)


def head_output(size: int, rows: list[tuple[int, float, float, float, float, float, int]]) -> np.ndarray:
    """Raw YOLOX output `(N, 85)`; each row = (index, dx, dy, log_w, log_h, objectness, class)."""
    raw = np.zeros((len(_grids(size)[0]), 85), dtype=np.float32)
    for idx, dx, dy, lw, lh, obj, cls in rows:
        raw[idx, :4] = (dx, dy, lw, lh)
        raw[idx, 4] = obj
        raw[idx, 5 + cls] = 0.9
    return raw


def test_letterbox_pads_with_114_and_keeps_ratio():
    image = np.random.default_rng(0).integers(0, 255, (288, 360, 3), dtype=np.uint8)
    blob, ratio = letterbox(image, 640)
    assert blob.shape == (3, 640, 640) and blob.dtype == np.float32
    assert ratio == pytest.approx(640 / 360)
    assert (blob[:, 512:, :] == 114).all() and (blob[:, :, :] != 114).any()
    expected = cv2.resize(image, (640, 512), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(blob[:, :512, :].transpose(1, 2, 0), expected.astype(np.float32))


def test_yolox_letterbox_coords_in_original_pixels():
    size, (h, w) = 640, (288, 360)
    ratio = size / w
    # letterboxed person: centre (320, 200), 100 x 200 px; stride-32 cell (col 10, row 6)
    idx = 80 * 80 + 40 * 40 + 6 * 20 + 10
    raw = head_output(size, [(idx, 0.0, 0.25, np.log(100 / 32), np.log(200 / 32), 1.0, 0)])
    out = postprocess(raw, size, ratio, (h, w), conf=0.3, nms_thr=0.45)
    assert out.shape == (1, 5)
    np.testing.assert_allclose(
        out[0, :4], [(320 - 50) / ratio, (200 - 100) / ratio, 100 / ratio, 200 / ratio], atol=1e-3
    )
    assert out[0, 4] == pytest.approx(0.9)


def test_postprocess_clips_to_image():
    size, (h, w) = 640, (288, 360)
    idx = 80 * 80 + 40 * 40 + 0 * 20 + 0  # centre (0, 0): box sticks out at left/top
    raw = head_output(size, [(idx, 0.0, 0.0, np.log(100 / 32), np.log(100 / 32), 1.0, 0)])
    out = postprocess(raw, size, size / w, (h, w), conf=0.3, nms_thr=0.45)
    x, y, bw, bh, _ = out[0]
    assert x == 0 and y == 0 and bw == pytest.approx(50 / (size / w)) and bh == pytest.approx(50 / (size / w))


def test_postprocess_keeps_persons_only_and_applies_nms():
    size, (h, w) = 640, (640, 640)
    base = 80 * 80 + 40 * 40
    raw = head_output(
        size,
        [
            (base + 6 * 20 + 10, 0.0, 0.0, np.log(100 / 32), np.log(200 / 32), 1.0, 0),
            (base + 6 * 20 + 11, -1.0, 0.0, np.log(100 / 32), np.log(200 / 32), 0.8, 0),  # same box, lower score
            (base + 12 * 20 + 3, 0.0, 0.0, np.log(100 / 32), np.log(200 / 32), 1.0, 2),  # a car
        ],
    )
    out = postprocess(raw, size, 1.0, (h, w), conf=0.3, nms_thr=0.45)
    assert out.shape == (1, 5) and out[0, 4] == pytest.approx(0.9)


def test_postprocess_nothing_above_conf_returns_empty():
    out = postprocess(head_output(416, []), 416, 1.0, (416, 416), conf=0.3, nms_thr=0.45)
    assert out.shape == (0, 5)


def test_detector_registered_with_input_sizes():
    assert registry.get("detector", "yolox_s").Params().input_size == 640
    assert registry.get("detector", "yolox_tiny").Params().input_size == 416
    assert registry.get("detector", "yolox_s").LICENCE == "Apache-2.0"


@pytest.mark.parametrize(
    ("name", "backend"),
    [
        pytest.param("yolox_s", "openvino", marks=needs_weights("yolox_s")),
        pytest.param("yolox_s", "onnxruntime", marks=needs_weights("yolox_s")),
        pytest.param("yolox_tiny", "openvino", marks=needs_weights("yolox_tiny")),
    ],
)
def test_yolox_detects_person_on_sample(name: str, backend: str):
    image = cv2.imread(str(FIXTURES / "terrace_c0_f525.jpg"))
    detector = registry.create("detector", name, {"backend": backend})
    (dets,) = detector.detect([image])
    assert dets.shape[1] == 5 and dets.dtype == np.float32
    assert (dets[:, 4] > 0.5).any()
    h, w = image.shape[:2]
    assert (dets[:, 0] >= 0).all() and (dets[:, 1] >= 0).all()
    assert (dets[:, 0] + dets[:, 2] <= w + 1e-3).all() and (dets[:, 1] + dets[:, 3] <= h + 1e-3).all()
    for gt in GT_BOXES:  # both people of the frame are found with the right extent
        assert max(iou_xywh(gt, d[:4]) for d in dets) > 0.6


@needs_weights("yolox_tiny")
def test_yolox_blank_frame_and_batch():
    detector = registry.create("detector", "yolox_tiny")
    blank = np.full((288, 360, 3), 128, dtype=np.uint8)
    sample = cv2.imread(str(FIXTURES / "terrace_c0_f525.jpg"))
    empty, found = detector.detect([blank, sample])
    assert empty.shape == (0, 5)
    assert len(found) >= 1
