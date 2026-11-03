from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mtmct import registry
from mtmct.config import paths
from mtmct.models import EXPORT_HINT, MODELS

FIXTURES = Path(__file__).parents[1] / "fixtures" / "models"
DIMS = {"osnet_ain_x1_0": 512, "osnet_x0_25": 512, "omz_0287": 256}


def needs_weights(name: str):
    path = paths().models / MODELS[name].filename
    hint = EXPORT_HINT if MODELS[name].kind == "export" else "uv run python -m mtmct.models"
    return pytest.mark.skipif(not path.exists(), reason=f"{path} missing: run `{hint}`")


def weights(*names: str):
    return pytest.mark.parametrize("name", [pytest.param(n, marks=needs_weights(n)) for n in names])


def crops() -> list[np.ndarray]:
    """Person 0 at frame 525 and 550 (1 s apart), person 1 at frame 525; EPFL Terrace camera 0."""
    return [cv2.imread(str(FIXTURES / f)) for f in ("person0_f525.jpg", "person0_f550.jpg", "person1_f525.jpg")]


@weights(*DIMS)
def test_reid_embed_shape_and_l2_norm(name: str):
    reid = registry.create("reid", name)
    emb = reid.embed(crops())
    assert reid.dim == DIMS[name]
    assert emb.shape == (3, DIMS[name]) and emb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    one_by_one = np.vstack([reid.embed([c]) for c in crops()])
    np.testing.assert_allclose(emb, one_by_one, atol=1e-4)  # batching does not change the result


@weights(*DIMS)
def test_reid_same_crop_higher_similarity_than_different_person(name: str):
    same_a, same_b, other = registry.create("reid", name).embed(crops())
    assert same_a @ same_b > same_a @ other + 0.1


@weights("osnet_ain_x1_0", "osnet_x0_25")
def test_reid_openvino_and_onnxruntime_agree(name: str):
    ov = registry.create("reid", name, {"backend": "openvino"}).embed(crops())
    ort = registry.create("reid", name, {"backend": "onnxruntime"}).embed(crops())
    np.testing.assert_allclose(ov, ort, atol=1e-3)


@needs_weights("osnet_x0_25")
def test_reid_empty_input():
    assert registry.create("reid", "osnet_x0_25").embed([]).shape == (0, 512)


def test_reid_models_registered():
    for name, dim in DIMS.items():
        cls = registry.get("reid", name)
        assert dim == cls.DIM
    with pytest.raises(ValueError):  # the OMZ model is an IR: no ONNX Runtime
        registry.create("reid", "omz_0287", {"backend": "onnxruntime"})
