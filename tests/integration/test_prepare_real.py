"""Real prepared data (run `mtmct fetch/prepare <dataset>` first); skipped when absent."""

from __future__ import annotations

import pytest

from mtmct.config import paths
from mtmct.data.base import load_sequence
from mtmct.io import GroundTruth, read_table

pytestmark = pytest.mark.data

EXPECTED = {
    ("epfl_terrace", "terrace1"): (["c0", "c1", "c2", "c3"], 25.0, (360, 288)),
    ("campus", "garden1"): (["Contour2", "HC2", "HC3", "IP1"], 30.0, (1920, 1080)),
    ("campus", "garden2"): (["HC1", "HC2", "HC3", "HC4"], 30.0, (1920, 1080)),
}


@pytest.mark.parametrize(("dataset", "sequence"), list(EXPECTED))
def test_prepared_sequence_is_consistent(dataset, sequence):
    root = paths().data / dataset / sequence
    if not (root / "cameras.json").exists():
        pytest.skip(f"{dataset}/{sequence} is not prepared")
    cams, fps, size = EXPECTED[(dataset, sequence)]
    seq = load_sequence(paths().data, dataset, sequence)
    assert [c.camera_id for c in seq.cameras] == cams
    for cam in seq.cameras:
        assert abs(cam.fps - fps) < 0.1 and (cam.width, cam.height) == size
        assert (seq.root / cam.source).exists() and cam.H() is not None
    gt = read_table(seq.root / "gt.parquet", GroundTruth)
    for cam in seq.cameras:
        g = gt[gt.camera_id == cam.camera_id]
        assert g.frame_id.max() < cam.n_frames
        assert (g.x + g.w).min() > 0 and (g.y + g.h).min() > 0 and g.x.max() < cam.width and g.y.max() < cam.height
    assert gt.world_x.notna().mean() > 0.99
    # the same person stands at roughly the same ground point in every camera that sees them
    spread = gt.groupby(["frame_id", "person_id"])[["world_x", "world_y"]].std().dropna()
    assert spread.median().max() < 1.5
