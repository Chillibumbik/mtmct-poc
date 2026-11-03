from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtmct.data.base import SequenceInfo
from mtmct.data.finetune import TUNE_SOURCES, read_frames, sample_reid_boxes, tune_gt, visible_fraction
from mtmct.io import GroundTruth, write_table


def _gt(seq: SequenceInfo) -> None:
    rows = [
        {"camera_id": c, "frame_id": f, "person_id": p, "x": 5.0 + p, "y": 4.0, "w": 10.0, "h": 20.0}
        for c in ("c0", "c1")
        for f in range(50)
        for p in (1, 2)
    ]
    write_table(pd.DataFrame(rows).assign(world_x=np.nan, world_y=np.nan), seq.root / "gt.parquet", GroundTruth)


def test_ft_dataset_respects_tune_split(tiny_sequence: SequenceInfo):
    _gt(tiny_sequence)
    gt = tune_gt(tiny_sequence, "tune")  # tune = first 1 s of 2 s; c1 runs 5 frames ahead on the common clock
    assert gt.ts.max() < 1.0
    assert gt[gt.camera_id == "c0"].frame_id.max() == 24
    assert gt[gt.camera_id == "c1"].frame_id.max() == 19
    with pytest.raises(ValueError):
        tune_gt(tiny_sequence, "test")


def test_ft_sources_exclude_test_data():
    assert ("campus", "garden2") not in TUNE_SOURCES and ("chirla", "seq_002") not in TUNE_SOURCES
    assert "test" not in TUNE_SOURCES.values()
    assert TUNE_SOURCES[("wildtrack", "main")] == "tune" and TUNE_SOURCES[("aic24", "scene_041_all")] == "tune"


def test_visible_fraction_counts_closer_occluder_and_border():
    gt = pd.DataFrame(
        {
            "camera_id": "c",
            "frame_id": 0,
            "x": [0.0, 5.0, 95.0],
            "y": [0.0, 10.0, 0.0],
            "w": 10.0,
            "h": [20.0, 20.0, 20.0],
        }
    )
    vis = visible_fraction(gt, 100, 100)
    # box 1 (bottom 30) is closer than box 0 (bottom 20) and covers half of its width over 10 of 20 rows
    np.testing.assert_allclose(vis, [1 - 0.25, 1.0, 0.5])


def test_sample_reid_boxes_caps_and_spreads():
    gt = pd.DataFrame(
        {"camera_id": "c", "frame_id": np.arange(100), "person_id": 7, "x": 0.0, "y": 0.0, "w": 40.0, "h": 100.0}
    )
    picked = sample_reid_boxes(gt, per_camera=5, min_h=64)
    assert picked.frame_id.tolist() == [0, 25, 50, 74, 99]
    assert sample_reid_boxes(gt.assign(h=50.0), min_h=64).empty


def test_read_frames_returns_requested_frames(tiny_sequence: SequenceInfo):
    cam = tiny_sequence.cameras[0]
    got = list(read_frames(tiny_sequence, cam, [7, 3, 40]))
    assert [f for f, _ in got] == [3, 7, 40]
    np.testing.assert_allclose([img.mean() for _, img in got], [3, 7, 40], atol=1.5)  # MJPG rounding
