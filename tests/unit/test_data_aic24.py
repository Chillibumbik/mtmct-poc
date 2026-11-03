from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import write_video

from mtmct.data import aic24
from mtmct.data.base import load_adapters, load_sequence
from mtmct.data.common import apply_homography

FIXTURES = Path(__file__).parents[1] / "fixtures" / "data" / "aic24"


def test_aic24_gt_parse_and_homography():
    gt = aic24.read_gt(FIXTURES / "ground_truth.txt", ["camera_0361"])
    assert list(gt.columns) == ["camera_id", "frame_id", "person_id", "x", "y", "w", "h"]
    assert set(gt.camera_id) == {"camera_0361"}
    first = gt.iloc[0]  # raw: "361 2 2 918 414 78 216 -6.35622 -6.68688" (cam obj frame x y w h xworld yworld)
    assert (first.person_id, first.frame_id, first.x, first.y, first.w, first.h) == (2, 2, 918, 414, 78, 216)
    # the stored homography is ground → image; the inverse puts the foot point near the GT's own ground position
    H = aic24.read_homography(FIXTURES / "calibration_0361.json")
    foot = np.array([[918 + 78 / 2, 414 + 216]])
    assert np.abs(apply_homography(H, foot)[0] - [-6.35622, -6.68688]).max() < 0.5


def test_aic24_adapter_registered_with_two_sequences():
    meta = load_adapters()["aic24"].meta()
    assert meta.sequences == ["scene_041", "scene_041_all"] and not meta.real and not meta.requires_auth


def test_aic24_prepare_four_and_all_cameras(tmp_path):
    raw = tmp_path / "raw" / aic24.SCENE
    for cam in aic24.CAMERAS:
        write_video(raw / cam / "video.mp4", 6, 30.0, size=(64, 48))
        (raw / cam / "calibration.json").write_text((FIXTURES / "calibration_0361.json").read_text())
    (raw / "ground_truth.txt").write_text((FIXTURES / "ground_truth.txt").read_text())
    data = tmp_path / "data"
    four, full = aic24.Aic24().prepare(tmp_path / "raw", data, lambda f, m: None)
    assert [c.camera_id for c in four.cameras] == aic24.CAMERAS[:4]
    assert len(full.cameras) == 10
    seq = load_sequence(data, "aic24", "scene_041")
    assert seq.overlap == "overlap" and seq.meta.splits["tune"] == (0.0, 0.5) and seq.meta.splits["test"] == (0.5, 1.0)
    gt = seq.gt()
    assert gt is not None and set(gt.camera_id) == {"camera_0361", "camera_0362"} and gt.world_x.notna().all()
    assert seq.camera("camera_0361").H() is not None


@pytest.mark.data
def test_aic24_real_gt_world_matches_homography():
    from mtmct.config import paths

    root = paths().data / "aic24" / "scene_041"
    if not (root / "cameras.json").exists():
        pytest.skip("aic24 is not prepared")
    seq = load_sequence(paths().data, "aic24", "scene_041")
    gt = seq.gt()
    assert gt is not None and len(seq.cameras) == 4
    raw = pd.read_csv(
        paths().raw / "aic24" / aic24.SCENE / "ground_truth.txt", sep=" ", header=None, names=aic24._GT_COLUMNS
    )
    raw["camera_id"] = "camera_" + raw["camera"].astype(str).str.zfill(4)
    merged = gt.merge(raw, on=["camera_id", "frame_id", "person_id"])
    err = np.hypot(merged.world_x - merged.xworld, merged.world_y - merged.yworld)
    assert err.median() < 0.5
