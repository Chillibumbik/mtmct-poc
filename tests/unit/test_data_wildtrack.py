from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from mtmct.data import wildtrack
from mtmct.data.base import load_adapters, load_sequence
from mtmct.geometry.homography import apply_h
from mtmct.io import GroundTruth, read_table

FIXTURES = Path(__file__).parents[1] / "fixtures" / "data" / "wildtrack"


def _annotation() -> list[dict]:
    return json.loads((FIXTURES / "annotation_frame.json").read_text())


def test_position_id_to_world_grid():
    # README: X = -3.0 + 0.025 * (ID % 480), Y = -9.0 + 0.025 * (ID // 480)
    assert np.allclose(
        wildtrack.position_to_world(np.array([0, 1, 480, 481])),
        [[-3, -9], [-2.975, -9], [-3, -8.975], [-2.975, -8.975]],
    )


def test_wildtrack_gt_parse():
    people = _annotation()
    gt = wildtrack.parse_annotations([[], people])  # annotated frame index is the list position
    assert list(gt.columns) == ["camera_id", "frame_id", "person_id", "x", "y", "w", "h", "world_x", "world_y"]
    assert set(gt.frame_id) == {1} and set(gt.camera_id) <= set(wildtrack.CAMERAS)
    p = people[0]
    seen = [v for v in p["views"] if v["xmax"] > 0]
    rows = gt[gt.person_id == p["personID"]]
    assert len(rows) == len(seen)  # views that do not see the person (all -1) are dropped
    v = seen[0]
    row = rows[rows.camera_id == f"C{v['viewNum'] + 1}"].iloc[0]
    assert (row.x, row.y, row.w, row.h) == (v["xmin"], v["ymin"], v["xmax"] - v["xmin"], v["ymax"] - v["ymin"])
    # one ground point per person, identical in every view (personID is the same across views)
    assert rows.groupby("person_id")[["world_x", "world_y"]].nunique().max().max() == 1
    assert np.allclose(
        rows[["world_x", "world_y"]].iloc[0], wildtrack.position_to_world(np.array([p["positionID"]]))[0]
    )
    assert (gt.x < 0).any()  # boxes partly outside the image are kept, not dropped


def test_wildtrack_calibration_roundtrip():
    extr = (FIXTURES / "extr_CVLab1.xml").read_text()
    intr = (FIXTURES / "intr_CVLab1.xml").read_text()
    H = wildtrack.homography_img2world(extr, intr)
    # ground point (cm, z=0) → image by OpenCV with the raw rvec/tvec/K → homography → the same point in metres
    rvec, tvec = wildtrack._numbers(extr, "rvec"), wildtrack._numbers(extr, "tvec")
    K = wildtrack._numbers(intr, "camera_matrix").reshape(3, 3)
    ground_m = np.array([[1.5, -2.0], [-0.5, 4.25], [3.0, 0.0]])
    ground_cm = np.c_[ground_m * 100, np.zeros(len(ground_m))]
    pixels, _ = cv2.projectPoints(ground_cm, rvec, tvec, K, None)
    assert np.abs(apply_h(H, pixels.reshape(-1, 2)) - ground_m).max() < 1e-6


def _tiny_zip(path: Path) -> None:
    people = _annotation()
    image = cv2.imencode(".png", np.zeros((32, 48, 3), np.uint8))[1].tobytes()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for frame in (0, 5):
            z.writestr(f"{wildtrack.ROOT}/annotations_positions/{frame:08d}.json", json.dumps(people))
            for cam in wildtrack.CAMERAS:
                z.writestr(f"{wildtrack.ROOT}/Image_subsets/{cam}/{frame:08d}.png", image)
        z.writestr(f"{wildtrack.ROOT}/Image_subsets/C1/00000010.png", image)  # extra unannotated frame is ignored
        for i, name in enumerate(wildtrack.CALIBRATION_NAMES):
            fixture = "CVLab1" if i < 4 else "IDIAP2"
            z.writestr(
                f"{wildtrack.ROOT}/calibrations/extrinsic/extr_{name}.xml",
                (FIXTURES / f"extr_{fixture}.xml").read_text(),
            )
            z.writestr(
                f"{wildtrack.ROOT}/calibrations/intrinsic_zero/intr_{name}.xml",
                (FIXTURES / f"intr_{fixture}.xml").read_text(),
            )


def test_wildtrack_prepare_uses_annotated_frames_as_cameras(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _tiny_zip(raw / wildtrack.ZIP_NAME)
    data = tmp_path / "data"
    wildtrack.Wildtrack().prepare(raw, data, lambda f, m: None)
    seq = load_sequence(data, "wildtrack", "main")
    assert [c.camera_id for c in seq.cameras] == wildtrack.CAMERAS and seq.overlap == "overlap"
    cam = seq.camera("C3")
    assert (cam.fps, cam.n_frames, cam.source, cam.frame_offset) == (2.0, 2, "frames/C3", 0)
    assert (cam.width, cam.height) == (48, 32) and cam.H() is not None
    assert sorted(p.name for p in (seq.root / cam.source).iterdir()) == ["000000.png", "000001.png"]
    assert seq.meta.splits == {"tune": (0.0, 0.9), "test": (0.9, 1.0), "all": (0.0, 1.0)}
    assert seq.meta.duration_s == 1.0
    gt = read_table(seq.root / "gt.parquet", GroundTruth)
    assert set(gt.frame_id) == {0, 1} and gt.world_x.notna().all() and seq.meta.n_persons == 4


def test_wildtrack_adapter_registered():
    meta = load_adapters()["wildtrack"].meta()
    assert meta.sequences == ["main"] and meta.download_gb == 6.8


@pytest.mark.data
def test_wildtrack_real_ids_and_geometry_are_consistent():
    from mtmct.config import paths

    if not (paths().data / "wildtrack" / "main" / "cameras.json").exists():
        pytest.skip("wildtrack is not prepared")
    seq = load_sequence(paths().data, "wildtrack", "main")
    gt = seq.gt()
    assert gt is not None and len(seq.cameras) == 7 and gt.frame_id.nunique() == 400
    assert all(len(list((seq.root / c.source).glob("*.png"))) == 400 for c in seq.cameras)
    # the homography puts each box foot within ~0.5 m of the annotated ground position
    for cam in seq.cameras:
        g = gt[gt.camera_id == cam.camera_id]
        foot = np.c_[g.x + g.w / 2, g.y + g.h]
        err = np.linalg.norm(apply_h(cam.H(), foot) - g[["world_x", "world_y"]].to_numpy(), axis=1)
        assert np.median(err) < 0.3
    # personID is a track id: a person moves little in 0.5 s (a handful of annotation glitches excepted)
    one = gt.drop_duplicates(["frame_id", "person_id"]).sort_values(["person_id", "frame_id"])
    step = np.hypot(one.groupby("person_id").world_x.diff(), one.groupby("person_id").world_y.diff())[
        one.groupby("person_id").frame_id.diff() == 1
    ]
    assert step.median() < 0.6 and (step > 3).sum() < 15
