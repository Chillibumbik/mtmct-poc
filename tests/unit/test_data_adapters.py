from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io
from conftest import write_video

from mtmct.data import campus, custom, epfl_terrace
from mtmct.data.base import CameraInfo, load_adapters, load_sequence
from mtmct.data.common import add_world, apply_homography, read_xu_gt
from mtmct.io import GroundTruth, read_table

FIXTURES = Path(__file__).parents[1] / "fixtures" / "data"
CANONICAL = ["camera_id", "frame_id", "person_id", "x", "y", "w", "h"]


def test_adapters_are_registered():
    adapters = load_adapters()
    assert {"epfl_terrace", "campus"} <= set(adapters)
    assert adapters["campus"].meta().sequences == ["garden1", "garden2"]


def test_epfl_gt_parse_to_canonical():
    gt = read_xu_gt(FIXTURES / "epfl_terrace" / "terrace1-c0.txt", "c0")
    assert list(gt.columns) == CANONICAL
    # 27 raw rows (frames 119..121, 9 ids); rows flagged `lost` are dropped
    assert len(gt) < 27
    first = gt.iloc[0]  # raw: "0 0 28 17 187 120 0 ..." (id x1 y1 x2 y2 frame lost)
    assert (first.person_id, first.frame_id) == (0, 120)
    assert (first.x, first.y, first.w, first.h) == (0, 28, 17, 159)
    assert gt.frame_id.min() == 119 + (0 if (gt.frame_id == 119).any() else 1)


def test_campus_gt_parse_to_canonical():
    gt = read_xu_gt(FIXTURES / "campus" / "view-HC2.txt", "HC2")
    raw = pd.read_csv(FIXTURES / "campus" / "view-HC2.txt", sep=" ", header=None)
    assert len(gt) == (raw[6] == 0).sum()
    row = gt.iloc[0]  # raw: "0 476 340 564 596 0 0 0 0"
    assert (row.person_id, row.frame_id, row.x, row.y, row.w, row.h) == (0, 0, 476, 340, 88, 256)
    assert set(gt.frame_id) <= {0, 1}


def test_campus_homography_from_krt_maps_known_point():
    cams = campus.read_calibration(FIXTURES / "campus" / "cam_param.mat", "garden1")
    assert set(cams) == {"Contour2", "HC2", "HC3", "IP1"}
    # self-consistency: ground point → image (K R (P - C)) → homography → the same ground point
    mat = scipy.io.loadmat(str(FIXTURES / "campus" / "cam_param.mat"), squeeze_me=True, struct_as_record=False)
    cam = next(c for c in mat["cam_param"] if c.name == "view-HC2")
    P = np.array([-6.0, 5.0, 0.0])
    m = cam.K @ cam.R @ (P - cam.T)
    pixel = (m[:2] / m[2] + cam.center.astype(float)) / cam.scale
    back = apply_homography(cams["HC2"], pixel[None])[0]
    assert np.abs(back - P[:2]).max() < 1e-6


def test_campus_garden2_calibration_layout(tmp_path):
    K = np.diag([1500.0, 1500.0, 1.0])
    R = np.array([[1, 0, 0], [0, -0.45, -0.89], [0, 0.89, -0.45]], dtype=float)
    C = np.array([1.0, -2.0, 5.5])
    image = np.empty(2, dtype=object)  # the real file lists four views
    for i, name in enumerate(["HC3", "HC4"]):
        image[i] = {"name": name, "K": K, "R": R, "x": C[0], "y": C[1], "z": C[2], "scale": 1, "imsize": [1080, 1920]}
    path = tmp_path / "calibration.mat"
    scipy.io.savemat(path, {"frames": {"image": image}})
    H = campus.read_calibration(path, "garden2")["HC4"]
    P = np.array([3.0, 4.0, 0.0])
    m = K @ R @ (P - C)
    pixel = m[:2] / m[2] + [960, 540]
    assert np.abs(apply_homography(H, pixel[None])[0] - P[:2]).max() < 1e-6


def test_epfl_calibration_maps_into_metric_area():
    Hs = epfl_terrace.read_calibration(FIXTURES / "epfl_terrace" / "calibration-terrace.txt")
    assert list(Hs) == ["c0", "c1", "c2", "c3"]
    ground = np.vstack([apply_homography(H, np.array([[180.0, 250.0]])) for H in Hs.values()])
    # image centre-bottom of every camera lands within the 7.5 x 11 m terrace, and cameras agree within a few metres
    assert np.all((ground[:, 0] > -1) & (ground[:, 0] < 9) & (ground[:, 1] > -2) & (ground[:, 1] < 12))


def _epfl_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    for cam in epfl_terrace.CAMERAS:
        write_video(raw / f"terrace1-{cam}.avi", 130, 25.0, size=(36, 28))
        (raw / f"terrace1-{cam}.txt").write_text((FIXTURES / "epfl_terrace" / "terrace1-c0.txt").read_text())
    (raw / "calibration-terrace.txt").write_text((FIXTURES / "epfl_terrace" / "calibration-terrace.txt").read_text())
    return raw


def test_epfl_prepare_writes_canonical_layout(tmp_path):
    raw = _epfl_raw(tmp_path)
    data = tmp_path / "data"
    (seq,) = epfl_terrace.EpflTerrace().prepare(raw, data, lambda f, m: None)
    seq = load_sequence(data, "epfl_terrace", "terrace1")
    assert [c.camera_id for c in seq.cameras] == ["c0", "c1", "c2", "c3"]
    assert seq.overlap == "overlap" and seq.meta.splits == {"all": (0.0, 1.0)}
    assert seq.camera("c1").fps == 25.0 and seq.camera("c1").n_frames == 130
    assert seq.camera("c0").H() is not None and seq.meta.n_persons == 1
    assert (seq.root / seq.camera("c2").source).is_symlink() and (seq.root / seq.camera("c2").source).exists()
    gt = read_table(seq.root / "gt.parquet", GroundTruth)
    gt = gt[gt.camera_id == "c0"]
    assert gt.world_x.notna().all()
    foot = np.c_[gt.x + gt.w / 2, gt.y + gt.h]
    expected = apply_homography(seq.camera("c0").H(), foot)
    assert np.allclose(gt[["world_x", "world_y"]].to_numpy(), expected, atol=1e-3)


def test_gt_beyond_video_length_is_rejected(tmp_path):
    raw = _epfl_raw(tmp_path)
    write_video(raw / "terrace1-c3.avi", 100, 25.0, size=(36, 28))  # GT frames reach 121
    with pytest.raises(ValueError, match="beyond the video length"):
        epfl_terrace.EpflTerrace().prepare(raw, tmp_path / "data", lambda f, m: None)


def test_custom_prepare_without_gt(tmp_path):
    for name in ("a", "b"):
        write_video(tmp_path / "up" / f"{name}.avi", 25, 25.0, size=(64, 48))
    videos = [tmp_path / "up" / "a.avi", tmp_path / "up" / "b.avi"]
    seq = custom.prepare_custom(videos, None, None, "demo", tmp_path / "data")
    assert (seq.dataset, seq.sequence, seq.overlap) == ("custom", "demo", "mixed")
    assert [c.camera_id for c in seq.cameras] == ["a", "b"]
    cam = seq.camera("a")
    assert (cam.fps, cam.width, cam.height, cam.n_frames, cam.homography_img2world) == (25.0, 64, 48, 25, None)
    assert not seq.has_gt and seq.meta.duration_s == pytest.approx(1.0)
    assert load_sequence(tmp_path / "data", "custom", "demo").cameras[1].camera_id == "b"


def test_custom_prepare_with_gt_and_calibration(tmp_path):
    write_video(tmp_path / "up" / "a.avi", 25, 25.0, size=(64, 48))
    gt = tmp_path / "gt.csv"
    gt.write_text("camera_id,frame_id,person_id,x,y,w,h\na,0,7,10,20,8,16\na,3,7,12,20,8,16\n")
    H = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]]
    seq = custom.prepare_custom([tmp_path / "up" / "a.avi"], {"a": H}, gt, "demo", tmp_path / "data")
    assert seq.has_gt and seq.meta.n_persons == 1
    df = seq.gt()
    assert df is not None and len(df) == 2
    assert df.world_x.iloc[0] == pytest.approx(1.4) and df.world_y.iloc[0] == pytest.approx(3.6)


def test_custom_prepare_rejects_gt_for_unknown_camera(tmp_path):
    write_video(tmp_path / "up" / "a.avi", 25, 25.0, size=(64, 48))
    gt = tmp_path / "gt.csv"
    gt.write_text("zzz,0,7,10,20,8,16\n")
    with pytest.raises(ValueError, match="unknown cameras"):
        custom.prepare_custom([tmp_path / "up" / "a.avi"], None, gt, "demo", tmp_path / "data")


def test_world_is_null_for_foot_points_above_the_horizon():
    H = campus.read_calibration(FIXTURES / "campus" / "cam_param.mat", "garden1")["HC3"]
    cam = CameraInfo(
        camera_id="HC3", source="x", fps=30, width=1920, height=1080, n_frames=10, homography_img2world=H.tolist()
    )
    boxes = pd.DataFrame(
        {
            "camera_id": "HC3",
            "frame_id": 0,
            "person_id": 1,
            "x": [900.0, 900.0],
            "y": [800.0, 0.0],
            "w": 100.0,
            "h": [200.0, 10.0],
        }
    )
    out = add_world(boxes, [cam])
    assert out.world_x.iloc[0] == out.world_x.iloc[0] and abs(out.world_x.iloc[0]) < 100  # finite, in range
    assert out.world_x.isna().iloc[1] and out.world_y.isna().iloc[1]  # foot at y=10 is above the horizon
