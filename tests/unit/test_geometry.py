import numpy as np
import pandas as pd
import pytest

from mtmct import registry
from mtmct.data.base import CameraInfo, SequenceInfo, SequenceMeta
from mtmct.geometry.homography import apply_h, to_world

H0 = np.array([[0.02, 0.001, -1.0], [0.0005, 0.03, -2.0], [0.0001, 0.0002, 1.0]])


def _seq(h0, h1=None) -> SequenceInfo:
    cams = [
        CameraInfo(camera_id="c0", source="v", fps=10, width=640, height=480, n_frames=100, homography_img2world=h0),
        CameraInfo(camera_id="c1", source="v", fps=10, width=640, height=480, n_frames=100, homography_img2world=h1),
    ]
    return SequenceInfo(
        dataset="t", sequence="s", root=".", cameras=cams, meta=SequenceMeta(overlap="overlap", duration_s=10)
    )


def _rows(tid, cam, foot):
    n = len(foot)
    return pd.DataFrame(
        {
            "camera_id": cam, "frame_id": np.arange(n), "ts": np.arange(n) / 10.0, "tracklet_id": tid,
            "x": 0.0, "y": 0.0, "w": 10.0, "h": 20.0, "score": 0.9, "quality": 0.5,
            "foot_x": foot[:, 0], "foot_y": foot[:, 1], "world_x": np.nan, "world_y": np.nan, "emb_idx": -1,
        }
    )  # fmt: skip


def test_homography_roundtrip():
    pts = np.random.default_rng(0).uniform(0, 600, size=(50, 2))
    world = apply_h(H0, pts)
    assert world.shape == (50, 2) and np.isfinite(world).all()
    np.testing.assert_allclose(apply_h(np.linalg.inv(H0), world), pts, atol=1e-6)


def test_apply_h_point_at_infinity_is_nan():
    h = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0]])  # w = x
    out = apply_h(h, np.array([[0.0, 5.0], [2.0, 4.0]]))
    assert np.isnan(out[0]).all() and np.allclose(out[1], [1.0, 2.0])


def test_to_world_projects_and_smooths_per_tracklet():
    rng = np.random.default_rng(1)
    world_true = np.column_stack([np.linspace(0, 5, 40), np.full(40, 2.0)])
    foot = apply_h(np.linalg.inv(H0), world_true)
    noisy = foot + rng.normal(0, 3.0, foot.shape)
    df = pd.concat([_rows(1, "c0", noisy), _rows(3, "c0", noisy[::-1]), _rows(2, "c1", noisy)], ignore_index=True)
    df = df.sample(frac=1, random_state=0)  # row order must not matter
    out = to_world(df, _seq(H0.tolist()), smooth_window=5)
    assert list(out.index) == list(df.index) and len(out) == len(df)
    c0 = out[out.tracklet_id == 1].sort_values("ts")
    raw_err = np.abs(apply_h(H0, noisy) - world_true)[2:-2].mean()
    smooth_err = np.abs(c0[["world_x", "world_y"]].to_numpy() - world_true)[2:-2].mean()
    assert smooth_err < raw_err
    assert out.loc[out.camera_id == "c1", ["world_x", "world_y"]].isna().all().all()  # no homography → NaN


def test_to_world_no_smoothing_is_exact():
    foot = np.random.default_rng(2).uniform(0, 600, size=(7, 2))
    out = to_world(_rows(1, "c0", foot), _seq(H0.tolist()), smooth_window=1)
    np.testing.assert_allclose(out[["world_x", "world_y"]].to_numpy(), apply_h(H0, foot))


def test_to_world_cuts_points_above_horizon_and_out_of_range():
    h = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, -0.01, 1.0]])  # horizon at image row y = 100
    foot = np.array([[320.0, 50.0], [320.0, 300.0], [320.0, 101.0]])  # above, below, just below (≈ 300 m away)
    out = to_world(_rows(1, "c0", foot), _seq(h.tolist()), smooth_window=1)
    assert out["world_x"].isna().tolist() == [True, False, True]
    np.testing.assert_allclose(out[["world_x", "world_y"]].iloc[1], [-1.6, -1.5])


def test_foot_point_bottom_center_registered():
    fp = registry.create("foot_point", "bottom_center")
    out = fp.foot(np.array([[10.0, 20.0, 4.0, 30.0]]), None)
    np.testing.assert_allclose(out, [[12.0, 50.0]])
    with pytest.raises(KeyError):
        registry.get("foot_point", "nope")
