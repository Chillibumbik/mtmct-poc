import numpy as np
import pandas as pd

from mtmct.config import RefineCfg
from mtmct.data.base import CameraInfo
from mtmct.io import TRACKLET_ID_CAMERA_FACTOR, Tracklets
from mtmct.refine.quality import box_quality
from mtmct.refine.refine import refine
from mtmct.video.reader import selected_frame_ids

FPS = 10.0
D = 16
GRID = {"c0": np.arange(100), "c1": np.arange(100)}  # every frame processed


def _unit(rng: np.random.Generator, v: np.ndarray | None = None, noise: float = 0.05) -> np.ndarray:
    v = rng.normal(size=D) if v is None else v
    x = v + rng.normal(0, noise, D)
    return x / np.linalg.norm(x)


def _tracklet(tid, frames, emb_idx, cam="c0", x0=100.0, vx=2.0, score=0.9):
    f = np.asarray(frames)
    x = x0 + vx * f
    return pd.DataFrame(
        {
            "camera_id": cam, "frame_id": f, "ts": f / FPS, "tracklet_id": tid,
            "x": x, "y": 50.0, "w": 40.0, "h": 100.0, "score": score, "quality": 0.5,
            "foot_x": x + 20.0, "foot_y": 150.0, "world_x": np.nan, "world_y": np.nan,
            "emb_idx": emb_idx,
        }
    )  # fmt: skip


def _embs(rng, n, v=None):
    return [_unit(rng, v) for _ in range(n)]


def test_refine_splits_on_appearance_jump():
    rng = np.random.default_rng(0)
    va, vb = rng.normal(size=D), rng.normal(size=D)
    emb = np.array(_embs(rng, 30, va) + _embs(rng, 30, vb))
    df = _tracklet(5, range(60), np.arange(60))
    other = _tracklet(TRACKLET_ID_CAMERA_FACTOR + 3, range(20), -np.ones(20, int), cam="c1")
    out = refine(pd.concat([df, other]), emb, RefineCfg(max_gap_s=0.0), FPS, GRID)
    c0 = out[out.camera_id == "c0"]
    assert c0.tracklet_id.nunique() == 2
    first, second = c0[c0.tracklet_id == 5], c0[c0.tracklet_id != 5]
    assert first.frame_id.max() < second.frame_id.min()
    assert abs(len(first) - 30) <= 2
    assert 5 < second.tracklet_id.iloc[0] < TRACKLET_ID_CAMERA_FACTOR  # allocated within camera c0's id range
    assert set(out[out.camera_id == "c1"].tracklet_id) == {TRACKLET_ID_CAMERA_FACTOR + 3}
    Tracklets.validate(out)


def test_refine_stitches_short_gap_same_person():
    rng = np.random.default_rng(1)
    v = rng.normal(size=D)
    emb = np.array(_embs(rng, 40, v))
    a = _tracklet(1, range(0, 20), np.arange(20))
    b = _tracklet(2, range(25, 45), np.arange(20, 40))  # same motion, 4 missing frames
    out = refine(pd.concat([a, b]), emb, RefineCfg(interp_gap=0), FPS, GRID)
    assert out.tracklet_id.unique().tolist() == [1]
    assert len(out) == 40


def test_refine_does_not_stitch_different_person_or_overlap_or_far():
    rng = np.random.default_rng(2)
    v, w = rng.normal(size=D), rng.normal(size=D)
    emb = np.array(_embs(rng, 20, v) + _embs(rng, 20, w) + _embs(rng, 20, v))
    a = _tracklet(1, range(0, 20), np.arange(20))
    other_person = _tracklet(2, range(25, 45), np.arange(20, 40))
    overlapping = _tracklet(3, range(10, 30), np.arange(40, 60), x0=100.0)
    far = _tracklet(4, range(25, 45), np.arange(40, 60), x0=900.0, vx=0.0)
    for b in (other_person, overlapping, far):
        out = refine(pd.concat([a, b]), emb, RefineCfg(interp_gap=0), FPS, GRID)
        assert out.tracklet_id.nunique() == 2


def test_refine_drops_short_and_low_score():
    emb = np.zeros((0, D))
    rows = [
        _tracklet(1, range(20), -np.ones(20, int)),
        _tracklet(2, range(5), -np.ones(5, int)),  # too short
        _tracklet(3, range(20), -np.ones(20, int), score=0.2),  # low score
    ]
    out = refine(pd.concat(rows), emb, RefineCfg(), FPS, GRID)
    assert out.tracklet_id.unique().tolist() == [1]


def test_refine_interpolates_gap():
    emb = np.zeros((0, D))
    frames = [*range(0, 10), *range(14, 24), *range(40, 50)]  # gaps of 3 and 16 missing frames
    df = _tracklet(1, frames, -np.ones(len(frames), int))
    df.loc[df.frame_id == 14, "score"] = 0.6
    out = refine(df, emb, RefineCfg(interp_gap=10), FPS, GRID)
    filled = out[out.frame_id.between(10, 13)]
    assert len(filled) == 4 and (filled.emb_idx == -1).all() and (filled.quality == 0).all()
    assert (filled.score == 0.6).all()  # min(0.9, 0.6)
    np.testing.assert_allclose(filled.x, 100.0 + 2.0 * filled.frame_id)
    np.testing.assert_allclose(filled.ts, filled.frame_id / FPS)
    assert not out.frame_id.between(24, 39).any()  # gap longer than interp_gap is left alone
    Tracklets.validate(out)


def test_refine_interpolates_only_onto_processed_frames():
    cam = CameraInfo(camera_id="c0", source="v.mp4", fps=25.0, width=640, height=480, n_frames=200)
    grid = selected_frame_ids(cam, cam.n_frames, 10.0)  # 25 -> 10 fps: steps 3, 2, 3, 2, ...
    assert set(np.diff(grid)) == {2, 3}
    seen = np.delete(grid[:30], [10, 20, 21])  # one and two processed frames missing
    df = _tracklet(1, seen, -np.ones(len(seen), int))
    df["ts"] = cam.ts(df["frame_id"].to_numpy())
    out = refine(df, np.zeros((0, D)), RefineCfg(interp_gap=10), FPS, {"c0": grid})
    assert out.frame_id.tolist() == grid[:30].tolist()
    np.testing.assert_allclose(out.ts, cam.ts(out.frame_id.to_numpy()))
    np.testing.assert_allclose(out.x, 100.0 + 2.0 * out.frame_id)
    full = _tracklet(2, grid[:30], -np.ones(30, int))
    assert len(refine(full, np.zeros((0, D)), RefineCfg(interp_gap=10), FPS, {"c0": grid})) == 30


def test_refine_disabled_returns_input():
    df = _tracklet(1, range(3), -np.ones(3, int))
    assert refine(df, np.zeros((0, D)), RefineCfg(enabled=False), FPS, GRID) is df


def _boxes(rows):
    return pd.DataFrame(rows, columns=["camera_id", "frame_id", "x", "y", "w", "h", "score"])


def test_quality_penalises_occlusion_and_border():
    df = _boxes(
        [
            ("c0", 0, 100, 100, 40, 100, 0.9),  # clean
            ("c0", 1, 100, 100, 40, 100, 0.9),  # occluded by the next row
            ("c0", 1, 110, 100, 40, 100, 0.9),
            ("c0", 2, 0, 100, 40, 100, 0.9),  # touches the left border
        ]
    )
    q = box_quality(df, 640, 480)
    assert q.shape == (4,) and ((q >= 0) & (q <= 1)).all()
    assert q[0] > q[1] and q[0] > q[3]


def test_quality_score_area_and_aspect():
    df = _boxes(
        [
            ("c0", 0, 100, 100, 40, 100, 0.9),
            ("c0", 1, 100, 100, 40, 100, 0.3),  # low score
            ("c0", 2, 100, 100, 10, 25, 0.9),  # tiny
            ("c0", 3, 100, 100, 100, 40, 0.9),  # lying / wide box
        ]
    )
    q = box_quality(df, 640, 480)
    assert q[0] == q.max()
    assert box_quality(df.iloc[:0], 640, 480).shape == (0,)
