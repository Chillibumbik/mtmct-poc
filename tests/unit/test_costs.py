import numpy as np
import pandas as pd
import pytest

from mtmct import registry
from mtmct.associate.combine import combine, same_camera_overlap
from mtmct.associate.topology import fit_topology
from mtmct.components import AssocContext, CostMatrix
from mtmct.data.base import CameraInfo, SequenceInfo, SequenceMeta
from mtmct.portrait.portrait import build_portraits

FPS = 10.0
D = 16
EYE = np.eye(3).tolist()


def _seq(n_cams=2, overlap="overlap", homography=True) -> SequenceInfo:
    cams = [
        CameraInfo(
            camera_id=f"c{i}", source="v", fps=FPS, width=640, height=480, n_frames=1000,
            homography_img2world=EYE if homography else None,
        )
        for i in range(n_cams)
    ]  # fmt: skip
    return SequenceInfo(
        dataset="t", sequence="s", root=".", cameras=cams, meta=SequenceMeta(overlap=overlap, duration_s=100)
    )


def _tracklet(tid, cam, t0, n, xy=(0.0, 0.0), v=(1.0, 0.0), emb_idx=-1, world=True):
    t = t0 + np.arange(n) / FPS
    wx, wy = xy[0] + v[0] * (t - t0), xy[1] + v[1] * (t - t0)
    return pd.DataFrame(
        {
            "camera_id": cam, "frame_id": np.round(t * FPS).astype(int), "ts": t, "tracklet_id": tid,
            "x": 0.0, "y": 0.0, "w": 10.0, "h": 20.0, "score": 0.9, "quality": 0.5,
            "foot_x": wx, "foot_y": wy, "world_x": wx if world else np.nan, "world_y": wy if world else np.nan,
            "emb_idx": emb_idx,
        }
    )  # fmt: skip


def _ctx(parts, emb=None, seq=None, topology=None, warn=None) -> AssocContext:
    tr = pd.concat(parts, ignore_index=True)
    emb = np.zeros((1, D), np.float16) if emb is None else emb
    portraits, mat, gal = build_portraits(tr, emb)
    return AssocContext(
        seq=seq or _seq(), tracklets=tr, portraits=portraits, portrait_emb=mat, gallery_emb=gal,
        topology=topology, warn=warn or (lambda m: None),
    )  # fmt: skip


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------- appearance ----------


def test_appearance_mean_matches_cosine_and_nan_for_missing():
    rng = np.random.default_rng(0)
    emb = _unit(rng.normal(size=(4, D))).astype(np.float16)
    ctx = _ctx(
        [
            _tracklet(1, "c0", 0, 2, emb_idx=np.array([0, 1])),
            _tracklet(2, "c1", 0, 2, emb_idx=np.array([2, 3])),
            _tracklet(3, "c1", 5, 2),  # no embeddings
        ],
        emb=emb,
    )
    cm = registry.create("cost", "appearance", {"agg": "mean"})(ctx)
    p = ctx.portrait_emb
    assert np.isclose(cm.dist[0, 1], np.clip(1 - p[0] @ p[1], 0, 1), atol=1e-5)
    assert np.isnan(cm.dist[2]).all() and np.isnan(cm.dist[:, 2]).all()
    assert not cm.cannot_link.any()


@pytest.mark.parametrize("agg", ["gallery_median", "gallery_min"])
def test_appearance_gallery_aggregates(agg):
    rng = np.random.default_rng(1)
    emb = _unit(rng.normal(size=(5, D)))
    ctx = _ctx(
        [
            _tracklet(1, "c0", 0, 2, emb_idx=np.array([0, 1])),
            _tracklet(2, "c1", 0, 3, emb_idx=np.array([2, 3, 4])),
            _tracklet(3, "c1", 5, 1),
        ],
        emb=emb.astype(np.float16),
    )
    cm = registry.create("cost", "appearance", {"agg": agg})(ctx)
    pair = np.clip(1 - ctx.gallery_emb[0] @ ctx.gallery_emb[1].T, 0, 1)
    expected = np.median(pair) if agg == "gallery_median" else pair.min()
    assert np.isclose(cm.dist[0, 1], expected, atol=1e-5) and np.isclose(cm.dist[1, 0], expected, atol=1e-5)
    assert np.isnan(cm.dist[2]).all()


# ---------- geometry ----------


def test_geometry_cost_close_people_low_far_cannot_link():
    ctx = _ctx(
        [
            _tracklet(1, "c0", 0.0, 21, xy=(0, 0)),  # 0..2 s, walks along x
            _tracklet(2, "c1", 0.5, 21, xy=(0.5, 0.3)),  # same person seen by c1, 0.3 m off
            _tracklet(3, "c1", 0.0, 21, xy=(0, 6.0)),  # someone 6 m away at the same time
            _tracklet(4, "c1", 3.0, 10, xy=(2.5, 0.0)),  # appears 1 s after #1 left, 0.5 m away: plausible
            _tracklet(5, "c0", 2.5, 10, xy=(20.0, 0.0)),  # 18 m in 0.5 s: impossible
        ]
    )
    cm = registry.create("cost", "geometry", {"gate_m": 1.5, "max_speed_mps": 3.0})(ctx)
    assert np.allclose(cm.dist, cm.dist.T, equal_nan=True) and (cm.cannot_link == cm.cannot_link.T).all()
    assert cm.dist[0, 1] == pytest.approx(0.3 / 1.5, abs=1e-6) and not cm.cannot_link[0, 1]
    assert cm.dist[0, 2] == 1.0 and cm.cannot_link[0, 2]
    assert np.isnan(cm.dist[0, 3]) and not cm.cannot_link[0, 3]
    assert np.isnan(cm.dist[0, 4]) and cm.cannot_link[0, 4]


def test_geometry_cost_missing_homography():
    warnings: list[str] = []
    parts = [_tracklet(1, "c0", 0, 5, world=False), _tracklet(2, "c1", 0, 5, world=False)]
    ctx = _ctx(parts, seq=_seq(homography=False), warn=warnings.append)
    cm = registry.create("cost", "geometry")(ctx)
    assert len(warnings) == 1 and "homograph" in warnings[0]
    assert np.isnan(cm.dist).all() and not cm.cannot_link.any()


# ---------- time ----------

TOPO = {"pairs": {"c0->c1": [4.0, 8.0]}}


def test_time_cost_outside_window_cannot_link():
    parts = [
        _tracklet(1, "c0", 0.0, 11),  # ends at t=1
        _tracklet(2, "c1", 6.0, 5),  # Δt = 5 → inside
        _tracklet(3, "c1", 20.0, 5),  # Δt = 19 → outside
        _tracklet(4, "c0", 30.0, 5),  # c1 → c0 never observed → unseen penalty
        _tracklet(5, "c1", 0.5, 5),  # co-visible with #1 in a non-overlapping layout
    ]
    seq = _seq(overlap="nonoverlap", homography=False)
    cm = registry.create("cost", "time", {"hard": True, "unseen_penalty": 0.5})(_ctx(parts, seq=seq, topology=TOPO))
    assert (cm.cannot_link == cm.cannot_link.T).all() and np.allclose(cm.dist, cm.dist.T, equal_nan=True)
    assert cm.dist[0, 1] == 0.0 and not cm.cannot_link[0, 1]
    assert cm.cannot_link[0, 2]
    assert cm.dist[1, 3] == 0.5 and not cm.cannot_link[1, 3]
    assert cm.cannot_link[0, 4]
    assert np.isnan(cm.dist[0, 3]) and not cm.cannot_link[0, 3]  # same camera: not this cost's business

    soft = registry.create("cost", "time", {"hard": False})(_ctx(parts, seq=seq, topology=TOPO))
    assert soft.dist[0, 2] == 1.0 and not soft.cannot_link[0, 2]
    assert soft.cannot_link[0, 4]  # co-visibility in a non-overlapping layout stays forbidden


def test_time_cost_missing_topology_warns():
    warnings: list[str] = []
    ctx = _ctx([_tracklet(1, "c0", 0, 3), _tracklet(2, "c1", 5, 3)], warn=warnings.append)
    cm = registry.create("cost", "time")(ctx)
    assert len(warnings) == 1 and np.isnan(cm.dist).all() and not cm.cannot_link.any()


def test_fit_topology_quantiles():
    seq = _seq(n_cams=3, homography=False)
    rows = []
    rng = np.random.default_rng(0)
    gaps = rng.uniform(5.0, 15.0, size=200)
    for pid, gap in enumerate(gaps):
        t0 = pid * 100.0
        for cam, a, b in [("c0", t0, t0 + 3), ("c1", t0 + 3 + gap, t0 + 6 + gap)]:
            for t in np.arange(a, b, 1 / FPS):
                rows.append({"camera_id": cam, "frame_id": round(t * FPS), "person_id": pid})
    gt = pd.DataFrame(rows).assign(x=0.0, y=0.0, w=1.0, h=1.0, world_x=np.nan, world_y=np.nan)
    topo = fit_topology(gt, seq, q=(0.05, 0.95), margin_s=2.0)
    assert set(topo["pairs"]) == {"c0->c1"}
    lo, hi = topo["pairs"]["c0->c1"]
    exp_lo, exp_hi = np.quantile(gaps, [0.05, 0.95])
    assert lo == pytest.approx(exp_lo - 2.0, abs=0.15) and hi == pytest.approx(exp_hi + 2.0, abs=0.15)


# ---------- combine ----------


def test_combine_ignores_nan():
    nan = np.nan
    a = CostMatrix(np.array([[0, 0.2, nan], [0.2, 0, nan], [nan, nan, 0]]), np.zeros((3, 3), bool))
    cl = np.zeros((3, 3), bool)
    cl[0, 2] = cl[2, 0] = True
    b = CostMatrix(np.array([[0, 0.6, nan], [0.6, 0, 0.4], [nan, 0.4, 0]]), cl)
    same = np.zeros((3, 3), bool)
    same[1, 2] = same[2, 1] = True
    dist, cannot = combine([(a, 1.0), (b, 3.0)], same)
    assert dist[0, 1] == pytest.approx((0.2 + 3 * 0.6) / 4)
    assert dist[1, 2] == pytest.approx(0.4)  # only b has information
    assert dist[0, 2] == 1.0  # nobody has information
    assert (np.diag(dist) == 0).all() and not np.diag(cannot).any()
    assert cannot[0, 2] and cannot[1, 2] and not cannot[0, 1]


def test_same_camera_overlap():
    ctx = _ctx(
        [_tracklet(1, "c0", 0, 10), _tracklet(2, "c0", 0.5, 10), _tracklet(3, "c0", 5, 3), _tracklet(4, "c1", 0, 9)]
    )
    m = same_camera_overlap(ctx.portraits)
    assert m[0, 1] and m[1, 0] and not m[0, 2] and not m[0, 3] and not np.diag(m).any()
