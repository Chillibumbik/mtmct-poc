import numpy as np
import pandas as pd

from mtmct.associate.run import associate, build_context, threshold_labels
from mtmct.config import AssociateCfg, CostSpec
from mtmct.data.base import CameraInfo, SequenceInfo, SequenceMeta
from mtmct.geometry.homography import apply_h, to_world
from mtmct.portrait.portrait import build_portraits

FPS = 10.0
D = 32
H = [np.diag([0.01, 0.01, 1.0]), np.array([[0.0, 0.02, -3.0], [0.02, 0.0, -1.0], [0.0, 0.0, 1.0]])]


def _seq() -> SequenceInfo:
    cams = [
        CameraInfo(camera_id=f"c{i}", source="v", fps=FPS, width=640, height=480, n_frames=600,
                   homography_img2world=H[i].tolist())
        for i in range(2)
    ]  # fmt: skip
    return SequenceInfo(
        dataset="t", sequence="s", root=".", cameras=cams, meta=SequenceMeta(overlap="overlap", duration_s=60)
    )


def _scene(seed=0):
    """4 people walking on parallel lines 2 m apart, all seen by both cameras; person 0 is split in c0."""
    rng = np.random.default_rng(seed)
    looks = rng.normal(size=(4, D))
    parts, embs = [], []
    spans = {0: [(0.0, 3.0), (3.5, 6.0)], 1: [(0.2, 6.0)], 2: [(1.0, 5.0)], 3: [(2.0, 6.0)]}
    tid = 0
    for cam in range(2):
        for person in range(4):
            for a, b in spans[person] if cam == 0 else [(spans[person][0][0] + 0.3, spans[person][-1][1])]:
                t = np.arange(round(a * FPS), round(b * FPS)) / FPS
                world = np.column_stack([0.8 * t, np.full(len(t), 2.0 * person)])
                foot = apply_h(np.linalg.inv(H[cam]), world)
                e = looks[person] + rng.normal(0, 0.5, (len(t), D))
                e /= np.linalg.norm(e, axis=1, keepdims=True)
                parts.append(
                    pd.DataFrame(
                        {
                            "camera_id": f"c{cam}", "frame_id": np.round(t * FPS).astype(int), "ts": t,
                            "tracklet_id": cam * 10_000_000 + tid, "x": foot[:, 0] - 10, "y": foot[:, 1] - 40,
                            "w": 20.0, "h": 40.0, "score": 0.9, "quality": rng.uniform(0.2, 1.0, len(t)),
                            "foot_x": foot[:, 0], "foot_y": foot[:, 1], "world_x": np.nan, "world_y": np.nan,
                            "emb_idx": sum(len(x) for x in embs) + np.arange(len(t)),
                        }
                    )
                )  # fmt: skip
                embs.append(e)
                tid += 1
    return pd.concat(parts, ignore_index=True), np.concatenate(embs).astype(np.float16)


def _ctx(tracklets, emb, warn=None):
    seq = _seq()
    tracklets = to_world(tracklets, seq)
    portraits, mat, gal = build_portraits(tracklets, emb, top_k=10)
    return build_context(seq, tracklets, portraits, mat, gal, topology=None, warn=warn or (lambda m: None))


def _person_of(tracklets: pd.DataFrame) -> pd.Series:
    """True person per tracklet, recovered from the constant world y of the synthetic scene."""
    return tracklets.groupby("tracklet_id")["world_y"].mean().round().div(2).astype(int)


OVERLAP = AssociateCfg(
    costs={"appearance": CostSpec(weight=1.0), "geometry": CostSpec(weight=1.0)},
    solver={"name": "agglomerative", "params": {"threshold": 0.45}},
)


def _same_partition(a: pd.Series, b: pd.Series) -> bool:
    a, b = a.sort_index(), b.sort_index()
    return (a.index == b.index).all() and (pd.factorize(a)[0] == pd.factorize(b)[0]).all()


def test_associate_recovers_people_with_geometry():
    tracklets, emb = _scene()
    ctx = _ctx(tracklets, emb)
    labels = associate(ctx, OVERLAP)
    assert labels.dtype == np.int64 and labels.index.name == "tracklet_id"
    assert _same_partition(labels, _person_of(ctx.tracklets))


def test_associate_single_tracklet():
    tracklets, emb = _scene()
    one = tracklets[tracklets.tracklet_id == tracklets.tracklet_id.iloc[0]]
    labels = associate(_ctx(one, emb), OVERLAP)
    assert labels.to_dict() == {one.tracklet_id.iloc[0]: 0}
    empty = associate(_ctx(one.iloc[:0], emb), OVERLAP)
    assert len(empty) == 0 and empty.dtype == np.int64


def test_associate_permutation_invariant():
    tracklets, emb = _scene(seed=1)
    ctx = _ctx(tracklets, emb)
    cfg = OVERLAP.model_copy(update={"rerank": {"k1": 5, "k2": 2, "lambda_": 0.3}})
    base = associate(ctx, cfg)
    perm = np.random.default_rng(0).permutation(len(ctx.portraits))
    shuffled = build_context(
        ctx.seq, ctx.tracklets.sample(frac=1, random_state=1), ctx.portraits.iloc[perm].reset_index(drop=True),
        ctx.portrait_emb[perm], [ctx.gallery_emb[i] for i in perm], None, lambda m: None,
    )  # fmt: skip
    assert _same_partition(associate(shuffled, cfg), base)


def test_threshold_labels_monotone_and_matches_associate():
    tracklets, emb = _scene()
    ctx = _ctx(tracklets, emb)
    curve = threshold_labels(ctx, OVERLAP, [0.0, 0.45, 1.0])
    assert list(curve) == [0.0, 0.45, 1.0]
    n_ids = [s.nunique() for s in curve.values()]
    assert n_ids[0] == len(ctx.portraits) and n_ids[0] >= n_ids[1] >= n_ids[2] >= 1
    assert _same_partition(curve[0.45], associate(ctx, OVERLAP))


def test_associate_warns_when_geometry_impossible():
    tracklets, emb = _scene()
    seq = _seq().model_copy(
        update={"cameras": [c.model_copy(update={"homography_img2world": None}) for c in _seq().cameras]}
    )
    tracklets = to_world(tracklets, seq)
    portraits, mat, gal = build_portraits(tracklets, emb)
    warnings: list[str] = []
    ctx = build_context(seq, tracklets, portraits, mat, gal, None, warnings.append)
    labels = associate(ctx, OVERLAP)
    assert len(warnings) == 1 and len(labels) == len(portraits)
