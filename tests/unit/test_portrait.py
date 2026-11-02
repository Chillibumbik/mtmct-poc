import numpy as np
import pandas as pd

from mtmct.io import Portraits
from mtmct.portrait.portrait import build_portraits

D = 8


def _rows(tid, cam, n, emb_idx, quality, t0=0.0, world=False):
    f = np.arange(n)
    return pd.DataFrame(
        {
            "camera_id": cam, "frame_id": f, "ts": t0 + f / 10.0, "tracklet_id": tid,
            "x": 0.0, "y": 0.0, "w": 10.0, "h": 20.0, "score": 0.9, "quality": quality,
            "foot_x": 100.0 + f, "foot_y": 200.0, "world_x": f * 0.1 if world else np.nan,
            "world_y": 1.0 if world else np.nan, "emb_idx": emb_idx,
        }
    )  # fmt: skip


def test_portrait_topk_prefers_quality():
    rng = np.random.default_rng(0)
    good, bad = rng.normal(size=D), rng.normal(size=D)
    emb = np.stack([good + rng.normal(0, 0.01, D) for _ in range(3)] + [bad] * 5)
    emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float16)
    quality = np.array([0.9, 0.8, 0.95, 0.1, 0.2, 0.1, 0.3, 0.2])
    df = _rows(7, "c0", 8, np.arange(8), quality)
    portraits, mat, galleries = build_portraits(df, emb, top_k=3)
    Portraits.validate(portraits)
    assert sorted(portraits.gallery_idx.iloc[0]) == [0, 1, 2]
    assert galleries[0].shape == (3, D)
    cos = mat[0] @ (good / np.linalg.norm(good))
    assert cos > 0.99 and np.isclose(np.linalg.norm(mat[0]), 1.0, atol=1e-5)


def test_portrait_fields_and_missing_embeddings():
    emb = np.eye(D, dtype=np.float16)[:4]
    df = pd.concat(
        [
            _rows(2, "c1", 5, np.array([-1, 0, -1, 1, -1]), 0.5, t0=3.0, world=True),
            _rows(1, "c0", 4, -1, 0.4),  # no embedded box
            _rows(3, "c0", 2, np.array([2, 3]), np.array([0.2, 0.6])),
        ],
        ignore_index=True,
    )
    portraits, mat, galleries = build_portraits(df, emb, top_k=10)
    Portraits.validate(portraits)
    assert list(portraits.tracklet_id) == [1, 2, 3] and mat.shape == (3, D)
    p1 = portraits.iloc[0]
    assert p1.n_boxes == 4 and p1.gallery_idx == [] and galleries[0].shape == (0, D)
    assert not mat[0].any()  # zero vector = no appearance information
    assert np.isclose(p1.mean_quality, 0.4)
    assert (p1.entry_x, p1.exit_x) == (100.0, 103.0)  # no world coords → image foot points
    p2 = portraits.iloc[1]
    assert np.isclose(p2.t_start, 3.0) and np.isclose(p2.t_end, 3.4) and p2.n_boxes == 5
    assert np.isclose(p2.entry_x, 0.0) and np.isclose(p2.exit_x, 0.4) and np.isclose(p2.exit_y, 1.0)
    np.testing.assert_allclose(mat[1], np.array([1, 1, 0, 0, 0, 0, 0, 0]) / np.sqrt(2), atol=1e-3)
    assert portraits.gallery_idx.iloc[2] == [3, 2]  # ordered by quality


def test_portrait_empty_input():
    cols = _rows(1, "c0", 1, -1, 0.5).iloc[:0]
    portraits, mat, galleries = build_portraits(cols, np.zeros((0, D), np.float16))
    assert len(portraits) == 0 and mat.shape == (0, D) and galleries == []
