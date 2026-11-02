import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from mtmct import registry
from mtmct.associate.rerank import k_reciprocal_rerank


def _same_partition(a, b) -> bool:
    return (pd.factorize(np.asarray(a))[0] == pd.factorize(np.asarray(b))[0]).all()


def _clustered(n_ids=8, per_id=5, noise=0.25, dim=32, seed=0):
    """Cosine distance matrix of noisy copies of `n_ids` random unit vectors; returns (dist, true labels)."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_ids, dim))
    truth = np.repeat(np.arange(n_ids), per_id)
    x = centres[truth] / np.linalg.norm(centres[truth], axis=1, keepdims=True)
    x = x + rng.normal(0, noise / np.sqrt(dim), x.shape)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    dist = np.clip(1 - x @ x.T, 0, 1)
    np.fill_diagonal(dist, 0)
    return dist, truth


def _solve(name, dist, cl=None, cameras=None, **params):
    cl = np.zeros(dist.shape, bool) if cl is None else cl
    return registry.create("solver", name, params).solve(dist, cl, cameras=cameras)


def test_agglomerative_recovers_known_clusters():
    dist, truth = _clustered()
    labels = _solve("agglomerative", dist, threshold=0.45)
    assert labels.dtype == np.int64 and labels.min() == 0
    assert _same_partition(labels, truth)
    assert list(pd.unique(labels)) == list(range(labels.max() + 1))  # numbered in order of first appearance


@pytest.mark.parametrize("method", ["average", "single", "complete"])
def test_agglomerative_matches_scipy_without_constraints(method):
    rng = np.random.default_rng(3)
    dist = rng.uniform(0, 1, (40, 40))
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    expected = fcluster(linkage(squareform(dist), method=method), t=0.35, criterion="distance")
    assert _same_partition(_solve("agglomerative", dist, linkage=method, threshold=0.35), expected)


def test_agglomerative_respects_cannot_link():
    # Chain A–B–C all close, but A and C cannot link: B joins one of them, never both.
    dist = np.array([[0, 0.1, 0.15], [0.1, 0, 0.12], [0.15, 0.12, 0]])
    cl = np.zeros((3, 3), bool)
    cl[0, 2] = cl[2, 0] = True
    labels = _solve("agglomerative", dist, cl, threshold=0.45)
    assert labels[0] != labels[2] and labels[1] == labels[0]  # A–B is the closest pair

    dist, _ = _clustered(seed=1)
    cl = np.zeros(dist.shape, bool)
    cl[0, 1] = cl[1, 0] = True  # split identity 0
    labels = _solve("agglomerative", dist, cl, threshold=0.45)
    assert labels[0] != labels[1]
    for a in range(len(labels)):  # no cluster contains a cannot-link pair
        assert not cl[a, labels == labels[a]].any()


def test_agglomerative_trivial_sizes():
    assert _solve("agglomerative", np.zeros((0, 0))).shape == (0,)
    assert list(_solve("agglomerative", np.zeros((1, 1)))) == [0]


def test_anchor_assigns_single_camera_tracklets_to_multi_camera_anchors():
    dist = np.full((6, 6), 0.9)
    np.fill_diagonal(dist, 0)
    for i, j, d in [(0, 1, 0.05), (0, 2, 0.3), (1, 2, 0.3), (3, 4, 0.05), (0, 5, 0.35), (1, 5, 0.35)]:
        dist[i, j] = dist[j, i] = d
    cameras = np.array(["c0", "c1", "c2", "c0", "c0", "c1"])
    cl = np.zeros((6, 6), bool)
    cl[2, 5] = cl[5, 2] = True
    labels = _solve("anchor", dist, cl, cameras=cameras, anchor_threshold=0.1, threshold=0.45)
    # anchor {0, 1} spans 2 cameras; 2 joins it; {3, 4} is single-camera → stays; 5 conflicts with 2 → alone
    assert _same_partition(labels, [0, 0, 0, 1, 1, 2])

    alone = _solve("anchor", dist, cl, cameras=np.array(["c0"] * 6), anchor_threshold=0.1, threshold=0.45)
    assert _same_partition(alone, [0, 0, 1, 2, 2, 3])  # no multi-camera anchor → tight clusters only


def test_hungarian_two_cameras():
    dist = np.full((5, 5), 0.9)
    cameras = np.array(["a", "a", "b", "b", "b"])
    for i, j, d in [(0, 3, 0.1), (1, 2, 0.2), (1, 3, 0.15), (0, 4, 0.3)]:
        dist[i, j] = dist[j, i] = d
    np.fill_diagonal(dist, 0)
    labels = _solve("hungarian", dist, cameras=cameras, threshold=0.45)
    # optimal 1:1 matching under the threshold: 0–3 + 1–2 (0.3) beats 1–3 + 0–4 (0.45)
    assert labels[0] == labels[3] and labels[1] == labels[2] and len(set(labels)) == 3

    with pytest.raises(ValueError, match="2 cameras"):
        _solve("hungarian", dist, cameras=np.array(["a", "b", "c", "a", "b"]))
    assert len(set(_solve("hungarian", dist, cameras=np.array(["a"] * 5)))) == 5


def test_rerank_symmetric_shape():
    dist, truth = _clustered(n_ids=6, per_id=6, noise=0.6, seed=4)
    dist[3, :] = dist[:, 3] = np.nan  # a tracklet without appearance stays unknown
    out = k_reciprocal_rerank(dist, k1=20, k2=6, lambda_=0.3)
    assert out.shape == dist.shape
    np.testing.assert_allclose(out, out.T, equal_nan=True)
    assert np.isnan(out[3, 2]) and np.isnan(out[2, 3])
    ok = ~np.isnan(out)
    assert (out[ok] >= 0).all() and (out[ok] <= 1).all() and (np.diag(out)[np.arange(36) != 3] == 0).all()
    same = truth[:, None] == truth[None, :]
    valid = ok & ~np.eye(36, dtype=bool)
    assert out[valid & same].mean() < out[valid & ~same].mean()
    assert k_reciprocal_rerank(np.zeros((1, 1))).shape == (1, 1)  # degenerate size
