from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtmct.eval import identity_metrics, reid_sanity


def _matches(cells: list[tuple[int, ...]], camera_id: str = "c0") -> pd.DataFrame:
    """cells: (person_id, global_id, n_boxes[, tracklet_id]) → that many matched boxes at consecutive timestamps;
    the tracklet defaults to one per (person, global id)."""
    rows = []
    t = 0
    for person, gid, n, *tid in cells:
        for _ in range(n):
            rows.append((person, gid, tid[0] if tid else 100 * person + gid, camera_id, t, t / 10))
            t += 1
    return pd.DataFrame(rows, columns=["person_id", "global_id", "tracklet_id", "camera_id", "frame_id", "ts"])


def _ids(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return m[["person_id"]], m[["global_id"]]


def test_identity_false_merge_counted() -> None:
    m = _matches([(0, 7, 30), (1, 7, 20), (2, 7, 4)])  # person 2 owns < max(5, 10%) → noise, not a merge
    res = identity_metrics(m, *_ids(m))
    assert res["false_merges"] == 1 and res["false_splits"] == 0
    (case,) = res["cases"]
    assert case["kind"] == "false_merge"
    assert case["global_ids"] == [7] and case["gt_person_ids"] == [0, 1]
    assert case["tracklet_ids"] == [7, 107]
    assert case["t"] == pytest.approx(3.0)  # first box of the second person under id 7
    assert res["pairwise_r"] == pytest.approx(1.0) and res["pairwise_p"] < 1.0
    # cluster metrics count tracklets (one per cell here), not boxes: 3 tracklets in one cluster
    assert res["purity"] == pytest.approx(1 / 3) and res["inverse_purity"] == pytest.approx(1.0)


def test_identity_false_split_counted() -> None:
    m = _matches([(0, 1, 40), (0, 2, 10), (0, 3, 3), (1, 4, 10)])
    res = identity_metrics(m, *_ids(m))
    assert res["false_splits"] == 1 and res["false_merges"] == 0
    (case,) = res["cases"]
    assert case["kind"] == "false_split"
    assert case["global_ids"] == [1, 2] and case["gt_person_ids"] == [0]
    assert res["pairwise_p"] == pytest.approx(1.0) and res["pairwise_r"] < 1.0
    assert (res["n_pred_ids"], res["n_gt_ids"]) == (4, 2)


def test_identity_pairwise_and_bcubed_values() -> None:
    # items are tracklets, whatever their length. person 0: tracklets 1 (50 boxes) and 2 under id A, 3 under id B;
    # person 1: tracklet 4 under id B
    m = _matches([(0, 0, 50, 1), (0, 0, 1, 2), (0, 1, 1, 3), (1, 1, 1, 4)])
    res = identity_metrics(m, *_ids(m))
    # tracklet pairs: same pred = {12, 34} = 2, same gt = {12, 13, 23} = 3, both = {12} = 1
    assert res["pairwise_p"] == pytest.approx(1 / 2)
    assert res["pairwise_r"] == pytest.approx(1 / 3)
    assert res["pairwise_f1"] == pytest.approx(2 * (1 / 2) * (1 / 3) / (1 / 2 + 1 / 3))
    # BCubed precision per item: 1, 1, 1/2, 1/2; recall: 2/3, 2/3, 1/3, 1
    assert res["bcubed_p"] == pytest.approx(3 / 4)
    assert res["bcubed_r"] == pytest.approx((2 / 3 + 2 / 3 + 1 / 3 + 1) / 4)


def test_identity_tracklet_labelled_by_majority_person() -> None:
    # tracklet 1: 5 boxes of person 0 and 2 of person 1 → person 0; tracklet 2 (person 1) shares its id
    m = _matches([(0, 0, 5, 1), (1, 0, 2, 1), (1, 0, 3, 2)])
    res = identity_metrics(m, *_ids(m))
    assert res["pairwise_p"] == pytest.approx(0.0) and res["pairwise_r"] == pytest.approx(1.0)
    assert res["purity"] == pytest.approx(1 / 2)


def test_identity_empty() -> None:
    m = _matches([])
    res = identity_metrics(m, *_ids(m))
    assert res["pairwise_f1"] == 0.0 and res["bcubed_f"] == 0.0 and res["handover_success"] is None
    assert res["cases"] == [] and res["transitions"] == []


def test_reid_sanity_rank1_perfect() -> None:
    emb = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    res = reid_sanity(emb, np.array([0, 0, 1, 1]), np.array(["a", "b", "a", "b"]))
    assert res == {"rank1": 1.0, "rank5": 1.0, "mAP": 1.0, "n_queries": 4}


def test_reid_sanity_values() -> None:
    # query 0 (cam a, person 0): gallery = rows 1, 2 (cam b); row 2 (person 1) is closer → rank1 miss, AP = 1/2
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.5, 0.5]])
    res = reid_sanity(emb, np.array([0, 0, 1, 2]), np.array(["a", "b", "b", "a"]))
    # queries with a positive in other cameras: 0 and 1 (persons 1, 2 have none)
    assert res["n_queries"] == 2
    assert res["rank1"] == pytest.approx(0.0)
    assert res["rank5"] == pytest.approx(1.0)
    assert res["mAP"] == pytest.approx(0.5)


def test_reid_sanity_no_queries() -> None:
    res = reid_sanity(np.zeros((0, 4)), np.array([], dtype=int), np.array([], dtype=str))
    assert res == {"rank1": None, "rank5": None, "mAP": None, "n_queries": 0}
