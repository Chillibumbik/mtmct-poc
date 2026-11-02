from __future__ import annotations

import json
import math
import time
from pathlib import Path

import motmetrics as mm
import numpy as np
import pandas as pd
import pytest

from mtmct.data.base import CameraInfo, SequenceInfo, SequenceMeta
from mtmct.eval import concat_cameras, evaluate_run, iou_matrix, match_boxes


def _seq(n_cams: int = 2, n_frames: int = 100, fps: float = 10.0) -> SequenceInfo:
    cams = [
        CameraInfo(camera_id=f"c{i}", source=f"c{i}.avi", fps=fps, width=4000, height=1000, n_frames=n_frames)
        for i in range(n_cams)
    ]
    meta = SequenceMeta(overlap="overlap", duration_s=n_frames / fps)
    return SequenceInfo(dataset="toy", sequence="s", root=Path("/nonexistent"), cameras=cams, meta=meta)


def _gt(rows: list[tuple[str, int, int, float, float]]) -> pd.DataFrame:
    """rows: (camera_id, frame_id, person_id, x, y); boxes are 50x50."""
    df = pd.DataFrame(rows, columns=["camera_id", "frame_id", "person_id", "x", "y"])
    return df.assign(w=50.0, h=50.0, world_x=np.nan, world_y=np.nan)


def _grid_gt(cams: list[str], frames: range, persons: range) -> pd.DataFrame:
    return _gt([(c, f, p, 100.0 * p, 0.0) for c in cams for f in frames for p in persons])


def _pred_from_gt(gt: pd.DataFrame, gid: dict[tuple[str, int], int] | None = None) -> pd.DataFrame:
    """Prediction with the GT boxes; tracklet = (camera, person); global id from `gid` (default: person id)."""
    keys = list(zip(gt["camera_id"], gt["person_id"], strict=True))
    cam_idx = gt["camera_id"].str[1:].astype(int).to_numpy()
    return pd.DataFrame(
        {
            "camera_id": gt["camera_id"].to_numpy(),
            "global_id": [gid[k] if gid else k[1] for k in keys],
            "tracklet_id": cam_idx * 10_000_000 + gt["person_id"].to_numpy(),
            "frame_id": gt["frame_id"].to_numpy(),
            "x": gt["x"].to_numpy(),
            "y": gt["y"].to_numpy(),
            "w": gt["w"].to_numpy(),
            "h": gt["h"].to_numpy(),
            "world_x": np.nan,
            "world_y": np.nan,
        }
    )


def _all_frames(seq: SequenceInfo) -> dict[str, np.ndarray]:
    return {c.camera_id: np.arange(c.n_frames) for c in seq.cameras}


def _assert_json_safe(res: dict) -> None:
    json.dumps(res, allow_nan=False)

    def walk(v: object) -> None:
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        else:
            assert v is None or isinstance(v, str | int | float), type(v)
            if isinstance(v, float):
                assert math.isfinite(v)

    walk(res)


def test_iou_matrix() -> None:
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [100, 100, 5, 5]], dtype=float)
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])
    assert iou_matrix(a[:0], b).shape == (0, 3)


def test_concat_cameras() -> None:
    seq = _seq(n_cams=3, n_frames=100)
    seq.cameras[1].n_frames = 120
    df = pd.DataFrame({"camera_id": ["c0", "c1", "c2"], "frame_id": [5, 5, 7]})
    out = concat_cameras(df, seq)
    assert out["frame_id"].tolist() == [5, 121 + 5, 2 * 121 + 7]
    assert df["frame_id"].tolist() == [5, 5, 7]  # input untouched


def test_match_boxes_hungarian_and_threshold() -> None:
    seq = _seq(n_cams=1)
    gt = _gt([("c0", 3, 1, 0.0, 0.0), ("c0", 3, 2, 30.0, 0.0), ("c0", 4, 1, 0.0, 0.0)])
    pred = _pred_from_gt(gt, {("c0", 1): 7, ("c0", 2): 8})
    pred.loc[1, "x"] = 10.0  # person 2 box shifted onto person 1 area: IoU(gt1)=0.67, IoU(gt2)=0.43
    pred.loc[2, "x"] = 40.0  # frame 4: IoU 10/90 < 0.5 → unmatched
    m = match_boxes(gt, pred, 0.5, seq)
    assert len(m) == 1
    row = m.iloc[0]
    assert (row["gt_row"], row["pred_row"], row["person_id"], row["global_id"]) == (0, 0, 1, 7)
    assert row["ts"] == pytest.approx(0.3)
    assert {"tracklet_id", "camera_id", "frame_id"} <= set(m.columns)


def test_eval_perfect_prediction_all_ones() -> None:
    seq = _seq()
    gt = _grid_gt(["c0", "c1"], range(0, 50), range(3))
    res = evaluate_run(gt, _pred_from_gt(gt), seq, _all_frames(seq))
    assert set(res) == {"per_camera", "per_camera_mean", "cross_camera", "identity", "reid", "threshold_curve"}
    for m in [*res["per_camera"].values(), res["per_camera_mean"], res["cross_camera"]]:
        for k in ("HOTA", "DetA", "AssA", "LocA", "IDF1", "IDP", "IDR", "MOTA", "MOTP"):
            assert m[k] == pytest.approx(1.0), k
        assert (m["IDSW"], m["FP"], m["FN"], m["Frag"]) == (0, 0, 0, 0)
    ident = res["identity"]
    for k in ("pairwise_p", "pairwise_r", "pairwise_f1", "bcubed_p", "bcubed_r", "bcubed_f", "purity"):
        assert ident[k] == pytest.approx(1.0), k
    assert ident["inverse_purity"] == pytest.approx(1.0)
    assert (ident["n_pred_ids"], ident["n_gt_ids"], ident["false_merges"], ident["false_splits"]) == (3, 3, 0, 0)
    assert ident["handover_success"] == pytest.approx(1.0)
    assert ident["cases"] == []
    assert res["reid"] == {"rank1": None, "rank5": None, "mAP": None, "n_queries": 0}
    assert res["threshold_curve"] == []
    _assert_json_safe(res)


def test_eval_cross_camera_swap_detected() -> None:
    seq = _seq()
    gt = _grid_gt(["c0", "c1"], range(0, 50), range(3))
    gid = {("c0", p): p for p in range(3)} | {("c1", p): (p + 1) % 3 for p in range(3)}
    res = evaluate_run(gt, _pred_from_gt(gt, gid), seq, _all_frames(seq))
    for cam in ("c0", "c1"):
        assert res["per_camera"][cam]["IDF1"] == pytest.approx(1.0)
        assert res["per_camera"][cam]["HOTA"] == pytest.approx(1.0)
    assert res["cross_camera"]["IDF1"] == pytest.approx(0.5)
    assert res["cross_camera"]["AssA"] < 1.0
    assert res["identity"]["pairwise_f1"] < 1.0
    assert res["identity"]["handover_success"] == pytest.approx(0.0)
    assert res["identity"]["false_merges"] == 3 and res["identity"]["false_splits"] == 3


def test_handover_success_rate() -> None:
    """Person 0 walks c0 → c1 → c2 → c3 (disjoint in time); prediction keeps the id twice of three times."""
    seq = _seq(n_cams=4)
    path = [("c0", range(0, 10), 7), ("c1", range(15, 25), 7), ("c2", range(30, 40), 7), ("c3", range(45, 55), 9)]
    gt = _gt([(c, f, 0, 0.0, 0.0) for c, frames, _ in path for f in frames])
    pred = _pred_from_gt(gt)
    pred["global_id"] = [g for _, frames, g in path for _ in frames]
    res = evaluate_run(gt, pred, seq, _all_frames(seq))
    ident = res["identity"]
    assert ident["handover_success"] == pytest.approx(2 / 3)
    trans = {(t["from_camera"], t["to_camera"]): (t["total"], t["kept"]) for t in ident["transitions"]}
    assert trans == {("c0", "c1"): (1, 1), ("c1", "c2"): (1, 1), ("c2", "c3"): (1, 0)}


def test_handover_overlapping_cameras_not_inflated() -> None:
    """Person seen simultaneously in c1 and c0 (c1 first), returning to c1 later: one transition c1 → c0, judged by
    each camera's majority id (c0 mostly 7, c1 mostly 7 despite a few boxes of id 8)."""
    seq = _seq(n_cams=2)
    gt = _gt(
        [(c, f, 0, 0.0, 0.0) for f in range(5, 40) for c in ("c0", "c1")] + [("c1", f, 0, 0.0, 0.0) for f in range(4)]
    )
    pred = _pred_from_gt(gt)
    pred["global_id"] = np.where((pred["camera_id"] == "c1") & (pred["frame_id"] < 10), 8, 7)
    ident = evaluate_run(gt, pred, seq, _all_frames(seq))["identity"]
    assert ident["transitions"] == [{"from_camera": "c1", "to_camera": "c0", "total": 1, "kept": 1}]
    assert ident["handover_success"] == pytest.approx(1.0)


def _random_case(seed: int) -> tuple[SequenceInfo, pd.DataFrame, pd.DataFrame]:
    """Random walks with overlaps, missed GT boxes, false positives, id switches and per-camera id permutations.

    Ids are unique per frame: persons 0..5 permuted, switched ids 2000+, false positives 1001..1999.
    """
    rng = np.random.default_rng(seed)
    seq = _seq(n_cams=3, n_frames=40)
    gt_rows, pred_rows = [], []
    fresh = 1000
    for ci, cam in enumerate(["c0", "c1", "c2"]):
        perm = rng.permutation(6)
        for p in range(5):
            xy = rng.uniform(0, 300, 2)
            switch = int(rng.integers(0, 60))
            for f in range(40):
                xy = xy + rng.normal(0, 8, 2)
                if rng.random() < 0.85:
                    gt_rows.append((cam, f, p, xy[0], xy[1]))
                if rng.random() < 0.8:
                    gid = int(perm[p]) if f < switch else 2000 + ci * 10 + p
                    noise = rng.normal(0, 6, 2)
                    pred_rows.append((cam, gid, ci * 100 + p, f, xy[0] + noise[0], xy[1] + noise[1]))
        for f in range(40):
            if rng.random() < 0.3:
                fresh += 1
                pred_rows.append((cam, fresh, fresh, f, *rng.uniform(0, 300, 2)))
    gt = _gt(gt_rows)
    pred = pd.DataFrame(pred_rows, columns=["camera_id", "global_id", "tracklet_id", "frame_id", "x", "y"])
    pred = pred.assign(w=50.0, h=50.0, world_x=np.nan, world_y=np.nan)
    return seq, gt, pred


def _motmetrics_idf1(gt: pd.DataFrame, pred: pd.DataFrame, frames: np.ndarray) -> float:
    acc = mm.MOTAccumulator(auto_id=True)
    for f in frames:
        g = gt[gt["frame_id"] == f]
        p = pred[pred["frame_id"] == f]
        iou = iou_matrix(g[["x", "y", "w", "h"]].to_numpy(float), p[["x", "y", "w", "h"]].to_numpy(float))
        dist = np.where(iou >= 0.5, 1.0 - iou, np.nan)
        acc.update(g["person_id"].tolist(), p["global_id"].tolist(), dist)
    return float(mm.metrics.create().compute(acc, metrics=["idf1"], name="x")["idf1"].iloc[0])


@pytest.mark.parametrize("seed", range(5))
def test_idf1_matches_motmetrics(seed: int) -> None:
    seq, gt, pred = _random_case(seed)
    frames = _all_frames(seq)
    res = evaluate_run(gt, pred, seq, frames)
    for cam in ("c0", "c1", "c2"):
        ref = _motmetrics_idf1(gt[gt["camera_id"] == cam], pred[pred["camera_id"] == cam], frames[cam])
        assert abs(res["per_camera"][cam]["IDF1"] - ref) < 1e-6
    cg, cp = concat_cameras(gt, seq), concat_cameras(pred, seq)
    ref = _motmetrics_idf1(cg, cp, np.unique(np.concatenate([cg["frame_id"], cp["frame_id"]])))
    assert abs(res["cross_camera"]["IDF1"] - ref) < 1e-6
    assert 0.0 < ref < 1.0


@pytest.mark.parametrize("seed", range(3))
def test_threshold_curve_matches_full_evaluation(seed: int) -> None:
    """Relabelling by tracklet must reproduce the full evaluation when the labels equal the prediction's."""
    seq, gt, pred = _random_case(seed)
    pred = pred[pred["global_id"] < 1000]  # tracklet → global id is a function only without switches and FPs
    labels = pred.drop_duplicates("tracklet_id").set_index("tracklet_id")["global_id"]
    merged = pd.Series(0, index=labels.index)
    res = evaluate_run(gt, pred, seq, _all_frames(seq), threshold_labels={0.7: merged, 0.3: labels})
    curve = res["threshold_curve"]
    assert [c["threshold"] for c in curve] == [0.3, 0.7]
    exact = curve[0]
    assert exact["IDF1"] == pytest.approx(res["cross_camera"]["IDF1"], abs=1e-12)
    assert exact["IDP"] == pytest.approx(res["cross_camera"]["IDP"], abs=1e-12)
    assert exact["IDR"] == pytest.approx(res["cross_camera"]["IDR"], abs=1e-12)
    for k in ("pairwise_p", "pairwise_r", "pairwise_f1"):
        assert exact[k] == pytest.approx(res["identity"][k], abs=1e-12)
    assert exact["n_pred_ids"] == res["identity"]["n_pred_ids"]
    assert curve[1]["pairwise_r"] == pytest.approx(1.0)
    assert curve[1]["n_pred_ids"] == 1
    _assert_json_safe(res)


def test_threshold_curve_unlabelled_tracklets_are_singletons() -> None:
    seq = _seq()
    gt = _grid_gt(["c0", "c1"], range(0, 20), range(2))
    pred = _pred_from_gt(gt)
    only_c0 = pd.Series({0: 0, 1: 1})  # tracklets of c1 are missing → each becomes its own id
    res = evaluate_run(gt, pred, seq, _all_frames(seq), threshold_labels={0.5: only_c0})
    pt = res["threshold_curve"][0]
    assert pt["n_pred_ids"] == 4
    assert pt["pairwise_p"] == pytest.approx(1.0)
    assert pt["IDF1"] == pytest.approx(0.5)


def test_eval_empty_prediction() -> None:
    seq = _seq()
    gt = _grid_gt(["c0", "c1"], range(0, 20), range(2))
    empty = _pred_from_gt(gt).iloc[:0]
    res = evaluate_run(gt, empty, seq, _all_frames(seq))
    _assert_json_safe(res)
    for m in [*res["per_camera"].values(), res["cross_camera"]]:
        assert (m["HOTA"], m["IDF1"], m["MOTA"], m["FP"]) == (0.0, 0.0, 0.0, 0)
        assert m["FN"] > 0
    ident = res["identity"]
    assert (ident["pairwise_f1"], ident["bcubed_f"], ident["purity"], ident["n_pred_ids"]) == (0.0, 0.0, 0.0, 0)
    assert ident["n_gt_ids"] == 2
    assert ident["handover_success"] is None


def test_eval_empty_gt_and_single_camera() -> None:
    seq = _seq(n_cams=1)
    gt = _grid_gt(["c0"], range(0, 20), range(2))
    res = evaluate_run(gt.iloc[:0], _pred_from_gt(gt), seq, _all_frames(seq))
    _assert_json_safe(res)
    assert res["cross_camera"]["FP"] == 40 and res["cross_camera"]["IDF1"] == 0.0
    assert res["identity"]["n_gt_ids"] == 0
    res = evaluate_run(gt, _pred_from_gt(gt), seq, _all_frames(seq))
    _assert_json_safe(res)
    assert res["cross_camera"]["IDF1"] == pytest.approx(1.0)
    assert res["identity"]["transitions"] == [] and res["identity"]["handover_success"] is None
    res = evaluate_run(gt.iloc[:0], _pred_from_gt(gt).iloc[:0], seq, {})
    _assert_json_safe(res)
    assert res["per_camera"] == {} and res["per_camera_mean"] == {}


def test_eval_frame_intersection() -> None:
    """GT annotated at 2 fps (every 5th frame), prediction at 10 fps: only shared frames are scored."""
    seq = _seq(n_cams=2, n_frames=100)
    gt = _grid_gt(["c0", "c1"], range(0, 100, 5), range(2))
    dense = _grid_gt(["c0", "c1"], range(0, 80), range(2))  # processed frames 0..79 only
    pred = _pred_from_gt(dense)
    pred.loc[pred["frame_id"] % 5 != 0, "x"] += 1000.0  # boxes off the annotated frames are garbage
    processed = np.arange(80)
    frames = {c: np.intersect1d(processed, np.arange(0, 100, 5)) for c in ("c0", "c1")}
    res = evaluate_run(gt, pred, seq, frames)
    for m in [*res["per_camera"].values(), res["cross_camera"]]:
        assert (m["FP"], m["FN"]) == (0, 0)
        assert m["MOTA"] == pytest.approx(1.0) and m["IDF1"] == pytest.approx(1.0)
    assert res["identity"]["n_gt_ids"] == 2


def test_eval_duplicate_global_id_in_frame_does_not_crash() -> None:
    seq = _seq()
    gt = _grid_gt(["c0", "c1"], range(0, 20), range(2))
    pred = _pred_from_gt(gt)
    pred["global_id"] = 0  # both people share one id in every frame of both cameras
    res = evaluate_run(gt, pred, seq, _all_frames(seq))
    _assert_json_safe(res)
    assert res["identity"]["false_merges"] == 1


def test_reid_sanity_via_evaluate_run() -> None:
    seq = _seq()
    gt = _grid_gt(["c0", "c1"], range(0, 20), range(3))
    pred = _pred_from_gt(gt)
    tids = sorted(pred["tracklet_id"].unique())
    portraits = pd.DataFrame({"tracklet_id": [*tids, 999], "camera_id": ["c0"] * 3 + ["c1"] * 3 + ["c1"]})
    person = [t % 10_000_000 for t in tids]
    emb = np.eye(4)[[*person, 3]]  # tracklet 999 has no GT match → excluded
    res = evaluate_run(gt, pred, seq, _all_frames(seq), portraits=portraits, portrait_emb=emb)
    assert res["reid"] == {"rank1": 1.0, "rank5": 1.0, "mAP": 1.0, "n_queries": 6}


def test_eval_cases_for_ui() -> None:
    seq = _seq(n_cams=1)
    gt = _grid_gt(["c0"], range(0, 20), range(2))
    pred = _pred_from_gt(gt)
    pred["global_id"] = 5  # both people merged into one id
    res = evaluate_run(gt, pred, seq, _all_frames(seq))
    (case,) = res["identity"]["cases"]
    assert case["kind"] == "false_merge"
    assert case["global_ids"] == [5]
    assert sorted(case["gt_person_ids"]) == [0, 1]
    assert case["tracklet_ids"] == [0, 1]
    assert case["camera_id"] == "c0" and case["t"] == pytest.approx(0.0)


@pytest.mark.slow
def test_eval_benchmark_4cams_2000frames_20people() -> None:
    seq = _seq(n_cams=4, n_frames=2000)
    rng = np.random.default_rng(0)
    gt = _grid_gt([f"c{i}" for i in range(4)], range(2000), range(20))
    pred = _pred_from_gt(gt)
    pred["x"] += rng.normal(0, 5, len(pred))
    pred = pred.sample(frac=0.9, random_state=0)
    labels = pred.drop_duplicates("tracklet_id").set_index("tracklet_id")["global_id"]
    curve = {t: labels for t in np.linspace(0.1, 0.9, 9)}
    t0 = time.perf_counter()
    res = evaluate_run(gt, pred, seq, _all_frames(seq), threshold_labels=curve)
    elapsed = time.perf_counter() - t0
    print(f"evaluate_run 4x2000x20: {elapsed:.1f} s")
    assert elapsed < 30
    assert res["cross_camera"]["IDF1"] > 0.9
