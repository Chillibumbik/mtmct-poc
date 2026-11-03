import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from mtmct.api.schemas import JobEventIn
from mtmct.config import AssociateCfg, ComponentSpec, CostSpec, DataCfg, GeometryCfg, RunConfig, paths
from mtmct.data.base import CameraInfo, SequenceInfo, SequenceMeta
from mtmct.geometry.foot import BottomCenter
from mtmct.io import TRACKLET_ID_CAMERA_FACTOR, GlobalTracks, GroundTruth, read_table, write_table
from mtmct.pipeline.run import MemoryLimitExceeded, RunCancelled, run_pipeline
from mtmct.pipeline.stages import oracle_tracklets
from mtmct.registry import register
from mtmct.video.reader import selected_frame_ids

W, H, FPS, N_FRAMES = 160, 120, 10.0, 30
PALETTE = {1: (0, 0, 255), 2: (0, 255, 0)}  # person_id → BGR colour of its box
CALLS = {"detect": 0, "embed": 0}


@register("detector", "stub_colour_boxes")
class StubDetector:
    """Boxes of the pure palette colours drawn on the synthetic frames."""

    LICENCE = "test"

    class Params(BaseModel):
        pass

    def __init__(self, params: Params) -> None:
        pass

    def detect(self, images: list[np.ndarray]) -> list[np.ndarray]:
        CALLS["detect"] += len(images)
        out = []
        for img in images:
            boxes = []
            for colour in PALETTE.values():
                ys, xs = np.nonzero(np.all(img == colour, axis=2))
                if len(xs):
                    boxes.append([xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, 0.9])
            out.append(np.asarray(boxes, np.float32).reshape(-1, 5))
        return out


@register("reid", "stub_mean_colour")
class StubReid:
    """Embedding = L2-normalised mean BGR colour of the crop."""

    LICENCE = "test"
    dim = 3

    class Params(BaseModel):
        pass

    def __init__(self, params: Params) -> None:
        pass

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        CALLS["embed"] += len(crops)
        e = np.array([c.reshape(-1, 3).mean(axis=0) + 1e-3 for c in crops], np.float32).reshape(-1, 3)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def _box(cam: int, pid: int, f: int) -> tuple[float, float, float, float]:
    return 10.0 + 3 * f + 20 * cam, 10.0 + 55 * (pid - 1), 20.0, 50.0


def make_sequence(root: Path) -> SequenceInfo:
    """2 overlapping cameras, 3 s at 10 fps (PNG frames), 2 people visible in both cameras."""
    gt = []
    for cam in range(2):
        d = root / "frames" / f"c{cam}"
        d.mkdir(parents=True)
        for f in range(N_FRAMES):
            img = np.zeros((H, W, 3), np.uint8)
            for pid, colour in PALETTE.items():
                x, y, w, h = _box(cam, pid, f)
                img[int(y) : int(y + h), int(x) : int(x + w)] = colour
                gt.append((f"c{cam}", f, pid, x, y, w, h, np.nan, np.nan))
            cv2.imwrite(str(d / f"{f:05d}.png"), img)
    cams = [
        CameraInfo(camera_id=f"c{c}", source=f"frames/c{c}", fps=FPS, width=W, height=H, n_frames=N_FRAMES)
        for c in range(2)
    ]
    seq = SequenceInfo(
        dataset="toy", sequence="seq", root=root, cameras=cams, meta=SequenceMeta(overlap="overlap", duration_s=3.0)
    )
    seq.save()
    cols = list(GroundTruth.to_schema().columns)
    write_table(pd.DataFrame(gt, columns=cols), root / "gt.parquet", GroundTruth)
    return seq


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MTMCT_HOME", str(tmp_path / "home"))
    paths.cache_clear()
    CALLS.update(detect=0, embed=0)
    yield tmp_path / "home"
    paths.cache_clear()


@pytest.fixture
def seq(tmp_path: Path) -> SequenceInfo:
    return make_sequence(tmp_path / "data" / "toy" / "seq")


def cfg(**kw) -> RunConfig:
    return RunConfig(
        data=DataCfg(target_fps=FPS),
        detect=ComponentSpec(name="stub_colour_boxes"),
        reid=ComponentSpec(name="stub_mean_colour"),
        **kw,
    )


def run(c: RunConfig, seq: SequenceInfo, out: Path, mode="full", events=None, cancel=lambda: False):
    sink = events if events is not None else []
    return run_pipeline(c, seq, mode, "all", out, sink.append, cancel)


def stage_dirs(home: Path) -> list[str]:
    root = home / "cache" / "v1" / "toy" / "seq"
    return sorted(p.name for p in root.iterdir()) if root.exists() else []


def test_oracle_tracklets_no_global_leak(seq):
    persons = range(1, 9)
    gt = [
        (c, f, p, 5.0 * p, 5.0, 4.0, 10.0, np.nan, np.nan)
        for c in ("c0", "c1")
        for f in range(N_FRAMES)
        for p in persons
    ]
    write_table(pd.DataFrame(gt, columns=list(GroundTruth.to_schema().columns)), seq.root / "gt.parquet", GroundTruth)
    frames = {c.camera_id: selected_frame_ids(c, c.n_frames, FPS) for c in seq.cameras}
    trk = oracle_tracklets(seq, frames, BottomCenter(BottomCenter.Params()))

    # recover the person behind each tracklet from its (unique per person) x position
    trk["person"] = (trk["x"] / 5).round().astype(int)
    per_person = trk.groupby(["camera_id", "person"])["tracklet_id"]
    assert (per_person.nunique() == 1).all()
    local = per_person.first()
    c0 = (local["c0"] % TRACKLET_ID_CAMERA_FACTOR).to_dict()
    c1 = (local["c1"] % TRACKLET_ID_CAMERA_FACTOR).to_dict()
    assert all(local["c1"] >= TRACKLET_ID_CAMERA_FACTOR) and all(local["c0"] < TRACKLET_ID_CAMERA_FACTOR)
    assert c0 != c1, "same local ids in both cameras would leak identity"
    assert sorted(c0.values()) == list(range(8)) and c0 != {p: p - 1 for p in persons}
    assert (trk["score"] == 1.0).all() and trk["quality"].between(0, 1).all()
    pd.testing.assert_frame_equal(
        trk.drop(columns="person"), oracle_tracklets(seq, frames, BottomCenter(BottomCenter.Params()))
    )


@pytest.mark.parametrize("mode", ["full", "oracle"])
def test_run_writes_all_outputs(home, seq, tmp_path, mode):
    events: list[JobEventIn] = []
    res = run(cfg(), seq, tmp_path / "run", mode, events)
    d = res.run_dir
    for name in ("config.yaml", "git_sha", "global_tracks.csv", "metrics.json", "runtime.json", "log.jsonl"):
        assert (d / name).exists(), name
    assert sorted(p.name for p in (d / "per_camera").iterdir()) == ["c0.txt", "c1.txt"]
    first = (d / "per_camera" / "c0.txt").read_text().splitlines()[0].split(",")
    assert len(first) == 10 and first[6:] == ["1", "-1", "-1", "-1"]

    metrics = json.loads((d / "metrics.json").read_text(), parse_constant=lambda c: pytest.fail(f"NaN in json: {c}"))
    assert metrics["mode"] == mode and metrics["run_id"] == "run" and metrics["warnings"] == []
    assert metrics["cross_camera"]["IDF1"] > 0.9 and metrics["identity"]["n_pred_ids"] == 2
    assert metrics["threshold_curve"]
    tracks = read_table(d / "global_tracks.csv", GlobalTracks)
    assert tracks["global_id"].nunique() == 2

    runtime = json.loads((d / "runtime.json").read_text())
    assert {"stages", "wall_time_s", "fps", "peak_rss_gb", "mean_cpu_pct", "frames_processed", "cache_bytes"} <= set(
        runtime
    )
    assert runtime["frames_processed"] == 2 * N_FRAMES and runtime["cache_bytes"] > 0
    heavy = ["detect", "track", "reid", "refine"] if mode == "full" else ["oracle", "reid"]
    assert all(not runtime["stages"][s]["cached"] for s in heavy)

    log = [json.loads(line) for line in (d / "log.jsonl").read_text().splitlines()]
    assert len(log) == len(events) and events[-1].progress == pytest.approx(1.0)
    progress = [e.progress for e in events]
    assert progress == sorted(progress)


def test_stage_cache_hit_skips_compute(home, seq, tmp_path):
    run(cfg(), seq, tmp_path / "a")
    calls = dict(CALLS)
    res = run(cfg(), seq, tmp_path / "b")
    assert calls == CALLS
    assert all(res.runtime["stages"][s]["cached"] for s in ("detect", "track", "reid", "refine"))
    assert json.loads((tmp_path / "a" / "metrics.json").read_text())["cross_camera"] == res.metrics["cross_camera"]


def test_stage_atomic_on_cancel(home, seq, tmp_path):
    cancel = {"on": False}

    def on_event(e: JobEventIn) -> None:
        cancel["on"] |= e.message == "track: start"  # the tracker loop checks cancellation mid-stage

    with pytest.raises(RunCancelled):
        run_pipeline(cfg(), seq, "full", "all", tmp_path / "a", on_event, lambda: cancel["on"])
    names = stage_dirs(home)
    assert any(n.startswith("detect-") for n in names)
    assert not any(n.startswith(("track-", ".track-", "reid-")) for n in names), names
    assert json.loads((tmp_path / "a" / "log.jsonl").read_text().splitlines()[-1])["level"] == "error"

    detected = CALLS["detect"]
    res = run(cfg(), seq, tmp_path / "b")
    assert CALLS["detect"] == detected
    assert res.runtime["stages"]["detect"]["cached"] and not res.runtime["stages"]["track"]["cached"]
    assert not any(n.startswith(".") for n in stage_dirs(home))


def test_ram_guard_aborts(home, seq, tmp_path):
    c = cfg()
    c.data.max_rss_gb = 0.001
    with pytest.raises(MemoryLimitExceeded, match="max_rss_gb"):
        run(c, seq, tmp_path / "a")
    assert stage_dirs(home) == []
    last = json.loads((tmp_path / "a" / "log.jsonl").read_text().splitlines()[-1])
    assert last["level"] == "error" and "MemoryLimitExceeded" in last["message"]


def test_association_change_reuses_heavy_cache(home, seq, tmp_path):
    run(cfg(), seq, tmp_path / "a")
    calls = dict(CALLS)
    assoc = AssociateCfg(solver=ComponentSpec(name="agglomerative", params={"linkage": "average", "threshold": 0.2}))
    res = run(cfg(associate=assoc), seq, tmp_path / "b")
    assert calls == CALLS
    stages = json.loads((tmp_path / "b" / "runtime.json").read_text())["stages"]
    assert all(stages[s]["cached"] for s in ("detect", "track", "reid", "refine"))
    assert not stages["associate"]["cached"]
    assert "threshold: 0.2" in (res.run_dir / "config.yaml").read_text()


def test_geometry_missing_homography_warns_but_succeeds(home, seq, tmp_path):
    events: list[JobEventIn] = []
    assoc = AssociateCfg(costs={"appearance": CostSpec(), "geometry": CostSpec(weight=0.5)})
    res = run(cfg(associate=assoc), seq, tmp_path / "a", events=events)
    warnings = [e for e in events if e.level == "warning"]
    assert warnings and res.metrics["warnings"] == [e.message for e in warnings]
    assert res.metrics["identity"]["n_pred_ids"] == 2


def test_stage_cache_recomputes_when_sequence_inputs_change(home, seq, tmp_path):
    run(cfg(), seq, tmp_path / "a")
    detected = CALLS["detect"]
    seq.meta.notes = "re-prepared"
    seq.save()
    events: list[JobEventIn] = []
    res = run(cfg(), seq, tmp_path / "b", events=events)
    assert CALLS["detect"] > detected
    assert not any(res.runtime["stages"][s]["cached"] for s in ("detect", "track", "reid", "refine"))
    assert any(e.level == "warning" and "stale" in e.message for e in events)
    assert res.metrics["warnings"] == []


def test_stage_cache_without_fingerprint_is_accepted(home, seq, tmp_path):
    run(cfg(), seq, tmp_path / "a")
    for m in (home / "cache").rglob("manifest.json"):
        body = json.loads(m.read_text())
        assert body.pop("inputs_fingerprint")
        m.write_text(json.dumps(body))
    seq.meta.notes = "re-prepared"
    seq.save()
    calls = dict(CALLS)
    res = run(cfg(), seq, tmp_path / "b")
    assert calls == CALLS
    assert all(res.runtime["stages"][s]["cached"] for s in ("detect", "track", "reid", "refine"))


@register("foot_point", "stub_top_center")
class StubTopCenter:
    LICENCE = "test"

    class Params(BaseModel):
        pass

    def __init__(self, params: Params) -> None:
        pass

    def foot(self, boxes_xywh: np.ndarray, keypoints: np.ndarray | None) -> np.ndarray:
        return np.column_stack([boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2, boxes_xywh[:, 1]])


def test_stage_cache_tracks_foot_point_spec(home, seq, tmp_path):
    run(cfg(), seq, tmp_path / "a", mode="oracle")
    res = run(
        cfg(geometry=GeometryCfg(foot_point=ComponentSpec(name="stub_top_center"))), seq, tmp_path / "b", "oracle"
    )
    assert not res.runtime["stages"]["oracle"]["cached"]
