"""Shared fixtures/helpers for API and worker tests (imported by test_api_*.py and test_worker*.py)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mtmct.api.schemas import JobEventIn
from mtmct.config import paths
from mtmct.data.base import SequenceInfo
from mtmct.io import GlobalTracks, write_table

REPO = Path(__file__).resolve().parents[2]

METRICS: dict[str, Any] = {
    "run_id": "1",
    "dataset": "toy",
    "sequence": "seq",
    "split": "all",
    "mode": "full",
    "per_camera": {"c0": {"HOTA": 0.7, "IDF1": 0.8}, "c1": {"HOTA": 0.6, "IDF1": 0.7}},
    "per_camera_mean": {"HOTA": 0.65, "IDF1": 0.75},
    "cross_camera": {"HOTA": 0.5, "AssA": 0.4, "IDF1": 0.6, "IDP": 0.6, "IDR": 0.6, "MOTA": 0.5},
    "identity": {
        "pairwise_f1": 0.7,
        "bcubed_f": 0.72,
        "purity": 0.8,
        "n_pred_ids": 2,
        "n_gt_ids": 3,
        "false_merges": 1,
        "false_splits": 2,
        "handover_success": 0.5,
        "cases": [
            {
                "kind": "false_merge",
                "global_ids": [1],
                "gt_person_ids": [5, 7],
                "tracklet_ids": [1, 10000001],
                "t": 0.4,
                "camera_id": "c0",
            },
            {
                "kind": "false_split",
                "global_ids": [1, 2],
                "gt_person_ids": [5],
                "tracklet_ids": [1, 2],
                "t": 0.0,
                "camera_id": "c0",
            },
        ],
        "transitions": [{"from_camera": "c0", "to_camera": "c1", "total": 2, "kept": 1}],
    },
    "reid": {"rank1": 0.9, "rank5": 0.95, "mAP": 0.8},
    "threshold_curve": [{"threshold": 0.3, "pairwise_p": 0.5, "pairwise_r": 0.6, "IDF1": 0.4}],
}
RUNTIME: dict[str, Any] = {"fps": 12.5, "peak_rss_gb": 2.1, "wall_time_s": 33.0, "stages": {"detect": {"time_s": 10.0}}}


def global_tracks_df() -> pd.DataFrame:
    """Gid 1: c0 tracklet 1 (frames 0..20 step 2) then c1 tracklet 10000001; gid 2: c0 tracklet 2."""
    rows = []
    for f in range(0, 21, 2):
        rows.append(("c0", 1, 1, f, 10.0 + f, 10.0, 20.0, 40.0, 1.0 + f / 10, 2.0))
        rows.append(("c0", 2, 2, f, 30.0, 12.0, 15.0, 30.0, None, None))
    for f in range(10, 31, 2):
        rows.append(("c1", 1, 10000001, f, 5.0, 8.0, 18.0, 36.0, 3.0, 2.0))
    cols = ["camera_id", "global_id", "tracklet_id", "frame_id", "x", "y", "w", "h", "world_x", "world_y"]
    return pd.DataFrame(rows, columns=cols)


def write_run_outputs(out_dir: Path, metrics: dict[str, Any] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics or METRICS))
    (out_dir / "runtime.json").write_text(json.dumps(RUNTIME))
    (out_dir / "config.yaml").write_text("profile: overlap\n")
    write_table(global_tracks_df(), out_dir / "global_tracks.csv", GlobalTracks)


@dataclass
class FakeResult:
    run_dir: Path
    metrics: dict[str, Any]
    runtime: dict[str, Any]


class FakeCancelled(Exception):
    pass


class FakeRunner:
    """Stands in for mtmct.pipeline.run.run_pipeline."""

    def __init__(self, metrics: dict[str, Any] | None = None, fail: str | None = None, cancel_polls: int = 0):
        self.metrics = metrics or METRICS
        self.fail = fail
        self.cancel_polls = cancel_polls  # >0: loop polling should_cancel() this many times, then finish
        self.calls: list[dict[str, Any]] = []

    def __call__(self, cfg, seq, mode, split, out_dir, on_event, should_cancel):
        self.calls.append({"cfg": cfg, "seq": seq, "mode": mode, "split": split, "out_dir": out_dir})
        on_event(JobEventIn(stage="detect", progress=0.1, stage_progress=0.5, message="detecting"))
        for _ in range(self.cancel_polls):
            if should_cancel():
                raise FakeCancelled
        if self.fail:
            raise RuntimeError(self.fail)
        on_event(JobEventIn(stage="associate", progress=0.9, stage_progress=1.0, message="associating"))
        write_run_outputs(out_dir, self.metrics)
        return FakeResult(out_dir, self.metrics, RUNTIME)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated MTMCT_HOME with a copy of the repo configs."""
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    monkeypatch.setenv("MTMCT_HOME", str(tmp_path))
    paths.cache_clear()
    yield tmp_path
    paths.cache_clear()


@pytest.fixture
def seq(home: Path, tiny_sequence: SequenceInfo) -> SequenceInfo:
    return tiny_sequence


@pytest.fixture
def client(home: Path) -> TestClient:
    from mtmct.api.app import create_app

    return TestClient(create_app())
