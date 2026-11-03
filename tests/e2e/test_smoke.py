"""Smoke e2e: in-process API + worker tick loop, oracle run on real epfl_terrace/terrace1 (needs data + models)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mtmct.config import paths

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
SEQ = ("epfl_terrace", "terrace1")
LINKED = ("configs", "data", "models")  # shared read-mostly; cache/outputs/db are fresh per test


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if not (REPO / "data" / SEQ[0] / SEQ[1] / "cameras.json").exists():
        pytest.skip("epfl_terrace/terrace1 is not prepared (download it on the Data page)")
    if not list((REPO / "models").glob("*.xml")) and not list((REPO / "models").glob("*.onnx")):
        pytest.skip("model weights are absent (run `make models`)")
    for name in LINKED:
        (tmp_path / name).symlink_to(REPO / name)
    monkeypatch.setenv("MTMCT_HOME", str(tmp_path))
    paths.cache_clear()
    yield tmp_path
    paths.cache_clear()


def test_oracle_run_end_to_end(home: Path) -> None:
    from mtmct.api.app import create_app
    from mtmct.worker.main import Worker

    worker = Worker()
    with TestClient(create_app()) as client:
        r = client.post(
            "/api/runs",
            json={
                "dataset": SEQ[0],
                "sequence": SEQ[1],
                "mode": "oracle",
                "overrides": {"data.max_seconds": 20},
                "name": "e2e smoke",
            },
        )
        assert r.status_code == 200, r.text
        run_id, job_id = r.json()["id"], r.json()["job_id"]

        t0 = time.monotonic()
        while worker.tick():
            assert time.monotonic() - t0 < 300, "oracle smoke exceeded 5 min"

        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "done", job.get("error")
        run = client.get(f"/api/runs/{run_id}").json()
        assert run["status"] == "done"
        assert run["key_metrics"]["cross_idf1"] >= 0.9, run["key_metrics"]

        body = client.get(f"/api/runs/{run_id}/metrics").json()
        assert {"cross_camera", "per_camera", "identity", "reid"} <= body["metrics"].keys()
        assert body["metrics"]["cross_camera"]["IDF1"] >= 0.9
        assert body["runtime"]["peak_rss_gb"] > 0
        assert body["runtime"]["wall_time_s"] < 120  # spec §11: oracle <= 2 min

        tracks = client.get(f"/api/runs/{run_id}/tracks", params={"camera": "c0", "t0": 0, "t1": 5}).json()
        assert any(f["pred"] for f in tracks["frames"])
