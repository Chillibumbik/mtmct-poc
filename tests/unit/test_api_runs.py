# ruff: noqa: F811 - pytest fixtures imported from test_api_fixtures
from __future__ import annotations

import io
import json
import zipfile

import cv2
import numpy as np
from test_api_fixtures import FakeRunner, client, home, seq, write_run_outputs  # noqa: F401

from mtmct.config import paths
from mtmct.db import Job, JobEvent, Run, open_session
from mtmct.worker.main import Worker


def _create(client, **kw):
    body = {"dataset": "toy", "sequence": "seq", "split": "all", "mode": "oracle"} | kw
    return client.post("/api/runs", json=body)


def _finish_run(client, **kw):
    r = _create(client, **kw)
    assert r.status_code == 200, r.text
    Worker(runner=FakeRunner()).tick()
    return r.json()["id"]


def test_create_run_enqueues_job(client, seq):
    r = _create(client, overrides={"associate.solver.params.threshold": 0.55}, name="my run")
    assert r.status_code == 200, r.text
    run = r.json()
    assert run["status"] == "queued" and run["name"] == "my run" and run["job_id"]
    assert run["config"]["associate"]["solver"]["params"]["threshold"] == 0.55
    assert run["config_diff"] == {"associate.solver.params.threshold": 0.55}
    job = client.get(f"/api/jobs/{run['job_id']}").json()
    assert job["kind"] == "run" and job["status"] == "queued" and job["run_id"] == run["id"]
    assert [x["id"] for x in client.get("/api/runs").json()] == [run["id"]]


def test_create_run_rejects_bad_input(client, seq):
    assert _create(client, dataset="nope").status_code == 404
    assert _create(client, sequence="../etc").status_code == 404
    assert _create(client, split="zzz").status_code == 422
    assert _create(client, overrides={"associate.bogus_field": 1}).status_code == 422


def test_run_detail_after_worker(client, seq):
    rid = _finish_run(client)
    run = client.get(f"/api/runs/{rid}").json()
    assert run["status"] == "done" and run["key_metrics"]["cross_hota"] == 0.5 and run["finished_at"]
    assert run["finished_at"].endswith("Z") or "+" in run["finished_at"]
    m = client.get(f"/api/runs/{rid}/metrics").json()
    assert m["metrics"]["cross_camera"]["HOTA"] == 0.5 and m["runtime"]["fps"] == 12.5
    assert client.get("/api/runs", params={"status": "done"}).json()[0]["id"] == rid
    assert client.get("/api/runs", params={"status": "failed"}).json() == []
    assert client.get("/api/runs/999").status_code == 404


def test_delete_run(client, seq):
    rid = _finish_run(client)
    out = paths().outputs / f"{rid:05d}"
    assert out.exists()
    assert client.delete(f"/api/runs/{rid}").status_code == 200
    assert not out.exists() and client.get(f"/api/runs/{rid}").status_code == 404
    queued = _create(client).json()
    assert client.delete(f"/api/runs/{queued['id']}").status_code == 409  # still queued


def test_compare_deltas(client, seq):
    a = _finish_run(client)
    better = json.loads(json.dumps(__import__("test_api_fixtures").METRICS))
    better["cross_camera"]["HOTA"] = 0.8
    better["identity"]["false_merges"] = 0
    _create(client)
    Worker(runner=FakeRunner(metrics=better)).tick()
    b = a + 1
    out = client.get("/api/compare", params={"runs": f"{a},{b}"}).json()
    rows = {r["metric"]: r for r in out["rows"]}
    assert [r["id"] for r in out["runs"]] == [a, b]
    hota = rows["cross_hota"]
    assert hota["values"][str(a)] == 0.5 and hota["deltas"][str(a)] == 0 and hota["deltas"][str(b)] == 0.3
    assert hota["best_run_id"] == b and hota["higher_is_better"] is True
    fm = rows["false_merges"]
    assert fm["best_run_id"] == b and fm["higher_is_better"] is False and fm["deltas"][str(b)] == -1
    assert client.get("/api/compare", params={"runs": "x"}).status_code == 422
    assert client.get("/api/compare", params={"runs": "1,99"}).status_code == 404


def test_sweep_expands_grid_and_worker_queues_children(client, seq):
    body = {
        "dataset": "toy",
        "sequence": "seq",
        "split": "tune",
        "mode": "oracle",
        "name": "thr",
        "grid": {
            "associate.solver.params.threshold": [0.3, 0.4, 0.5],
            "associate.solver.params.linkage": ["average", "single"],
        },
    }
    r = client.post("/api/sweeps", json=body)
    assert r.status_code == 200, r.text
    sw = r.json()
    assert sw["n_runs"] == 6 and len(sw["run_ids"]) == 6
    runs = [client.get(f"/api/runs/{i}").json() for i in sw["run_ids"]]
    assert all(x["sweep_id"] == sw["id"] and x["status"] == "queued" and x["job_id"] is None for x in runs)
    assert {x["config_diff"]["associate.solver.params.threshold"] for x in runs} == {0.3, 0.4, 0.5}
    Worker(runner=FakeRunner()).tick()  # sweep job → child run jobs
    jobs = client.get("/api/jobs").json()
    assert sorted(j["kind"] for j in jobs) == ["run"] * 6 + ["sweep"]
    assert (
        client.post("/api/sweeps", json=body | {"grid": {"a.b": list(range(30)), "c.d": list(range(30))}}).status_code
        == 422
    )


def test_tracks_window_gt_and_limit(client, seq):
    rid = _finish_run(client)
    out = client.get(f"/api/runs/{rid}/tracks", params={"camera": "c0", "t0": 0, "t1": 0.5}).json()
    assert out["camera_id"] == "c0" and out["fps"] == 25.0 and out["frame_offset"] == 0 and out["width"] == 64
    frames = out["frames"]
    assert [f["frame_id"] for f in frames] == [0, 2, 4, 6, 8, 10, 12]
    assert frames[1]["ts"] == 2 / 25 and {b["id"] for b in frames[1]["pred"]} == {1, 2}
    assert frames[0]["gt"] == []
    # camera c1 has frame_offset=5: ts is on the common clock
    c1 = client.get(f"/api/runs/{rid}/tracks", params={"camera": "c1", "t0": 0, "t1": 10}).json()
    assert c1["frame_offset"] == 5 and c1["frames"][0]["ts"] == (10 + 5) / 25
    assert client.get(f"/api/runs/{rid}/tracks", params={"camera": "zz"}).status_code == 404


def test_tracks_window_capped_at_30_s(client, seq, tmp_path):
    rid = _finish_run(client)
    import pandas as pd

    from mtmct.io import GlobalTracks, write_table

    n = 1000
    df = pd.DataFrame(
        {
            "camera_id": "c0",
            "global_id": 1,
            "tracklet_id": 1,
            "frame_id": range(n),
            "x": 1.0,
            "y": 1.0,
            "w": 5.0,
            "h": 5.0,
            "world_x": None,
            "world_y": None,
        }
    )
    write_table(df, paths().outputs / f"{rid:05d}" / "global_tracks.csv", GlobalTracks)
    out = client.get(f"/api/runs/{rid}/tracks", params={"camera": "c0", "t0": 0, "t1": 1000}).json()
    # the cap is on time, never on frames, so a window is complete at any fps (here 25 fps)
    assert len(out["frames"]) == 750 and out["frames"][-1]["frame_id"] == 749


def test_tracks_include_gt(client, seq):
    import pandas as pd

    from mtmct.io import GroundTruth, write_table

    gt = pd.DataFrame(
        {
            "camera_id": ["c0", "c0"],
            "frame_id": [2, 3],
            "person_id": [5, 5],
            "x": 1.0,
            "y": 1.0,
            "w": 5.0,
            "h": 5.0,
            "world_x": None,
            "world_y": None,
        }
    )
    write_table(gt, seq.root / "gt.parquet", GroundTruth)
    rid = _finish_run(client)
    frames = client.get(f"/api/runs/{rid}/tracks", params={"camera": "c0", "t0": 0, "t1": 0.5}).json()["frames"]
    by = {f["frame_id"]: f for f in frames}
    assert by[2]["gt"][0]["id"] == 5 and 3 in by and by[3]["pred"] == []


def test_trajectories_person_crops_errors(client, seq):
    rid = _finish_run(client)
    traj = client.get(f"/api/runs/{rid}/trajectories").json()
    assert traj["available"] is True and {t["global_id"] for t in traj["trajectories"]} == {1}
    assert traj["bounds"][0] <= 1.0 and traj["trajectories"][0]["points"][0]["camera_id"] == "c0"

    person = client.get(f"/api/runs/{rid}/person/1").json()
    assert [(s["camera_id"], s["tracklet_ids"]) for s in person["segments"]] == [("c0", [1]), ("c1", [10000001])]
    assert person["segments"][0]["t_start"] == 0.0 and person["segments"][0]["crop_url"].startswith("/api/runs/")
    assert client.get(f"/api/runs/{rid}/person/77").status_code == 404

    errors = client.get(f"/api/runs/{rid}/errors").json()
    merge = errors["false_merges"][0]
    assert merge["gt_person_ids"] == [5, 7] and merge["camera_id"] == "c0"
    assert [t["tracklet_id"] for t in merge["tracklets"]] == [1, 10000001]
    assert merge["tracklets"][0]["global_id"] == 1 and len(merge["tracklets"][0]["crop_urls"]) == 4
    assert errors["false_splits"][0]["global_ids"] == [1, 2]
    assert errors["transitions"] == [{"from_camera": "c0", "to_camera": "c1", "total": 2, "kept": 1}]

    url = merge["tracklets"][0]["crop_urls"][0]
    r = client.get(url)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[0] == 128
    # crop picks the box at the first quarter of the tracklet life: frame 2 of 11 → pixel value 2
    assert abs(int(img.mean()) - 2) <= 3
    assert client.get(f"/api/runs/{rid}/crops/1", params={"i": 9}).status_code == 404
    assert client.get(f"/api/runs/{rid}/crops/424242").status_code == 404


def test_export_zip(client, seq):
    rid = _finish_run(client)
    r = client.get(f"/api/runs/{rid}/export")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    names = set(zipfile.ZipFile(io.BytesIO(r.content)).namelist())
    assert {"global_tracks.csv", "metrics.json", "config.yaml"} <= names


def test_results_missing_for_unfinished_run(client, seq):
    rid = _create(client).json()["id"]
    assert client.get(f"/api/runs/{rid}/metrics").status_code == 404
    assert client.get(f"/api/runs/{rid}/tracks", params={"camera": "c0"}).status_code == 404
    write_run_outputs(paths().outputs / f"{rid:05d}")
    assert client.get(f"/api/runs/{rid}/tracks", params={"camera": "c0"}).status_code == 200
    with open_session() as s:  # events table exists and is empty for a fresh job
        assert s.query(JobEvent).count() == 0 and s.query(Job).count() == 1 and s.query(Run).count() == 1
