# ruff: noqa: F811 - pytest fixtures imported from test_api_fixtures
from __future__ import annotations

import json

from test_api_fixtures import FakeRunner, client, home, seq  # noqa: F401

from mtmct.api.jobs import changed_jobs
from mtmct.db import Job, open_session
from mtmct.worker.main import Worker


def _run_job(client) -> dict:
    r = client.post("/api/runs", json={"dataset": "toy", "sequence": "seq", "mode": "oracle"})
    return r.json()


def test_sse_streams_events_in_order(client, seq):
    run = _run_job(client)
    Worker(runner=FakeRunner()).tick()
    with client.stream("GET", f"/api/jobs/{run['job_id']}/events") as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    events = [json.loads(line[5:]) for line in text.splitlines() if line.startswith("data:")]
    msgs = [e["message"] for e in events if e.get("message") in ("detecting", "associating")]
    assert msgs == ["detecting", "associating"]
    ids = [e["id"] for e in events if "job_id" in e]
    assert ids == sorted(ids) and len(ids) == 2
    assert "event: end" in text
    # resume after the first event id
    with client.stream("GET", f"/api/jobs/{run['job_id']}/events", headers={"Last-Event-ID": str(ids[0])}) as r:
        again = "".join(r.iter_text())
    assert f'"id": {ids[0]},' not in again and f'"id":{ids[0]},' not in again


def test_cancel_job(client, seq):
    queued = _run_job(client)
    r = client.post(f"/api/jobs/{queued['job_id']}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert client.get(f"/api/runs/{queued['id']}").json()["status"] == "cancelled"
    assert client.post(f"/api/jobs/{queued['job_id']}/cancel").status_code == 409
    assert Worker(runner=FakeRunner()).tick() is False  # cancelled jobs are not picked up

    running = _run_job(client)
    with open_session() as s:
        job = s.get(Job, running["job_id"])
        job.status = "running"
        s.add(job)
        s.commit()
    r = client.post(f"/api/jobs/{running['job_id']}/cancel")
    assert r.json()["status"] == "running"  # flag only; the worker finishes the cancellation
    with open_session() as s:
        assert s.get(Job, running["job_id"]).cancel_requested is True
    assert client.post("/api/jobs/9999/cancel").status_code == 404


def test_cancel_racing_worker_claim_sets_flag(client, seq):
    from mtmct.api.jobs import _cancel

    queued = _run_job(client)
    with open_session() as api:
        job = api.get(Job, queued["job_id"])
        assert job.status == "queued"
        with open_session() as worker:  # the worker claims the job after the API has read it
            claimed = worker.get(Job, queued["job_id"])
            claimed.status = "running"
            worker.add(claimed)
            worker.commit()
        _cancel(api, job)
        api.commit()
    with open_session() as s:
        job = s.get(Job, queued["job_id"])
        assert job.status == "running" and job.cancel_requested is True


def test_cancel_sweep_cascades(client, seq):
    body = {
        "dataset": "toy",
        "sequence": "seq",
        "mode": "oracle",
        "grid": {"associate.solver.params.threshold": [0.3, 0.4]},
    }
    sw = client.post("/api/sweeps", json=body).json()
    assert client.post(f"/api/jobs/{sw['job_id']}/cancel").json()["status"] == "cancelled"
    assert {client.get(f"/api/runs/{i}").json()["status"] for i in sw["run_ids"]} == {"cancelled"}


def test_restart_failed_job(client, seq):
    run = _run_job(client)
    Worker(runner=FakeRunner(fail="boom")).tick()
    assert client.get(f"/api/runs/{run['id']}").json()["status"] == "failed"
    r = client.post(f"/api/jobs/{run['job_id']}/restart")
    assert r.status_code == 200 and r.json()["id"] != run["job_id"] and r.json()["status"] == "queued"
    after = client.get(f"/api/runs/{run['id']}").json()
    assert after["status"] == "queued" and after["error"] is None and after["job_id"] == r.json()["id"]
    assert client.post(f"/api/jobs/{r.json()['id']}/restart").status_code == 409


def test_jobs_list_filters_and_progress(client, seq):
    _run_job(client)
    Worker(runner=FakeRunner()).tick()
    _run_job(client)
    jobs = client.get("/api/jobs").json()
    assert [j["status"] for j in jobs] == ["queued", "done"]  # newest first
    assert client.get("/api/jobs", params={"status": "done"}).json()[0]["progress"] == 1.0
    assert client.get("/api/jobs/12345").status_code == 404


def test_changed_jobs_reports_only_changes(client, seq):
    _run_job(client)
    seen: dict = {}
    with open_session() as s:
        first = changed_jobs(s, seen)
        assert [j.status for j in first] == ["queued"]
        assert changed_jobs(s, seen) == []
        job = s.get(Job, first[0].id)
        job.status = "running"
        job.progress = 0.4
        s.add(job)
        s.commit()
        assert [j.status for j in changed_jobs(s, seen)] == ["running"]
