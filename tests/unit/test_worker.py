# ruff: noqa: F811 - pytest fixtures imported from test_api_fixtures
from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import select
from test_api_fixtures import FakeRunner, home, seq  # noqa: F401

from mtmct.config import load_config, paths
from mtmct.db import Dataset, Job, JobEvent, Metric, Run, Sweep, open_session
from mtmct.worker.main import Worker


def _make_run(sequence: str = "seq", mode: str = "full") -> tuple[int, int]:
    cfg = load_config(None, "overlap")
    with open_session() as s:
        run = Run(
            name="r",
            dataset="toy",
            sequence=sequence,
            split="all",
            mode=mode,
            profile="overlap",
            overrides={},
            config=cfg.model_dump(mode="json"),
        )
        s.add(run)
        s.commit()
        job = Job(kind="run", title="run", run_id=run.id, payload={"run_id": run.id})
        s.add(job)
        s.commit()
        run.job_id = job.id
        s.add(run)
        s.commit()
        return run.id, job.id


def test_worker_marks_stale_running_failed(home, seq):
    run_id, job_id = _make_run()
    with open_session() as s:
        job = s.get(Job, job_id)
        job.status = "running"
        run = s.get(Run, run_id)
        run.status = "running"
        s.add(job)
        s.add(run)
        s.add(Dataset(id="toy", status="downloading", job_id=job_id))
        s.commit()
    Worker(runner=FakeRunner()).recover_stale()
    with open_session() as s:
        job = s.get(Job, job_id)
        assert job.status == "failed" and "worker restarted" in job.error
        assert s.get(Run, run_id).status == "failed"
        assert s.get(Dataset, "toy").status == "error"
        assert any("worker restarted" in e.message for e in s.exec(select(JobEvent)).all())


def test_worker_runs_run_job_and_stores_key_metrics(home, seq):
    run_id, job_id = _make_run()
    runner = FakeRunner()
    assert Worker(runner=runner).tick() is True
    assert Worker(runner=runner).tick() is False  # queue empty
    with open_session() as s:
        job = s.get(Job, job_id)
        run = s.get(Run, run_id)
        assert job.status == "done" and job.progress == 1.0 and run.status == "done" and run.finished_at
        key = {m.name: m.value for m in s.exec(select(Metric).where(Metric.run_id == run_id)).all()}
        events = s.exec(select(JobEvent).where(JobEvent.job_id == job_id).order_by(JobEvent.id)).all()
    assert key["cross_hota"] == 0.5 and key["per_camera_idf1"] == 0.75 and key["false_merges"] == 1
    assert key["fps"] == 12.5 and key["peak_rss_gb"] == 2.1 and key["reid_rank1"] == 0.9
    assert [e.message for e in events if e.stage] == ["detecting", "associating"]
    assert runner.calls[0]["out_dir"] == paths().outputs / f"{run_id:05d}"
    assert (runner.calls[0]["out_dir"] / "metrics.json").exists()


def test_worker_job_failure_recorded(home, seq):
    run_id, job_id = _make_run()
    Worker(runner=FakeRunner(fail="boom")).tick()
    with open_session() as s:
        assert s.get(Job, job_id).status == "failed" and "boom" in s.get(Job, job_id).error
        assert s.get(Run, run_id).status == "failed" and "boom" in s.get(Run, run_id).error


def test_worker_cancel_running_job(home, seq):
    run_id, job_id = _make_run()
    with open_session() as s:
        job = s.get(Job, job_id)
        job.cancel_requested = True
        s.add(job)
        s.commit()
    Worker(runner=FakeRunner(cancel_polls=3)).tick()
    with open_session() as s:
        assert s.get(Job, job_id).status == "cancelled" and s.get(Run, run_id).status == "cancelled"


def test_worker_missing_pipeline_fails_cleanly(home, seq, monkeypatch):
    _, job_id = _make_run()
    monkeypatch.setitem(__import__("sys").modules, "mtmct.pipeline.run", None)  # import raises ImportError
    Worker().tick()
    with open_session() as s:
        assert s.get(Job, job_id).status == "failed" and "pipeline" in s.get(Job, job_id).error


def test_worker_sweep_creates_run_jobs(home, seq):
    run_ids = [_make_run()[0] for _ in range(2)]
    with open_session() as s:
        for rid in run_ids:  # runs belong to the sweep; drop their own jobs
            run = s.get(Run, rid)
            s.delete(s.get(Job, run.job_id))
            run.job_id = None
            run.sweep_id = 1
            s.add(run)
        job = Job(kind="sweep", title="sweep", payload={"sweep_id": 1})
        s.add(job)
        s.commit()
        jid = job.id
    Worker(runner=FakeRunner()).tick()
    with open_session() as s:
        assert s.get(Job, jid).status == "done"
        kids = s.exec(select(Job).where(Job.kind == "run").order_by(Job.id)).all()
        assert [j.run_id for j in kids] == run_ids and all(j.status == "queued" for j in kids)
        assert [s.get(Run, r).job_id for r in run_ids] == [j.id for j in kids]


def test_worker_sweep_with_sweep_row_finishes_done(home, seq):
    """The final event reads the sweep name after the session is closed: it must not hit an expired instance."""
    rid = _make_run()[0]
    with open_session() as s:
        sweep = Sweep(name="grid")
        s.add(sweep)
        s.flush()
        run = s.get(Run, rid)
        s.delete(s.get(Job, run.job_id))
        run.job_id, run.sweep_id = None, sweep.id
        s.add(run)
        job = Job(kind="sweep", title="sweep", payload={"sweep_id": sweep.id})
        s.add(job)
        s.commit()
        jid = job.id
    Worker(runner=FakeRunner()).tick()
    with open_session() as s:
        assert s.get(Job, jid).status == "done", s.get(Job, jid).error


def test_worker_dataset_prepare_custom_import_error(home, monkeypatch):
    raw = paths().raw / "custom" / "mine"
    raw.mkdir(parents=True)
    (raw / "c0.mp4").write_bytes(b"x")
    with open_session() as s:
        s.add(
            Job(
                kind="dataset_prepare",
                title="p",
                dataset_id="custom",
                payload={
                    "dataset_id": "custom",
                    "name": "mine",
                    "videos": [str(raw / "c0.mp4")],
                    "calibration": None,
                    "gt": None,
                },
            )
        )
        s.commit()
    monkeypatch.setitem(__import__("sys").modules, "mtmct.data.custom", None)  # import raises ImportError
    Worker(runner=FakeRunner()).tick()
    with open_session() as s:
        job = s.exec(select(Job)).one()
        assert job.status == "failed" and "custom" in job.error
        assert s.get(Dataset, "custom").status == "error"
    assert not raw.exists()


def test_make_proxy_downscales_and_keeps_fps(tmp_path: Path):
    import cv2
    from conftest import write_video

    from mtmct.worker.media import make_proxy

    src = tmp_path / "a.avi"
    write_video(src, 20, 25.0, size=(320, 200))
    dst = tmp_path / "out" / "a.mp4"
    make_proxy(src, dst, max_side=160)
    cap = cv2.VideoCapture(str(dst))
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (160, 100)
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(25.0, abs=0.5)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20


def test_ensure_media_makes_proxies_and_previews(home, seq):
    from mtmct.worker.media import ensure_media

    ensure_media(seq)
    for cam in ("c0", "c1"):
        assert (seq.root / "proxy" / f"{cam}.mp4").stat().st_size > 0
        assert (seq.root / "preview" / f"{cam}.jpg").stat().st_size > 0
