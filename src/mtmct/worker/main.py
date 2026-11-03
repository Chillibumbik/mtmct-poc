"""Background job worker: one job at a time, state in SQLite (spec §10.1). Run as `python -m mtmct.worker`."""

from __future__ import annotations

import fcntl
import json
import logging
import math
import shutil
import signal
import threading
import time
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import update
from sqlmodel import Session, col, delete, select

from mtmct.api.schemas import JobEventIn, KeyMetrics
from mtmct.config import RunConfig, paths
from mtmct.data.base import Progress, SequenceInfo, load_adapters, load_sequence
from mtmct.db import Dataset, Job, JobEvent, Metric, Run, Sweep, open_session, utcnow
from mtmct.worker.media import ensure_media

log = logging.getLogger("mtmct.worker")

Runner = Callable[..., Any]  # run_pipeline(cfg, seq, mode, split, out_dir, on_event, should_cancel) -> RunResult

# KeyMetrics field → (source, path inside metrics.json / runtime.json)
KEY_METRIC_SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    "cross_hota": ("metrics", ("cross_camera", "HOTA")),
    "cross_idf1": ("metrics", ("cross_camera", "IDF1")),
    "cross_assa": ("metrics", ("cross_camera", "AssA")),
    "per_camera_hota": ("metrics", ("per_camera_mean", "HOTA")),
    "per_camera_idf1": ("metrics", ("per_camera_mean", "IDF1")),
    "pairwise_f1": ("metrics", ("identity", "pairwise_f1")),
    "bcubed_f": ("metrics", ("identity", "bcubed_f")),
    "false_merges": ("metrics", ("identity", "false_merges")),
    "false_splits": ("metrics", ("identity", "false_splits")),
    "handover_success": ("metrics", ("identity", "handover_success")),
    "reid_rank1": ("metrics", ("reid", "rank1")),
    "fps": ("runtime", ("fps",)),
    "peak_rss_gb": ("runtime", ("peak_rss_gb",)),
    "wall_time_s": ("runtime", ("wall_time_s",)),
}


class JobCancelled(Exception):
    pass


def extract_key_metrics(metrics: dict[str, Any], runtime: dict[str, Any]) -> dict[str, float]:
    """Finite numeric headline values of a run (missing / null / NaN entries are skipped)."""
    out: dict[str, float] = {}
    for name, (source, path) in KEY_METRIC_SOURCES.items():
        node: Any = metrics if source == "metrics" else runtime
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, int | float) and not isinstance(node, bool) and math.isfinite(node):
            out[name] = float(node)
    assert set(out) <= set(KeyMetrics.model_fields)
    return out


class JobContext:
    """Persists a job's events/progress and answers cancellation polls."""

    def __init__(self, job_id: int):
        self.job_id = job_id
        self._cancel = False
        self._last_poll = 0.0

    def emit(self, ev: JobEventIn) -> None:
        with open_session() as s:
            s.add(
                JobEvent(
                    job_id=self.job_id,
                    level=ev.level,
                    stage=ev.stage,
                    progress=ev.progress,
                    stage_progress=ev.stage_progress,
                    message=ev.message,
                    data=ev.data,
                )
            )
            job = s.get(Job, self.job_id)
            if job is not None:
                if ev.stage:
                    job.stage = ev.stage
                if ev.stage_progress is not None:
                    job.stage_progress = ev.stage_progress
                if ev.progress is not None:
                    job.progress = ev.progress
                    job.eta_s = _eta(job.started_at, ev.progress, ev.data)
                s.add(job)
            s.commit()

    def should_cancel(self) -> bool:
        if not self._cancel and time.monotonic() - self._last_poll > 0.2:
            self._last_poll = time.monotonic()
            with open_session() as s:
                job = s.get(Job, self.job_id)
                self._cancel = bool(job and job.cancel_requested)
        return self._cancel

    def progress(self, stage: str, lo: float, hi: float) -> Progress:
        """Progress callback for adapter code: maps 0..1 into [lo, hi] of the job; aborts on cancel."""
        last_p, last_msg, last_t = -1.0, "", 0.0

        def cb(fraction: float, message: str) -> None:
            if self.should_cancel():
                raise JobCancelled
            nonlocal last_p, last_msg, last_t
            now = time.monotonic()
            if message == last_msg and fraction < 1 and fraction - last_p < 0.01 and now - last_t < 1.0:
                return
            last_p, last_msg, last_t = fraction, message, now
            self.emit(
                JobEventIn(stage=stage, progress=lo + (hi - lo) * fraction, stage_progress=fraction, message=message)
            )

        return cb


def _eta(started_at: datetime | None, progress: float, data: dict[str, Any]) -> float | None:
    if isinstance(data.get("eta_s"), int | float):
        return float(data["eta_s"])
    if started_at is None or progress < 0.02:
        return None
    return (utcnow() - started_at).total_seconds() * (1 - progress) / progress


class Worker:
    def __init__(self, runner: Runner | None = None, poll_interval: float = 0.5):
        self.runner = runner
        self.poll_interval = poll_interval

    # ---- lifecycle ----
    def recover_stale(self) -> int:
        """Jobs left `running` by a dead worker become `failed`; their runs and datasets follow."""
        reason = "worker restarted while the job was running"
        with open_session() as s:
            stale = s.exec(select(Job).where(Job.status == "running")).all()
            for job in stale:
                assert job.id is not None
                s.add(JobEvent(job_id=job.id, level="error", message=reason))
                self._finalize(s, job, "failed", reason)
            s.commit()
        return len(stale)

    def serve(self, stop: threading.Event) -> None:
        self.recover_stale()
        while not stop.is_set():
            if not self.tick():
                stop.wait(self.poll_interval)

    def tick(self) -> bool:
        """Run the oldest queued job to completion. Returns False when the queue is empty."""
        with open_session() as s:
            job = s.exec(select(Job).where(Job.status == "queued").order_by(col(Job.id))).first()
            if job is None:
                return False
            job_id, kind = job.id, job.kind
            claimed = s.execute(
                update(Job)
                .where(col(Job.id) == job_id, col(Job.status) == "queued")
                .values(status="running", started_at=utcnow())
            )
            if job.run_id:
                s.execute(update(Run).where(col(Run.id) == job.run_id).values(status="running"))
            s.commit()
            if claimed.rowcount == 0:  # type: ignore[attr-defined]
                return True
        assert job_id is not None
        ctx = JobContext(job_id)
        log.info("job %s (%s) started", job_id, kind)
        try:
            self._handlers()[kind](self, job_id, ctx)
            status, error = "done", None
        except Exception as e:
            if isinstance(e, JobCancelled) or ctx.should_cancel():
                status, error = "cancelled", None
            else:
                status, error = "failed", f"{type(e).__name__}: {e}"
                log.exception("job %s failed", job_id)
                ctx.emit(JobEventIn(level="error", message=error, data={"traceback": traceback.format_exc()[-4000:]}))
        with open_session() as s:
            job = s.get(Job, job_id)
            assert job is not None
            self._finalize(s, job, status, error)
            s.commit()
        return True

    @staticmethod
    def _finalize(s: Session, job: Job, status: str, error: str | None) -> None:
        job.status, job.error, job.finished_at, job.eta_s = status, error, utcnow(), None
        if status == "done":
            job.progress, job.stage_progress = 1.0, None
        s.add(job)
        if job.run_id and (run := s.get(Run, job.run_id)):
            run.status, run.error, run.finished_at = status, error, job.finished_at
            s.add(run)
        if status != "done":
            for ds in s.exec(select(Dataset).where(Dataset.job_id == job.id)):
                ds.status = "error" if status == "failed" else "absent"
                ds.status_message = error or "отменено"
                s.add(ds)

    # ---- handlers ----
    @classmethod
    def _handlers(cls) -> dict[str, Callable[[Worker, int, JobContext], None]]:
        return {
            "run": cls._do_run,
            "eval_only": cls._do_run,
            "sweep": cls._do_sweep,
            "dataset_fetch": cls._do_dataset,
            "dataset_prepare": cls._do_dataset,
        }

    @staticmethod
    def _default_runner() -> Runner:
        try:
            from mtmct.pipeline.run import run_pipeline
        except ImportError as e:
            raise RuntimeError(f"pipeline is not available: {e}") from e
        return run_pipeline

    def _do_run(self, job_id: int, ctx: JobContext) -> None:
        with open_session() as s:
            job = s.get(Job, job_id)
            run = s.get(Run, job.run_id) if job and job.run_id else None
            if run is None:
                raise RuntimeError("run not found (deleted?)")
            run_id, cfg_dict, dataset, sequence = run.id, run.config, run.dataset, run.sequence
            mode, split = run.mode, run.split
        assert run_id is not None
        cfg = RunConfig.model_validate(cfg_dict)
        seq = load_sequence(paths().data, dataset, sequence)
        out_dir = paths().outputs / f"{run_id:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        runner = self.runner or self._default_runner()
        result = runner(
            cfg, seq, mode=mode, split=split, out_dir=out_dir, on_event=ctx.emit, should_cancel=ctx.should_cancel
        )
        sha = out_dir / "git_sha"
        with open_session() as s:
            s.exec(delete(Metric).where(col(Metric.run_id) == run_id))
            for name, value in extract_key_metrics(result.metrics, result.runtime).items():
                s.add(Metric(run_id=run_id, scope="key", name=name, value=value))
            if sha.exists() and (run := s.get(Run, run_id)):
                run.git_sha = sha.read_text().strip() or None
                s.add(run)
            s.commit()

    def _do_sweep(self, job_id: int, ctx: JobContext) -> None:
        with open_session() as s:
            job = s.get(Job, job_id)
            assert job is not None
            sweep = s.get(Sweep, job.payload["sweep_id"])
            name = sweep.name if sweep else ""  # read before commit: the instance expires and detaches
            runs = s.exec(select(Run).where(Run.sweep_id == job.payload["sweep_id"]).order_by(col(Run.id))).all()
            for run in runs:
                child = Job(
                    kind="run", title=f"Прогон #{run.id}: {run.name}", run_id=run.id, payload={"run_id": run.id}
                )
                s.add(child)
                s.flush()
                run.job_id = child.id
                s.add(run)
            s.commit()
        ctx.emit(
            JobEventIn(
                stage="sweep",
                progress=1.0,
                message=f"поставлено в очередь прогонов: {len(runs)} ({name})",
            )
        )

    def _do_dataset(self, job_id: int, ctx: JobContext) -> None:
        with open_session() as s:
            job = s.get(Job, job_id)
            assert job is not None
            kind, payload = job.kind, dict(job.payload)
        ds_id: str = payload["dataset_id"]
        self._set_dataset(ds_id, "downloading" if kind == "dataset_fetch" else "preparing", job_id)
        if ds_id == "custom":
            seqs = self._prepare_custom(payload, ctx)
        else:
            adapter = load_adapters()[ds_id]
            raw = paths().raw / ds_id
            if kind == "dataset_fetch":
                adapter.fetch(raw, ctx.progress("fetch", 0.0, 0.5))
            self._set_dataset(ds_id, "preparing", job_id)
            seqs = adapter.prepare(raw, paths().data, ctx.progress("prepare", 0.5, 0.65))
        for i, seq in enumerate(seqs):
            lo = 0.65 + 0.35 * i / len(seqs)
            ensure_media(seq, ctx.progress("proxy", lo, lo + 0.35 / len(seqs)), ctx.should_cancel)
        self._set_dataset(ds_id, "ready", job_id)
        ctx.emit(JobEventIn(stage="proxy", progress=1.0, message=f"готово: {', '.join(s.sequence for s in seqs)}"))

    @staticmethod
    def _prepare_custom(payload: dict[str, Any], ctx: JobContext) -> list[SequenceInfo]:
        raw = paths().raw / "custom" / payload["name"]
        try:
            try:
                from mtmct.data.custom import prepare_custom
            except ImportError as e:
                raise RuntimeError(f"custom dataset preparation is not available: {e}") from e
            calibration = json.loads(Path(payload["calibration"]).read_text()) if payload.get("calibration") else None
            gt = Path(payload["gt"]) if payload.get("gt") else None
            ctx.emit(JobEventIn(stage="prepare", progress=0.2, message="подготовка видео"))
            seq = prepare_custom([Path(v) for v in payload["videos"]], calibration, gt, payload["name"], paths().data)
            return [seq]
        except BaseException:
            shutil.rmtree(raw, ignore_errors=True)
            raise

    @staticmethod
    def _set_dataset(ds_id: str, status: str, job_id: int | None) -> None:
        with open_session() as s:
            ds = s.get(Dataset, ds_id) or Dataset(id=ds_id)
            ds.status, ds.status_message, ds.job_id = (
                status,
                "",
                job_id if status in ("downloading", "preparing") else None,
            )
            s.add(ds)
            s.commit()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    paths().db.mkdir(parents=True, exist_ok=True)
    lock = open(paths().db / "worker.lock", "w")  # noqa: SIM115 - held for the process lifetime
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another worker is already running") from None
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    log.info("worker started, home=%s", paths().root)
    Worker().serve(stop)
