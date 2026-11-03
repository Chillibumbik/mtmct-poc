"""Job queue endpoints and SSE progress streams (spec §10.1)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import update
from sqlmodel import Session, col, select
from sse_starlette.sse import EventSourceResponse

from mtmct.api.common import SessionDep, job_out
from mtmct.api.schemas import JobEvent as JobEventOut
from mtmct.api.schemas import JobOut
from mtmct.db import Dataset, Job, JobEvent, Run, open_session, utcnow

router = APIRouter()

TERMINAL = ("done", "failed", "cancelled")
POLL_S = 0.5


def _get_job(session: Session, job_id: int) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")
    return job


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(
    session: SessionDep, status: str | None = None, kind: str | None = None, limit: int = 200
) -> list[JobOut]:
    stmt = select(Job)
    if status:
        stmt = stmt.where(Job.status == status)
    if kind:
        stmt = stmt.where(Job.kind == kind)
    return [job_out(j) for j in session.exec(stmt.order_by(col(Job.id).desc()).limit(min(limit, 1000))).all()]


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, session: SessionDep) -> JobOut:
    return job_out(_get_job(session, job_id))


def _cancel(session: Session, job: Job) -> None:
    """Queued jobs are cancelled at once; running ones get a flag the worker polls. Sweeps cascade to children."""
    cancelled = False
    if job.status == "queued":  # conditional: the worker may have claimed the job since we read it
        won = session.execute(
            update(Job)
            .where(col(Job.id) == job.id, col(Job.status) == "queued")
            .values(status="cancelled", finished_at=utcnow())
        )
        session.refresh(job)
        cancelled = won.rowcount == 1  # type: ignore[attr-defined]
    if job.status == "running":
        job.cancel_requested = True
    if cancelled:
        if job.run_id and (run := session.get(Run, job.run_id)):
            run.status, run.finished_at = "cancelled", job.finished_at
            session.add(run)
        for ds in session.exec(select(Dataset).where(Dataset.job_id == job.id)):
            ds.status, ds.status_message = "absent", "отменено"
            session.add(ds)
    session.add(job)
    if job.kind == "sweep":
        for run in session.exec(select(Run).where(Run.sweep_id == job.payload.get("sweep_id"))):
            if run.status in ("queued", "running"):
                if run.job_id and (child := session.get(Job, run.job_id)):
                    _cancel(session, child)
                else:
                    run.status, run.finished_at = "cancelled", utcnow()
                    session.add(run)


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: int, session: SessionDep) -> JobOut:
    job = _get_job(session, job_id)
    if job.status in TERMINAL:
        raise HTTPException(409, f"job already {job.status}")
    _cancel(session, job)
    session.commit()
    session.refresh(job)
    return job_out(job)


@router.post("/jobs/{job_id}/restart", response_model=JobOut)
def restart_job(job_id: int, session: SessionDep) -> JobOut:
    """Queue a copy of a failed/cancelled job; finished stages are picked up from the cache."""
    old = _get_job(session, job_id)
    if old.status not in ("failed", "cancelled") or old.kind == "sweep":
        raise HTTPException(409, "only failed or cancelled dataset/run jobs can be restarted")
    job = Job(kind=old.kind, title=old.title, run_id=old.run_id, dataset_id=old.dataset_id, payload=old.payload)
    session.add(job)
    session.flush()
    if old.run_id and (run := session.get(Run, old.run_id)):
        run.status, run.error, run.finished_at, run.job_id = "queued", None, None, job.id
        session.add(run)
    if old.dataset_id and (ds := session.get(Dataset, old.dataset_id)):
        ds.status = "downloading" if old.kind == "dataset_fetch" else "preparing"
        ds.status_message, ds.job_id = "в очереди", job.id
        session.add(ds)
    session.commit()
    session.refresh(job)
    return job_out(job)


def _read_events(job_id: int, after: int) -> tuple[list[JobEventOut], str]:
    """Status is read before the events: events are written before the terminal status, so none are missed."""
    with open_session() as s:
        job = s.get(Job, job_id)
        status = job.status if job else "cancelled"
        rows = s.exec(
            select(JobEvent).where(JobEvent.job_id == job_id, col(JobEvent.id) > after).order_by(col(JobEvent.id))
        ).all()
        return [JobEventOut.model_validate(r, from_attributes=True) for r in rows], status


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: int, request: Request, after: int = 0) -> EventSourceResponse:
    """SSE: `job_event` messages in id order, then `end` once the job is finished. Supports Last-Event-ID."""
    with open_session() as s:
        _get_job(s, job_id)
    header = request.headers.get("last-event-id", "")
    last_id = max(after, int(header) if header.isdigit() else 0)

    async def stream() -> AsyncIterator[dict[str, Any]]:
        nonlocal last_id
        while True:
            events, status = await run_in_threadpool(_read_events, job_id, last_id)
            for ev in events:
                last_id = ev.id
                yield {"id": str(ev.id), "event": "job_event", "data": ev.model_dump_json()}
            if status in TERMINAL:
                yield {"event": "end", "data": json.dumps({"status": status})}
                return
            await asyncio.sleep(POLL_S)

    return EventSourceResponse(stream())


def changed_jobs(session: Session, seen: dict[int, tuple[Any, ...]], limit: int = 50) -> list[JobOut]:
    """Recent jobs whose (status, progress, stage) differ from `seen`; updates `seen`."""
    out = []
    for job in session.exec(select(Job).order_by(col(Job.id).desc()).limit(limit)).all():
        sig = (job.status, round(job.progress, 3), job.stage)
        if job.id is not None and seen.get(job.id) != sig:
            seen[job.id] = sig
            out.append(job_out(job))
    return out


@router.get("/events")
async def global_events() -> EventSourceResponse:
    """SSE: `job` messages whenever any job changes status/progress/stage."""

    def poll(seen: dict[int, tuple[Any, ...]]) -> list[JobOut]:
        with open_session() as s:
            return changed_jobs(s, seen)

    async def stream() -> AsyncIterator[dict[str, Any]]:
        seen: dict[int, tuple[Any, ...]] = {}
        while True:
            for job in await run_in_threadpool(poll, seen):
                yield {"event": "job", "data": job.model_dump_json()}
            await asyncio.sleep(1.0)

    return EventSourceResponse(stream())
