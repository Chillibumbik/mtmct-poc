"""Runs, sweeps, comparison and result downloads (spec §10.1)."""

from __future__ import annotations

import itertools
import json
import shutil
import tempfile
import zipfile
from collections.abc import Iterator
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from sqlmodel import Session, col, delete, select

from mtmct.api.common import SessionDep, get_run, key_metrics_by_run, load_seq, run_dir, run_out, runs_out
from mtmct.api.schemas import CompareOut, CompareRow, KeyMetrics, RunCreate, RunMetrics, RunOut, SweepCreate, SweepOut
from mtmct.config import load_config
from mtmct.db import Job, Metric, Run, Sweep

router = APIRouter()

MAX_SWEEP_RUNS = 200
LOWER_IS_BETTER = {"false_merges", "false_splits", "peak_rss_gb", "wall_time_s"}
EXPORT_FILES = ("global_tracks.csv", "metrics.json", "runtime.json", "config.yaml", "git_sha")


def _build_run(
    dataset: str,
    sequence: str,
    split: str,
    mode: str,
    profile: str | None,
    overrides: dict[str, Any],
    name: str | None,
    sweep_id: int | None = None,
) -> Run:
    seq = load_seq(dataset, sequence)
    if split not in seq.meta.splits:
        raise HTTPException(422, f"unknown split {split!r}; available: {list(seq.meta.splits)}")
    if profile is None and dataset == "custom" and seq.overlap != "mixed":
        profile = seq.overlap
    try:
        cfg = load_config(dataset, profile, overrides)
    except (ValidationError, AttributeError, TypeError, KeyError) as e:
        raise HTTPException(422, f"invalid overrides: {e}") from e
    return Run(
        name=name or f"{dataset}/{sequence}/{split}/{mode}",
        dataset=dataset,
        sequence=sequence,
        split=split,
        mode=mode,
        profile=cfg.profile,
        overrides=overrides,
        config=cfg.model_dump(mode="json"),
        sweep_id=sweep_id,
    )


def _enqueue_run(session: Session, run: Run, kind: str = "run") -> Job:
    job = Job(kind=kind, title=f"Прогон #{run.id}: {run.name}", run_id=run.id, payload={"run_id": run.id})
    session.add(job)
    session.flush()
    run.job_id = job.id
    session.add(run)
    return job


@router.post("/runs", response_model=RunOut)
def create_run(body: RunCreate, session: SessionDep) -> RunOut:
    run = _build_run(
        body.dataset,
        body.sequence,
        body.split,
        body.mode,
        body.profile,
        body.overrides,
        body.name,
        body.parent_sweep_id,
    )
    session.add(run)
    session.flush()
    _enqueue_run(session, run)
    session.commit()
    session.refresh(run)
    return run_out(run, KeyMetrics(), detail=True)


@router.post("/sweeps", response_model=SweepOut)
def create_sweep(body: SweepCreate, session: SessionDep) -> SweepOut:
    if not body.grid or any(not values for values in body.grid.values()):
        raise HTTPException(422, "grid must have at least one key and no empty value lists")
    keys = list(body.grid)
    combos = list(itertools.product(*body.grid.values()))
    if len(combos) > MAX_SWEEP_RUNS:
        raise HTTPException(422, f"sweep has {len(combos)} runs, the limit is {MAX_SWEEP_RUNS}")
    name = body.name or f"sweep {body.dataset}/{body.sequence}"
    runs = []
    for combo in combos:
        point = dict(zip(keys, combo, strict=True))
        label = ", ".join(f"{k.rsplit('.', 1)[-1]}={v}" for k, v in point.items())
        runs.append(
            _build_run(
                body.dataset,
                body.sequence,
                body.split,
                body.mode,
                body.profile,
                {**body.base_overrides, **point},
                f"{name} [{label}]",
            )
        )
    sweep = Sweep(name=name)
    session.add(sweep)
    session.flush()
    for run in runs:
        run.sweep_id = sweep.id
        session.add(run)
    job = Job(kind="sweep", title=f"Sweep «{name}»: {len(runs)} прогонов", payload={"sweep_id": sweep.id})
    session.add(job)
    session.flush()
    sweep.job_id = job.id
    session.add(sweep)
    session.commit()
    assert sweep.id is not None and job.id is not None
    return SweepOut(
        id=sweep.id,
        name=name,
        created_at=sweep.created_at,
        n_runs=len(runs),
        run_ids=[r.id for r in runs if r.id is not None],
        job_id=job.id,
    )


@router.get("/runs", response_model=list[RunOut])
def list_runs(
    session: SessionDep,
    dataset: str | None = None,
    sequence: str | None = None,
    split: str | None = None,
    mode: str | None = None,
    status: str | None = None,
    sweep_id: int | None = None,
    limit: Annotated[int, Query(ge=1, le=2000)] = 500,
) -> list[RunOut]:
    stmt = select(Run)
    for column, value in (
        (Run.dataset, dataset),
        (Run.sequence, sequence),
        (Run.split, split),
        (Run.mode, mode),
        (Run.status, status),
        (Run.sweep_id, sweep_id),
    ):
        if value is not None:
            stmt = stmt.where(column == value)
    return runs_out(session, list(session.exec(stmt.order_by(col(Run.id).desc()).limit(limit)).all()))


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run_detail(run_id: int, session: SessionDep) -> RunOut:
    run = get_run(session, run_id)
    return run_out(run, key_metrics_by_run(session, [run_id])[run_id], detail=True)


@router.delete("/runs/{run_id}")
def delete_run(run_id: int, session: SessionDep) -> dict[str, bool]:
    run = get_run(session, run_id)
    if run.status in ("queued", "running"):
        raise HTTPException(409, "cancel the run's job before deleting it")
    session.exec(delete(Metric).where(col(Metric.run_id) == run_id))
    session.delete(run)
    session.commit()
    shutil.rmtree(run_dir(run_id), ignore_errors=True)
    return {"deleted": True}


def read_json_result(run_id: int, filename: str, required: bool = True) -> dict[str, Any]:
    path = run_dir(run_id) / filename
    if not path.exists():
        if required:
            raise HTTPException(404, f"run {run_id} has no {filename} (not finished?)")
        return {}
    return json.loads(path.read_text())


@router.get("/runs/{run_id}/metrics", response_model=RunMetrics)
def run_metrics(run_id: int, session: SessionDep) -> RunMetrics:
    get_run(session, run_id)
    return RunMetrics(
        metrics=read_json_result(run_id, "metrics.json"), runtime=read_json_result(run_id, "runtime.json", False)
    )


class EvalRequest(BaseModel):
    overrides: dict[str, Any] = {}


@router.post("/runs/{run_id}/eval", response_model=RunOut)
def reevaluate_run(run_id: int, body: EvalRequest, session: SessionDep) -> RunOut:
    """Re-run association + evaluation of a finished run (heavy stages come from the cache)."""
    run = get_run(session, run_id)
    if run.status in ("queued", "running"):
        raise HTTPException(409, "run is still in the queue")
    fresh = _build_run(
        run.dataset, run.sequence, run.split, run.mode, run.profile, {**run.overrides, **body.overrides}, run.name
    )
    run.overrides, run.config, run.profile = fresh.overrides, fresh.config, fresh.profile
    run.status, run.error, run.finished_at = "queued", None, None
    _enqueue_run(session, run, kind="eval_only")
    session.commit()
    session.refresh(run)
    return run_out(run, key_metrics_by_run(session, [run_id])[run_id], detail=True)


def _round(x: float) -> float:
    return round(x, 6)


@router.get("/compare", response_model=CompareOut)
def compare(runs: Annotated[str, Query(description="comma-separated run ids")], session: SessionDep) -> CompareOut:
    try:
        ids = [int(x) for x in runs.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(422, "runs must be comma-separated integers") from None
    if not ids:
        raise HTTPException(422, "no run ids given")
    rows_db = {r.id: r for r in session.exec(select(Run).where(col(Run.id).in_(ids))).all()}
    if missing := [i for i in ids if i not in rows_db]:
        raise HTTPException(404, f"runs not found: {missing}")
    km = key_metrics_by_run(session, ids)
    rows = []
    for metric in KeyMetrics.model_fields:
        values: dict[int, float | None] = {i: getattr(km[i], metric) for i in ids}
        base = values[ids[0]]
        higher = metric not in LOWER_IS_BETTER
        present = {i: v for i, v in values.items() if v is not None}
        best = (max if higher else min)(present, key=lambda i: present[i]) if present else None
        rows.append(
            CompareRow(
                metric=metric,
                values=values,
                deltas={i: None if v is None or base is None else _round(v - base) for i, v in values.items()},
                best_run_id=best,
                higher_is_better=higher,
            )
        )
    return CompareOut(runs=[run_out(rows_db[i], km[i]) for i in ids], rows=rows)


@router.get("/runs/{run_id}/export")
def export_run(run_id: int, session: SessionDep) -> StreamingResponse:
    get_run(session, run_id)
    src = run_dir(run_id)
    if not (src / "global_tracks.csv").exists():
        raise HTTPException(404, f"run {run_id} has no results")
    buf = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)  # noqa: SIM115 - closed by the generator
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in EXPORT_FILES:
            if (src / name).exists():
                zf.write(src / name, name)
    buf.seek(0)

    def chunks() -> Iterator[bytes]:
        try:
            while data := buf.read(1 << 20):
                yield data
        finally:
            buf.close()

    return StreamingResponse(
        chunks(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="run_{run_id:05d}.zip"'},
    )
