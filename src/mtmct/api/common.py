"""Helpers shared by the API routers: ORM → response models, path-safe lookups."""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, HTTPException
from sqlmodel import Session, col, select

from mtmct.api.schemas import JobOut, KeyMetrics, RunOut
from mtmct.config import load_config, paths
from mtmct.data.base import SequenceInfo, load_sequence
from mtmct.db import Job, Metric, Run, get_session

SessionDep = Annotated[Session, Depends(get_session)]
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def safe_name(value: str) -> str:
    """Dataset / sequence / camera ids that reach the filesystem must be plain names (no separators, no `..`)."""
    if not SAFE_NAME.match(value) or ".." in value:
        raise HTTPException(404, f"not found: {value!r}")
    return value


def load_seq(dataset: str, sequence: str) -> SequenceInfo:
    try:
        return load_sequence(paths().data, safe_name(dataset), safe_name(sequence))
    except FileNotFoundError:
        raise HTTPException(404, f"sequence {dataset}/{sequence} is not prepared") from None


def run_dir(run_id: int) -> Path:
    return paths().outputs / f"{run_id:05d}"


def get_run(session: Session, run_id: int) -> Run:
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(404, f"run {run_id} not found")
    return run


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


@cache
def _profile_defaults(configs_dir: str, profile: str) -> dict[str, Any]:
    return flatten(load_config(None, profile).model_dump(mode="json"))


def config_diff(config: dict[str, Any], profile: str) -> dict[str, Any]:
    """Dotted keys where `config` differs from `load_config(None, profile)` (removed keys map to None)."""
    base = _profile_defaults(str(paths().configs), profile)
    cur = flatten(config)
    diff = {k: v for k, v in cur.items() if k not in base or base[k] != v}
    diff.update({k: None for k in base.keys() - cur.keys()})
    return diff


def key_metrics_by_run(session: Session, run_ids: list[int]) -> dict[int, KeyMetrics]:
    values: dict[int, dict[str, float]] = {i: {} for i in run_ids}
    if run_ids:
        rows = session.exec(select(Metric).where(col(Metric.run_id).in_(run_ids), Metric.scope == "key")).all()
        for m in rows:
            values[m.run_id][m.name] = m.value
    return {i: KeyMetrics.model_validate(v) for i, v in values.items()}


def run_out(run: Run, key_metrics: KeyMetrics, detail: bool = False) -> RunOut:
    assert run.id is not None
    return RunOut(
        id=run.id,
        name=run.name,
        dataset=run.dataset,
        sequence=run.sequence,
        split=run.split,
        mode=run.mode,  # type: ignore[arg-type]
        profile=run.profile,
        status=run.status,  # type: ignore[arg-type]
        created_at=run.created_at,
        finished_at=run.finished_at,
        job_id=run.job_id,
        sweep_id=run.sweep_id,
        overrides=run.overrides,
        config=run.config if detail else None,
        config_diff=config_diff(run.config, run.profile),
        git_sha=run.git_sha,
        key_metrics=key_metrics,
        error=run.error,
    )


def runs_out(session: Session, runs: list[Run]) -> list[RunOut]:
    km = key_metrics_by_run(session, [r.id for r in runs if r.id is not None])
    return [run_out(r, km[r.id]) for r in runs if r.id is not None]


def job_out(job: Job) -> JobOut:
    assert job.id is not None
    return JobOut(
        id=job.id,
        kind=job.kind,  # type: ignore[arg-type]
        status=job.status,  # type: ignore[arg-type]
        title=job.title,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        progress=job.progress,
        stage=job.stage,
        stage_progress=job.stage_progress,
        eta_s=job.eta_s,
        error=job.error,
        run_id=job.run_id,
        dataset_id=job.dataset_id,
    )
