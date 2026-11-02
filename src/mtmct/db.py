"""SQLite state shared by the API and the worker process (spec §10.1). Artifacts stay on disk."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import cache
from typing import Any

from sqlalchemy import JSON, Column, Engine, event
from sqlmodel import Field, Session, SQLModel, create_engine

from mtmct.config import paths


def utcnow() -> datetime:
    return datetime.now(UTC)


class Dataset(SQLModel, table=True):
    __tablename__ = "datasets"
    id: str = Field(primary_key=True)
    status: str = "absent"
    status_message: str = ""
    job_id: int | None = None


class Sweep(SQLModel, table=True):
    __tablename__ = "sweeps"
    id: int | None = Field(default=None, primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=utcnow)
    job_id: int | None = None


class Run(SQLModel, table=True):
    __tablename__ = "runs"
    id: int | None = Field(default=None, primary_key=True)
    name: str
    dataset: str = Field(index=True)
    sequence: str
    split: str
    mode: str
    profile: str
    status: str = Field(default="queued", index=True)
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    job_id: int | None = None
    sweep_id: int | None = Field(default=None, index=True)
    overrides: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    git_sha: str | None = None
    error: str | None = None


class Job(SQLModel, table=True):
    __tablename__ = "jobs"
    id: int | None = Field(default=None, primary_key=True)
    kind: str
    status: str = Field(default="queued", index=True)
    title: str
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: float = 0.0
    stage: str | None = None
    stage_progress: float | None = None
    eta_s: float | None = None
    error: str | None = None
    run_id: int | None = None
    dataset_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    cancel_requested: bool = False


class JobEvent(SQLModel, table=True):
    __tablename__ = "job_events"
    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(index=True)
    ts: datetime = Field(default_factory=utcnow)
    level: str = "info"
    stage: str | None = None
    progress: float | None = None
    stage_progress: float | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class Metric(SQLModel, table=True):
    """Headline numbers only (scope="key"); the full metrics.json on disk is the source of truth."""

    __tablename__ = "metrics"
    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(index=True)
    scope: str
    name: str
    value: float


@cache
def _engine(db_file: str) -> Engine:
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn: Any, _record: Any) -> None:
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

    SQLModel.metadata.create_all(engine)
    return engine


def get_engine() -> Engine:
    """Engine for `paths().db / "mtmct.sqlite"`; the schema is created on first use."""
    db_dir = paths().db
    db_dir.mkdir(parents=True, exist_ok=True)
    return _engine(str(db_dir / "mtmct.sqlite"))


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with Session(get_engine()) as session:
        yield session


open_session = contextmanager(get_session)
