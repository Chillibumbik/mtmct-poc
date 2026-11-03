"""HTTP API contract (spec §10.1). Shared by the backend (zone C) and, via OpenAPI, the web app (zone D)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

JobKind = Literal["dataset_fetch", "dataset_prepare", "run", "sweep"]
JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]
DatasetStatus = Literal["absent", "downloading", "preparing", "ready", "error"]
RunMode = Literal["full", "oracle"]


# ---------- system / registry ----------
class LicenceItem(BaseModel):
    kind: Literal["model", "dataset", "library"]
    name: str
    licence: str
    note: str = ""


class SystemOut(BaseModel):
    version: str
    git_sha: str | None
    cpu: str
    cpu_threads: int
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    openvino_devices: list[str]
    licences: list[LicenceItem]


class ComponentOut(BaseModel):
    category: str
    name: str
    params_schema: dict[str, Any]
    licence: str
    description: str


class RegistryOut(BaseModel):
    components: dict[str, list[ComponentOut]]


class ProfileOut(BaseModel):
    name: Literal["overlap", "nonoverlap"]
    description: str
    config: dict[str, Any]  # fully merged RunConfig for this profile (without dataset overrides)


# ---------- datasets ----------
class SequenceOut(BaseModel):
    name: str
    cameras: list[str]
    duration_s: float
    splits: dict[str, tuple[float, float]]
    has_gt: bool
    has_calibration: bool
    preview_urls: dict[str, str]  # camera_id → URL of a JPEG preview frame
    media_ready: bool  # proxy videos generated


class DatasetOut(BaseModel):
    id: str
    title: str
    licence: str
    source_url: str
    overlap: Literal["overlap", "nonoverlap", "mixed"]
    real: bool
    n_cameras: str
    download_gb: float
    requires_auth: bool
    description: str
    status: DatasetStatus
    status_message: str = ""
    job_id: int | None = None  # active fetch/prepare job
    sequences: list[SequenceOut] = Field(default_factory=list)
    custom: bool = False


# ---------- runs ----------
class RunCreate(BaseModel):
    dataset: str
    sequence: str
    split: str = "all"
    mode: RunMode = "full"
    profile: Literal["overlap", "nonoverlap"] | None = None  # None → dataset default
    overrides: dict[str, Any] = Field(
        default_factory=dict
    )  # dotted keys, e.g. {"associate.solver.params.threshold": 0.5}
    name: str | None = None


class SweepCreate(BaseModel):
    dataset: str
    sequence: str
    split: str = "tune"
    mode: RunMode = "full"
    profile: Literal["overlap", "nonoverlap"] | None = None
    base_overrides: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, list[Any]]  # dotted key → values; cartesian product
    name: str | None = None


class SweepOut(BaseModel):
    id: int
    name: str
    created_at: datetime
    n_runs: int
    run_ids: list[int]
    job_id: int


class KeyMetrics(BaseModel):
    """Headline numbers shown in tables (subset of metrics.json)."""

    cross_hota: float | None = None
    cross_idf1: float | None = None
    cross_assa: float | None = None
    per_camera_hota: float | None = None
    per_camera_idf1: float | None = None
    pairwise_f1: float | None = None
    bcubed_f: float | None = None
    false_merges: int | None = None
    false_splits: int | None = None
    handover_success: float | None = None
    reid_rank1: float | None = None
    fps: float | None = None
    peak_rss_gb: float | None = None
    wall_time_s: float | None = None


class RunOut(BaseModel):
    id: int
    name: str
    dataset: str
    sequence: str
    split: str
    mode: RunMode
    profile: str
    status: JobStatus
    created_at: datetime
    finished_at: datetime | None
    job_id: int | None
    sweep_id: int | None
    overrides: dict[str, Any]
    config: dict[str, Any] | None = None  # full merged config (detail view only)
    config_diff: dict[str, Any] = Field(default_factory=dict)  # dotted key → value differing from profile default
    git_sha: str | None = None
    key_metrics: KeyMetrics = KeyMetrics()
    error: str | None = None


class RunMetrics(BaseModel):
    """Exactly the content of outputs/<run>/metrics.json (spec §9.7) plus runtime.json."""

    metrics: dict[str, Any]
    runtime: dict[str, Any]


class TrackletCrop(BaseModel):
    tracklet_id: int
    camera_id: str
    t_start: float
    t_end: float
    crop_urls: list[str]
    gt_person_id: int | None
    global_id: int


class ErrorCase(BaseModel):
    kind: Literal["false_merge", "false_split"]
    global_ids: list[int]  # predicted ids involved
    gt_person_ids: list[int]
    tracklets: list[TrackletCrop]
    t: float  # suggested player time (seconds)
    camera_id: str


class ErrorsOut(BaseModel):
    false_merges: list[ErrorCase]
    false_splits: list[ErrorCase]
    transitions: list[TransitionCell]


class TransitionCell(BaseModel):
    from_camera: str
    to_camera: str
    total: int
    kept: int


class Box(BaseModel):
    x: float
    y: float
    w: float
    h: float
    id: int  # global_id for predictions, person_id for GT
    tracklet_id: int | None = None


class FrameBoxes(BaseModel):
    frame_id: int
    ts: float
    pred: list[Box]
    gt: list[Box] = Field(default_factory=list)


class TrackBoxes(BaseModel):
    camera_id: str
    fps: float
    frame_offset: int
    width: int
    height: int
    frames: list[FrameBoxes]


class TrajectoryPoint(BaseModel):
    ts: float
    x: float
    y: float
    camera_id: str


class Trajectory(BaseModel):
    global_id: int
    points: list[TrajectoryPoint]


class TrajectoriesOut(BaseModel):
    available: bool  # False when the dataset has no calibration
    bounds: tuple[float, float, float, float] | None  # xmin, ymin, xmax, ymax (metres)
    trajectories: list[Trajectory]


class RouteSegment(BaseModel):
    camera_id: str
    t_start: float
    t_end: float
    tracklet_ids: list[int]
    crop_url: str | None


class PersonRoute(BaseModel):
    global_id: int
    segments: list[RouteSegment]
    gt_person_ids: list[int]


class CompareRow(BaseModel):
    metric: str
    values: dict[int, float | None]  # run_id → value
    deltas: dict[int, float | None]  # run_id → value - value of the first run
    best_run_id: int | None
    higher_is_better: bool


class CompareOut(BaseModel):
    runs: list[RunOut]
    rows: list[CompareRow]


# ---------- jobs ----------
class JobOut(BaseModel):
    id: int
    kind: JobKind
    status: JobStatus
    title: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    progress: float  # 0..1 overall
    stage: str | None
    stage_progress: float | None
    eta_s: float | None
    error: str | None
    run_id: int | None
    dataset_id: str | None


class JobEvent(BaseModel):
    id: int
    job_id: int
    ts: datetime
    level: Literal["debug", "info", "warning", "error"]
    stage: str | None
    progress: float | None  # overall 0..1
    stage_progress: float | None
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class JobEventIn(BaseModel):
    """What pipeline code emits; the worker persists it as JobEvent."""

    level: Literal["debug", "info", "warning", "error"] = "info"
    stage: str | None = None
    progress: float | None = None
    stage_progress: float | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


ErrorsOut.model_rebuild()
