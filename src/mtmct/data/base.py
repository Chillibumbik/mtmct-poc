"""Canonical dataset layout (spec §5.3) and the adapter interface.

data/<dataset>/<sequence>/
  cameras.json   {"cameras": [CameraInfo...]}
  meta.json      SequenceMeta (overlap type, splits, licence, source)
  gt.parquet     optional, GroundTruth schema
  topology.json  optional, camera transition windows (non-overlap)
  videos/<camera_id>.<ext> | frames/<camera_id>/*.jpg
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from mtmct.io import GroundTruth, read_table

Overlap = Literal["overlap", "nonoverlap", "mixed"]
Progress = Callable[[float, str], None]  # (fraction 0..1, message)


class CameraInfo(BaseModel):
    camera_id: str
    source: str  # relative to the sequence root: a video file or a directory of frames
    fps: float
    width: int
    height: int
    n_frames: int  # frames in the camera stream
    frame_offset: int = 0  # added to frame_id to put all cameras on one clock: ts = (frame_id + offset) / fps
    homography_img2world: list[list[float]] | None = None  # 3x3, image pixels → ground plane (metres)

    def ts(self, frame_id: int | np.ndarray) -> float | np.ndarray:
        return (frame_id + self.frame_offset) / self.fps

    def H(self) -> np.ndarray | None:
        return None if self.homography_img2world is None else np.asarray(self.homography_img2world, dtype=np.float64)


class SequenceMeta(BaseModel):
    overlap: Overlap
    # split name → (start, end) as fractions of the sequence duration, e.g. {"tune": (0, 0.9), "test": (0.9, 1)}
    splits: dict[str, tuple[float, float]] = Field(default_factory=lambda: {"all": (0.0, 1.0)})
    duration_s: float
    gt_frame_step: int = 1  # GT annotates every k-th frame (frames in between are unannotated, not empty)
    n_persons: int | None = None
    notes: str = ""


class SequenceInfo(BaseModel):
    dataset: str
    sequence: str
    root: Path
    cameras: list[CameraInfo]
    meta: SequenceMeta

    @property
    def overlap(self) -> Overlap:
        return self.meta.overlap

    @property
    def has_gt(self) -> bool:
        return (self.root / "gt.parquet").exists()

    def camera(self, camera_id: str) -> CameraInfo:
        for cam in self.cameras:
            if cam.camera_id == camera_id:
                return cam
        raise KeyError(camera_id)

    def camera_index(self, camera_id: str) -> int:
        return [c.camera_id for c in self.cameras].index(camera_id)

    def split_range(self, split: str) -> tuple[float, float]:
        """Split → absolute (t0, t1) seconds."""
        if split not in self.meta.splits:
            raise KeyError(f"sequence {self.dataset}/{self.sequence} has no split {split!r}: {list(self.meta.splits)}")
        a, b = self.meta.splits[split]
        return a * self.meta.duration_s, b * self.meta.duration_s

    def gt(self) -> pd.DataFrame | None:
        return read_table(self.root / "gt.parquet", GroundTruth) if self.has_gt else None

    def topology(self) -> dict[str, Any] | None:
        p = self.root / "topology.json"
        return json.loads(p.read_text()) if p.exists() else None

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        cams = {"cameras": [c.model_dump() for c in self.cameras]}
        (self.root / "cameras.json").write_text(json.dumps(cams, indent=2))
        (self.root / "meta.json").write_text(self.meta.model_dump_json(indent=2))


def load_sequence(data_root: Path, dataset: str, sequence: str) -> SequenceInfo:
    root = data_root / dataset / sequence
    if not (root / "cameras.json").exists():
        raise FileNotFoundError(f"sequence not prepared: {root}")
    cams = json.loads((root / "cameras.json").read_text())["cameras"]
    meta = SequenceMeta.model_validate_json((root / "meta.json").read_text())
    return SequenceInfo(
        dataset=dataset, sequence=sequence, root=root, cameras=[CameraInfo(**c) for c in cams], meta=meta
    )


def list_sequences(data_root: Path, dataset: str) -> list[str]:
    d = data_root / dataset
    return sorted(p.name for p in d.iterdir() if (p / "cameras.json").exists()) if d.exists() else []


class DatasetMeta(BaseModel):
    id: str
    title: str
    licence: str
    source_url: str
    overlap: Overlap
    real: bool
    n_cameras: str  # e.g. "4" or "3-4"
    download_gb: float
    requires_auth: bool = False
    sequences: list[str]
    description: str = ""


class DatasetAdapter(Protocol):
    id: str

    def meta(self) -> DatasetMeta: ...

    def fetch(self, raw_dir: Path, progress: Progress) -> None:
        """Download raw files into `raw_dir` (idempotent, resumable)."""
        ...

    def prepare(self, raw_dir: Path, data_root: Path, progress: Progress) -> list[SequenceInfo]:
        """Convert raw files into the canonical layout under `data_root/<id>/<sequence>/`."""
        ...


ADAPTERS: dict[str, DatasetAdapter] = {}

ADAPTER_MODULES = (
    "mtmct.data.epfl_terrace",
    "mtmct.data.campus",
    "mtmct.data.wildtrack",
    "mtmct.data.chirla",
    "mtmct.data.aic24",
)


def register_adapter(adapter: DatasetAdapter) -> DatasetAdapter:
    ADAPTERS[adapter.id] = adapter
    return adapter


def load_adapters() -> dict[str, DatasetAdapter]:
    import importlib

    for module in ADAPTER_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as e:
            if e.name != module:
                raise
    return ADAPTERS
