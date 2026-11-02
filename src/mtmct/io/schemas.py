"""Data contracts (spec §5.3, §7). Every table written or read by the pipeline goes through these schemas."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing.pandas import Series


class _Base(pa.DataFrameModel):
    class Config:
        coerce = True
        strict = True


class GroundTruth(_Base):
    """Canonical GT: one row per annotated box. `frame_id` is the 0-based video frame index of that camera."""

    camera_id: Series[str]
    frame_id: Series[int] = pa.Field(ge=0)
    person_id: Series[int]
    x: Series[float]
    y: Series[float]
    w: Series[float] = pa.Field(gt=0)
    h: Series[float] = pa.Field(gt=0)
    world_x: Series[float] = pa.Field(nullable=True)
    world_y: Series[float] = pa.Field(nullable=True)


class Detections(_Base):
    camera_id: Series[str]
    frame_id: Series[int] = pa.Field(ge=0)
    ts: Series[float]
    x: Series[float]
    y: Series[float]
    w: Series[float] = pa.Field(gt=0)
    h: Series[float] = pa.Field(gt=0)
    score: Series[float] = pa.Field(ge=0, le=1)


class Tracklets(_Base):
    """One row per box of a single-camera tracklet. `tracklet_id` is unique within a sequence."""

    camera_id: Series[str]
    frame_id: Series[int] = pa.Field(ge=0)
    ts: Series[float]
    tracklet_id: Series[int]
    x: Series[float]
    y: Series[float]
    w: Series[float] = pa.Field(gt=0)
    h: Series[float] = pa.Field(gt=0)
    score: Series[float] = pa.Field(ge=0, le=1)
    quality: Series[float] = pa.Field(ge=0, le=1)
    foot_x: Series[float]
    foot_y: Series[float]
    world_x: Series[float] = pa.Field(nullable=True)
    world_y: Series[float] = pa.Field(nullable=True)
    emb_idx: Series[int] = pa.Field(ge=-1)


class EmbeddingIndex(_Base):
    emb_idx: Series[int] = pa.Field(ge=0)
    camera_id: Series[str]
    frame_id: Series[int] = pa.Field(ge=0)
    tracklet_id: Series[int]


class Portraits(_Base):
    """One row per tracklet. Row i corresponds to row i of `portraits.npy` (mean embedding)."""

    tracklet_id: Series[int] = pa.Field(unique=True)
    camera_id: Series[str]
    t_start: Series[float]
    t_end: Series[float]
    n_boxes: Series[int] = pa.Field(ge=1)
    mean_quality: Series[float] = pa.Field(ge=0, le=1)
    gallery_idx: Series[object]  # list[int]: rows of embeddings.npy used for the gallery
    entry_x: Series[float] = pa.Field(nullable=True)
    entry_y: Series[float] = pa.Field(nullable=True)
    exit_x: Series[float] = pa.Field(nullable=True)
    exit_y: Series[float] = pa.Field(nullable=True)


class GlobalTracks(_Base):
    """Final output (spec §7): `global_tracks.csv`."""

    camera_id: Series[str]
    global_id: Series[int] = pa.Field(ge=0)
    frame_id: Series[int] = pa.Field(ge=0)
    x: Series[float]
    y: Series[float]
    w: Series[float] = pa.Field(gt=0)
    h: Series[float] = pa.Field(gt=0)
    world_x: Series[float] = pa.Field(nullable=True)
    world_y: Series[float] = pa.Field(nullable=True)


TRACKLET_ID_CAMERA_FACTOR = 10_000_000
"""tracklet_id = camera_index * TRACKLET_ID_CAMERA_FACTOR + local_track_id."""
