"""In-camera trackers on top of the Roboflow `trackers` package (Apache-2.0), without ReID inside the tracker."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import supervision as sv
from pydantic import BaseModel, Field
from trackers import BoTSORTTracker, ByteTrackTracker, OCSORTTracker

from mtmct.registry import register


class _BaseParams(BaseModel):
    lost_track_buffer: int = Field(30, ge=0, description="Frames a lost track is kept, expressed at 30 fps")
    frame_rate: float = Field(30.0, gt=0, description="Processed frame rate; scales lost_track_buffer")
    track_activation_threshold: float = Field(0.5, ge=0, le=1, description="Min score to start a new track")
    high_conf_det_threshold: float = Field(0.6, ge=0, le=1, description="Score split between 1st and 2nd association")
    minimum_consecutive_frames: int = Field(2, ge=1, description="Matches needed before a track gets an id")


class _LibTracker:
    """Adapter from `components.Tracker` (xywh arrays) to a `trackers` tracker (`sv.Detections`)."""

    LICENCE = "Apache-2.0 (roboflow/trackers)"
    USES_FRAME: ClassVar[bool] = False

    def __init__(self, params: BaseModel) -> None:
        self.params = params
        self._tracker = self._build(params)

    def _build(self, params: Any) -> Any:
        raise NotImplementedError

    def reset(self) -> None:
        self._tracker.reset()

    def update(self, dets: np.ndarray, image: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 5)
        xyxy = np.column_stack([dets[:, 0], dets[:, 1], dets[:, 0] + dets[:, 2], dets[:, 1] + dets[:, 3]])
        det = sv.Detections(xyxy=xyxy.astype(np.float32), confidence=dets[:, 4])
        out = self._tracker.update(det, image if self.USES_FRAME else None)
        if out.tracker_id is None or len(out) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        keep = out.tracker_id >= 0  # unconfirmed tracks have id -1
        b, ids = out.xyxy[keep], out.tracker_id[keep]
        score = out.confidence[keep] if out.confidence is not None else np.ones(len(ids))
        return np.column_stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1], score, ids]).astype(np.float32)


@register("tracker", "bytetrack")
class ByteTrack(_LibTracker):
    """ByteTrack: two-stage IoU association of high- and low-score detections (default)."""

    class Params(_BaseParams):
        minimum_iou_threshold: float = Field(0.1, ge=0, le=1)

    def _build(self, p: Any) -> ByteTrackTracker:
        return ByteTrackTracker(
            lost_track_buffer=p.lost_track_buffer,
            frame_rate=p.frame_rate,
            track_activation_threshold=p.track_activation_threshold,
            minimum_consecutive_frames=p.minimum_consecutive_frames,
            minimum_iou_threshold=p.minimum_iou_threshold,
            high_conf_det_threshold=p.high_conf_det_threshold,
        )


@register("tracker", "botsort")
class BoTSORT(_LibTracker):
    """BoT-SORT without ReID: ByteTrack-style association plus camera motion compensation."""

    USES_FRAME = True

    class Params(_BaseParams):
        enable_cmc: bool = Field(True, description="Camera motion compensation (needs the frame)")
        cmc_downscale: int = Field(2, ge=1)
        minimum_iou_threshold_first_assoc: float = Field(0.2, ge=0, le=1)
        minimum_iou_threshold_second_assoc: float = Field(0.5, ge=0, le=1)
        minimum_iou_threshold_unconfirmed_assoc: float = Field(0.3, ge=0, le=1)

    def _build(self, p: Any) -> BoTSORTTracker:
        return BoTSORTTracker(
            lost_track_buffer=p.lost_track_buffer,
            frame_rate=p.frame_rate,
            track_activation_threshold=p.track_activation_threshold,
            minimum_consecutive_frames=p.minimum_consecutive_frames,
            minimum_iou_threshold_first_assoc=p.minimum_iou_threshold_first_assoc,
            minimum_iou_threshold_second_assoc=p.minimum_iou_threshold_second_assoc,
            minimum_iou_threshold_unconfirmed_assoc=p.minimum_iou_threshold_unconfirmed_assoc,
            high_conf_det_threshold=p.high_conf_det_threshold,
            enable_cmc=p.enable_cmc,
            cmc_downscale=p.cmc_downscale,
        )


@register("tracker", "ocsort")
class OCSORT(_LibTracker):
    """OC-SORT: observation-centric re-update and direction consistency; robust to occlusion."""

    class Params(_BaseParams):
        minimum_consecutive_frames: int = Field(3, ge=1, description="Matches needed before a track gets an id")
        minimum_iou_threshold: float = Field(0.3, ge=0, le=1)
        direction_consistency_weight: float = Field(0.2, ge=0)
        delta_t: int = Field(3, ge=1, description="Frames back used to estimate velocity direction")

    def _build(self, p: Any) -> OCSORTTracker:
        return OCSORTTracker(
            lost_track_buffer=p.lost_track_buffer,
            frame_rate=p.frame_rate,
            minimum_consecutive_frames=p.minimum_consecutive_frames,
            minimum_iou_threshold=p.minimum_iou_threshold,
            direction_consistency_weight=p.direction_consistency_weight,
            high_conf_det_threshold=p.high_conf_det_threshold,
            delta_t=p.delta_t,
        )
