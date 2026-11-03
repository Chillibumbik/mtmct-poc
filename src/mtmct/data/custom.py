"""User-uploaded videos (+ optional calibration and ground truth) → canonical sequence `custom/<name>`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mtmct.data.base import SequenceInfo
from mtmct.data.common import build_sequence, copy_video

GT_COLUMNS = ["camera_id", "frame_id", "person_id", "x", "y", "w", "h"]


def read_custom_gt(path: Path) -> pd.DataFrame:
    """CSV `camera_id,frame_id,person_id,x,y,w,h` (header optional); frame_id is 0-based."""
    df = pd.read_csv(path, header=None, dtype=str)
    if df.shape[1] != len(GT_COLUMNS):
        raise ValueError(f"expected {len(GT_COLUMNS)} columns, got {df.shape[1]}")
    df.columns = GT_COLUMNS
    if df.iloc[0, 1] == "frame_id":
        df = df.iloc[1:]
    return df.astype(
        {"frame_id": "int32", "person_id": "int64", "x": "float32", "y": "float32", "w": "float32", "h": "float32"}
    ).reset_index(drop=True)


def prepare_custom(
    videos: list[Path], calibration: dict | None, gt: Path | None, name: str, data_root: Path
) -> SequenceInfo:
    """Camera id = video file stem; `calibration` = {camera_id: 3x3 image→world homography}."""
    homographies = {cam: np.asarray(H, dtype=float) for cam, H in (calibration or {}).items()}
    for cam, H in homographies.items():
        if H.shape != (3, 3):
            raise ValueError(f"homography of camera {cam!r} must be 3x3, got {H.shape}")
    return build_sequence(
        data_root,
        "custom",
        name,
        {v.stem: v for v in videos},
        read_custom_gt(gt) if gt else None,
        overlap="mixed",
        splits={"all": (0.0, 1.0)},
        homographies=homographies,
        place=copy_video,
        notes="User upload. Cameras are assumed synchronised (frame_offset 0).",
    )
