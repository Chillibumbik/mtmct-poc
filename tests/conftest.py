from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mtmct.data.base import CameraInfo, SequenceInfo, SequenceMeta


def write_video(path: Path, n_frames: int, fps: float, size: tuple[int, int] = (64, 48)) -> None:
    """Synthetic video whose frame i has pixel value i % 256 (to identify decoded frames)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for i in range(n_frames):
        writer.write(np.full((h, w, 3), i % 256, dtype=np.uint8))
    writer.release()


@pytest.fixture
def tiny_sequence(tmp_path: Path) -> SequenceInfo:
    """2 cameras, 25 fps, 2 s synthetic videos; camera c1 starts 5 frames later on the common clock."""
    root = tmp_path / "data" / "toy" / "seq"
    write_video(root / "videos" / "c0.avi", 50, 25.0)
    write_video(root / "videos" / "c1.avi", 50, 25.0)
    seq = SequenceInfo(
        dataset="toy",
        sequence="seq",
        root=root,
        cameras=[
            CameraInfo(camera_id="c0", source="videos/c0.avi", fps=25.0, width=64, height=48, n_frames=50),
            CameraInfo(
                camera_id="c1", source="videos/c1.avi", fps=25.0, width=64, height=48, n_frames=50, frame_offset=5
            ),
        ],
        meta=SequenceMeta(overlap="overlap", duration_s=2.0, splits={"all": (0, 1), "tune": (0, 0.5)}),
    )
    seq.save()
    return seq
