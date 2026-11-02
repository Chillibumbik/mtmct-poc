"""EPFL Terrace seq.1: 4 overlapping 360x288 @ 25 fps cameras, per-view boxes with cross-view IDs (Xu re-annotation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mtmct.data.base import DatasetMeta, Progress, SequenceInfo, register_adapter
from mtmct.data.common import build_sequence, read_xu_gt
from mtmct.data.fetch import download_all

SEQUENCE = "terrace1"
CAMERAS = ["c0", "c1", "c2", "c3"]
VIDEO_URL = "https://documents.epfl.ch/groups/c/cv/cvlab-pom-video3/www/terrace1-{cam}.avi"
CALIBRATION_URL = "https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/calibration-terrace.txt"
GT_URL = "https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/master/EPFL/Terrace/terrace1-{cam}.txt"
_VIDEO_SIZES = {"c0": 82429516, "c1": 82289204, "c2": 85208752, "c3": 98673344}  # bytes, for resume/skip checks

# The EPFL homographies map an image point to "top view" units, not metres. The affine map below converts these
# units to metres on the Tsai ground plane (fit of the Tsai terrace calibration to the GT foot points of all four
# cameras, median reprojection error 1.5 px; see agent-dump/reports/T1_formats.md).
_TOPVIEW_TO_METRES = np.array([[0.021604, 0.0, -0.379085], [0.0, -0.021446, 8.831136], [0.0, 0.0, 1.0]])


def read_calibration(path: Path) -> dict[str, np.ndarray]:
    """`calibration-terrace.txt` → {camera: 3x3 image→ground homography in metres}."""
    rows = [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    out = {}
    for i, cam in enumerate(CAMERAS):
        H = np.array(rows[4 * i : 4 * i + 3], dtype=float)  # the 4th row of each block is the head-plane height
        out[cam] = _TOPVIEW_TO_METRES @ H
    return out


class EpflTerrace:
    id = "epfl_terrace"

    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            id=self.id,
            title="EPFL Terrace",
            licence="Research use only, (c) CVLab EPFL; cite Fleuret et al. 2008. Annotations: Xu et al. (MIT)",
            source_url="https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/",
            overlap="overlap",
            real=True,
            n_cameras="4",
            download_gb=0.34,
            sequences=[SEQUENCE],
            description=(
                "Четыре полностью перекрывающиеся камеры на террасе, 360x288, 25 к/с, около 3,5 минут и до 9 человек. "
                "Быстрый набор для отладки: межкамерные ID из переразметки Xu, гомографии на плоскость земли."
            ),
        )

    def fetch(self, raw_dir: Path, progress: Progress) -> None:
        jobs: list[tuple[str, Path, int | None]] = [(CALIBRATION_URL, raw_dir / "calibration-terrace.txt", None)]
        for cam in CAMERAS:
            jobs.append((GT_URL.format(cam=cam), raw_dir / f"{SEQUENCE}-{cam}.txt", None))
            jobs.append((VIDEO_URL.format(cam=cam), raw_dir / f"{SEQUENCE}-{cam}.avi", _VIDEO_SIZES[cam]))
        download_all(jobs, progress)

    def prepare(self, raw_dir: Path, data_root: Path, progress: Progress) -> list[SequenceInfo]:
        progress(0.0, f"{self.id}: building {SEQUENCE}")
        gt = pd.concat([read_xu_gt(raw_dir / f"{SEQUENCE}-{cam}.txt", cam) for cam in CAMERAS], ignore_index=True)
        seq = build_sequence(
            data_root,
            self.id,
            SEQUENCE,
            {cam: raw_dir / f"{SEQUENCE}-{cam}.avi" for cam in CAMERAS},
            gt,
            overlap="overlap",
            splits={"all": (0.0, 1.0)},
            homographies=read_calibration(raw_dir / "calibration-terrace.txt"),
            notes="Cameras are hardware-synchronised (frame_offset 0). Homographies: image → ground plane, metres.",
        )
        progress(1.0, f"{self.id}: done")
        return [seq]


register_adapter(EpflTerrace())
