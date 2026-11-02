"""CAMPUS Garden1 / Garden2 (UCLA, Xu et al.): 4 partially overlapping 1080p @ 30 fps cameras, cross-view IDs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from mtmct.data.base import DatasetMeta, Progress, SequenceInfo, register_adapter
from mtmct.data.common import build_sequence, read_xu_gt
from mtmct.data.fetch import download_all

BASE_URL = "https://bitbucket.org/merayxu/multiview-object-tracking-dataset/raw/master/CAMPUS"

# sequence → (Bitbucket folder, views, calibration file, splits)
SEQUENCES = {
    "garden1": ("Garden1", ["Contour2", "HC2", "HC3", "IP1"], "cam_param.mat", {"tune": (0.0, 1.0), "all": (0.0, 1.0)}),
    "garden2": ("Garden2", ["HC1", "HC2", "HC3", "HC4"], "calibration.mat", {"test": (0.0, 1.0), "all": (0.0, 1.0)}),
}
# IP1 of Garden1 starts ~2 s after the other views (estimated from the cross-view GT, see T1_formats.md)
FRAME_OFFSETS = {"garden1": {"IP1": 60}}
SIZES: dict[str, int] = {
    "garden1/cam_param.mat": 765,
    "garden1/view-Contour2.mp4": 84936062,
    "garden1/view-HC2.mp4": 97955198,
    "garden1/view-HC3.mp4": 88478132,
    "garden1/view-IP1.mp4": 109964896,
    "garden2/calibration.mat": 22172109,
    "garden2/view-HC1.mp4": 197951378,
    "garden2/view-HC2.mp4": 201113350,
    "garden2/view-HC3.mp4": 200247191,
    "garden2/view-HC4.mp4": 199313715,
}  # bytes, for resume/skip checks


def homography_img2world(
    K: np.ndarray, R: np.ndarray, C: np.ndarray, scale: float, center: tuple[float, float]
) -> np.ndarray:
    """Image pixel → ground plane (Z=0, metres) for the model K·R·(P − C) ∝ (scale·x − cx, scale·y − cy, 1)."""
    KR = K @ R
    world_to_image = np.c_[KR[:, 0], KR[:, 1], -KR @ C]
    pixel_to_centered = np.array([[scale, 0, -center[0]], [0, scale, -center[1]], [0, 0, 1.0]])
    return np.linalg.inv(world_to_image) @ pixel_to_centered


def read_calibration(path: Path, sequence: str) -> dict[str, np.ndarray]:
    """{view: 3x3 homography}. Garden1 `cam_param.mat`: K,R,T(=C),scale,center; Garden2 `calibration.mat`: K,R,x,y,z."""
    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    out = {}
    if sequence == "garden1":
        for c in mat["cam_param"]:
            out[str(c.name).removeprefix("view-")] = homography_img2world(
                np.asarray(c.K, float),
                np.asarray(c.R, float),
                np.asarray(c.T, float),
                float(c.scale),
                (float(c.center[0]), float(c.center[1])),
            )
    else:
        for im in mat["frames"].image:
            h, w = (float(v) for v in im.imsize)
            out[str(im.name)] = homography_img2world(
                np.asarray(im.K, float),
                np.asarray(im.R, float),
                np.array([im.x, im.y, im.z], float),
                float(im.scale),
                (w / 2 * im.scale, h / 2 * im.scale),
            )
    return out


class Campus:
    id = "campus"

    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            id=self.id,
            title="CAMPUS (Garden1, Garden2)",
            licence="MIT (Xu et al., UCLA VCLA)",
            source_url="https://bitbucket.org/merayxu/multiview-object-tracking-dataset",
            overlap="overlap",
            real=True,
            n_cameras="4",
            download_gb=1.2,
            sequences=list(SEQUENCES),
            description=(
                "Сад кампуса UCLA: 4 камеры 1080p, 30 к/с, частичное перекрытие, 15-25 человек в кадре, плотные "
                "сцены. Garden1 используется для подбора параметров (tune), Garden2 для итоговой оценки (test)."
            ),
        )

    def fetch(self, raw_dir: Path, progress: Progress) -> None:
        jobs: list[tuple[str, Path, int | None]] = []
        for seq, (folder, views, calib, _) in SEQUENCES.items():
            names = [calib, *(f"view-{v}.{ext}" for v in views for ext in ("txt", "mp4"))]
            jobs += [(f"{BASE_URL}/{folder}/{n}", raw_dir / seq / n, SIZES.get(f"{seq}/{n}")) for n in names]
        download_all(jobs, progress)

    def prepare(self, raw_dir: Path, data_root: Path, progress: Progress) -> list[SequenceInfo]:
        out = []
        for i, (seq, (_, views, calib, splits)) in enumerate(SEQUENCES.items()):
            progress(i / len(SEQUENCES), f"{self.id}: building {seq}")
            src = raw_dir / seq
            gt = pd.concat([read_xu_gt(src / f"view-{v}.txt", v) for v in views], ignore_index=True)
            out.append(
                build_sequence(
                    data_root,
                    self.id,
                    seq,
                    {v: src / f"view-{v}.mp4" for v in views},
                    gt,
                    overlap="overlap",
                    splits=splits,
                    homographies=read_calibration(src / calib, seq),
                    frame_offsets=FRAME_OFFSETS.get(seq),
                    notes="Partial view overlap; frame_offset of Garden1/IP1 estimated from GT (README claims sync).",
                )
            )
        progress(1.0, f"{self.id}: done")
        return out


register_adapter(Campus())
