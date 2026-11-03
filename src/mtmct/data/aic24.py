"""AI City 2024 Track 1, val scene_041 (NVIDIA PhysicalAI-SmartSpaces): 10 synthetic 1080p @ 30 fps cameras."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mtmct.data.base import DatasetMeta, Progress, SequenceInfo, register_adapter
from mtmct.data.common import build_sequence

REPO = "nvidia/PhysicalAI-SmartSpaces"
SCENE = "MTMC_Tracking_2024/val/scene_041"
CAMERAS = [f"camera_{n:04d}" for n in range(361, 371)]
# `scene_041` keeps the first four cameras (bounded compute); `scene_041_all` has every camera.
SEQUENCES = {"scene_041": CAMERAS[:4], "scene_041_all": CAMERAS}
SPLITS = {"tune": (0.0, 0.5), "test": (0.5, 1.0), "all": (0.0, 1.0)}
_GT_COLUMNS = ["camera", "person_id", "frame_id", "x", "y", "w", "h", "xworld", "yworld"]


def read_gt(path: Path, cameras: list[str]) -> pd.DataFrame:
    """`camera_id obj_id frame_id x y w h xworld yworld` (frame ids are 0-based, as the videos) → canonical boxes."""
    raw = pd.read_csv(path, sep=" ", header=None, names=_GT_COLUMNS)
    raw["camera_id"] = "camera_" + raw["camera"].astype(str).str.zfill(4)
    raw = raw[raw["camera_id"].isin(cameras) & (raw["w"] > 0) & (raw["h"] > 0)]
    return raw[["camera_id", "frame_id", "person_id", "x", "y", "w", "h"]].reset_index(drop=True)


def read_homography(path: Path) -> np.ndarray:
    """`calibration.json` stores the ground (metres) → image homography; the pipeline needs its inverse."""
    return np.linalg.inv(np.asarray(json.loads(path.read_text())["homography matrix"], dtype=float))


class Aic24:
    id = "aic24"

    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            id=self.id,
            title="AI City 2024 Track 1 (scene_041)",
            licence="CC BY 4.0 (NVIDIA PhysicalAI-SmartSpaces)",
            source_url="https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces",
            overlap="overlap",
            real=False,
            n_cameras="4 / 10",
            download_gb=2.1,
            sequences=list(SEQUENCES),
            description=(
                "Синтетическая сцена (Omniverse) из валидации AI City 2024: 10 камер 1080p, 30 к/с, около 13 минут, "
                "28 человек, гомографии на плоскость земли. scene_041 - первые 4 камеры (быстрее), scene_041_all - все "
                "10. Первая половина времени - tune, вторая - test. Глубина и другие сцены не скачиваются."
            ),
        )

    def fetch(self, raw_dir: Path, progress: Progress) -> None:
        from huggingface_hub import hf_hub_download

        files = [f"{SCENE}/ground_truth.txt"]
        for cam in CAMERAS:
            files += [f"{SCENE}/{cam}/calibration.json", f"{SCENE}/{cam}/video.mp4"]
        for i, name in enumerate(files):
            progress(i / len(files), f"{self.id}: {name.removeprefix(SCENE + '/')}")
            hf_hub_download(REPO, name, repo_type="dataset", local_dir=raw_dir)
        progress(1.0, f"{self.id}: done")

    def prepare(self, raw_dir: Path, data_root: Path, progress: Progress) -> list[SequenceInfo]:
        src = raw_dir / SCENE
        out = []
        for i, (name, cameras) in enumerate(SEQUENCES.items()):
            progress(i / len(SEQUENCES), f"{self.id}: building {name}")
            out.append(
                build_sequence(
                    data_root,
                    self.id,
                    name,
                    {cam: src / cam / "video.mp4" for cam in cameras},
                    read_gt(src / "ground_truth.txt", cameras),
                    overlap="overlap",
                    splits=SPLITS,
                    homographies={cam: read_homography(src / cam / "calibration.json") for cam in cameras},
                    notes="Synthetic, synchronised cameras. Homographies: image → ground, metres.",
                )
            )
        progress(1.0, f"{self.id}: done")
        return out


register_adapter(Aic24())
