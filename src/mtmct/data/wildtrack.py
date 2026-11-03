"""WILDTRACK (EPFL/ETH): 7 overlapping 1080p cameras, 400 annotated frames at 2 fps, person IDs shared by all views.

Raw: one zip with `Image_subsets/C1..C7/<frame>.png` (undistorted), `annotations_positions/<frame>.json` and
`calibrations/{extrinsic,intrinsic_zero}/*.xml`. Only the annotated frames are used as the camera source.
"""

from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mtmct.data.base import CameraInfo, DatasetMeta, Progress, SequenceInfo, SequenceMeta, register_adapter
from mtmct.data.fetch import download
from mtmct.io import GroundTruth, write_table

URL = "https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip"
ZIP_NAME = "Wildtrack_dataset_full.zip"
ZIP_SIZE = 6807496358
ROOT = "Wildtrack_dataset"
SEQUENCE = "main"
CAMERAS = [f"C{i}" for i in range(1, 8)]
CALIBRATION_NAMES = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]  # C1..C7
FPS = 2.0  # the annotated frames are every 5th frame of a 10 fps stream
SPLITS = {"tune": (0.0, 0.9), "test": (0.9, 1.0), "all": (0.0, 1.0)}
GRID_W, GRID_STEP_M, GRID_ORIGIN_M = 480, 0.025, (-3.0, -9.0)  # positionID → ground point (README)


def position_to_world(position_id: np.ndarray) -> np.ndarray:
    """positionID on the 480-wide, X-first, 2.5 cm grid → ground point (x, y) in metres."""
    pid = np.asarray(position_id)
    return np.c_[GRID_ORIGIN_M[0] + GRID_STEP_M * (pid % GRID_W), GRID_ORIGIN_M[1] + GRID_STEP_M * (pid // GRID_W)]


def _numbers(xml_text: str, name: str) -> np.ndarray:
    """Values of `<name>`: either plain text (extrinsics) or an opencv-matrix with a <data> child (intrinsics)."""
    node = ET.fromstring(xml_text).find(name)
    if node is None:
        raise ValueError(f"calibration XML has no <{name}>")
    return np.array((node.findtext("data") or node.text or "").split(), dtype=float)


def homography_img2world(extrinsic_xml: str, intrinsic_xml: str) -> np.ndarray:
    """Image pixel → ground plane Z=0, metres. `tvec` is in the units of the annotation grid (centimetres)."""
    R, _ = cv2.Rodrigues(_numbers(extrinsic_xml, "rvec"))
    K = _numbers(intrinsic_xml, "camera_matrix").reshape(3, 3)
    world_to_image = K @ np.c_[R[:, 0], R[:, 1], _numbers(extrinsic_xml, "tvec") / 100.0]
    return np.linalg.inv(world_to_image)


def parse_annotations(frames: list[list[dict]]) -> pd.DataFrame:
    """Per-frame annotation lists (index = annotated frame) → canonical boxes.

    Views that do not see the person are all -1; visible boxes may extend past the image border (kept as is).
    """
    rows = []
    for frame_id, people in enumerate(frames):
        for p in people:
            world = position_to_world(np.array([p["positionID"]]))[0]
            for v in p["views"]:
                if v["xmax"] > max(v["xmin"], 0) and v["ymax"] > max(v["ymin"], 0):
                    rows.append(
                        (
                            f"C{v['viewNum'] + 1}",
                            frame_id,
                            p["personID"],
                            v["xmin"],
                            v["ymin"],
                            v["xmax"] - v["xmin"],
                            v["ymax"] - v["ymin"],
                            world[0],
                            world[1],
                        )
                    )
    cols = ["camera_id", "frame_id", "person_id", "x", "y", "w", "h", "world_x", "world_y"]
    return pd.DataFrame(rows, columns=cols)


def _open_zip(path: Path) -> zipfile.ZipFile:
    """The official zip stores the offset of some members 2**32 too high; fix those entries before reading."""
    z = zipfile.ZipFile(path)
    for info in z.infolist():
        z.fp.seek(info.header_offset)  # type: ignore[union-attr]
        if z.fp.read(4) != b"PK\x03\x04":  # type: ignore[union-attr]
            info.header_offset -= 1 << 32
    return z


def _read(z: zipfile.ZipFile, name: str) -> bytes:
    """Member bytes read straight from the local header (zipfile's own reader rejects the corrected offsets)."""
    info = z.getinfo(name)
    fp = z.fp
    assert fp is not None
    fp.seek(info.header_offset + 26)
    name_len, extra_len = struct.unpack("<HH", fp.read(4))
    fp.seek(name_len + extra_len, 1)
    data = fp.read(info.compress_size)
    return data if info.compress_type == zipfile.ZIP_STORED else zlib.decompress(data, -15)


class Wildtrack:
    id = "wildtrack"

    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            id=self.id,
            title="WILDTRACK",
            licence="Явной лицензии нет: исследовательское использование, цитировать Chavdarova et al. 2018",
            source_url="https://www.epfl.ch/labs/cvlab/data/data-wildtrack/",
            overlap="overlap",
            real=True,
            n_cameras="7",
            download_gb=6.8,
            sequences=[SEQUENCE],
            description=(
                "Семь сильно перекрывающихся камер 1080p, площадь ETH Zurich, до 40 человек в кадре. Размечено "
                "400 кадров с частотой 2 к/с (около 200 с), у каждого человека один ID во всех видах и кадрах. Для "
                "сравнения с литературой: первые 90 % кадров - tune, последние 10 % - test."
            ),
        )

    def fetch(self, raw_dir: Path, progress: Progress) -> None:
        download(URL, raw_dir / ZIP_NAME, progress, expected_size=ZIP_SIZE)

    def prepare(self, raw_dir: Path, data_root: Path, progress: Progress) -> list[SequenceInfo]:
        root = data_root / self.id / SEQUENCE
        with _open_zip(raw_dir / ZIP_NAME) as z:
            stems = sorted(Path(n).stem for n in z.namelist() if n.startswith(f"{ROOT}/annotations_positions/0"))
            frames = [json.loads(_read(z, f"{ROOT}/annotations_positions/{s}.json")) for s in stems]
            homographies = {}
            for cam, name in zip(CAMERAS, CALIBRATION_NAMES, strict=True):
                extr = _read(z, f"{ROOT}/calibrations/extrinsic/extr_{name}.xml").decode()
                intr = _read(z, f"{ROOT}/calibrations/intrinsic_zero/intr_{name}.xml").decode()
                homographies[cam] = homography_img2world(extr, intr)
            total = len(CAMERAS) * len(stems)
            for c, cam in enumerate(CAMERAS):
                (root / "frames" / cam).mkdir(parents=True, exist_ok=True)
                for i, stem in enumerate(stems):
                    dst = root / "frames" / cam / f"{i:06d}.png"
                    info = z.getinfo(f"{ROOT}/Image_subsets/{cam}/{stem}.png")
                    if not (dst.exists() and dst.stat().st_size == info.file_size):
                        dst.write_bytes(_read(z, info.filename))
                    if i % 50 == 0:
                        progress((c * len(stems) + i) / total, f"{self.id}: {cam} frame {i}/{len(stems)}")
        first = cv2.imread(str(root / "frames" / CAMERAS[0] / "000000.png"))
        assert first is not None
        height, width = first.shape[:2]
        cameras = [
            CameraInfo(
                camera_id=cam,
                source=f"frames/{cam}",
                fps=FPS,
                width=width,
                height=height,
                n_frames=len(stems),
                homography_img2world=homographies[cam].tolist(),
            )
            for cam in CAMERAS
        ]
        gt = parse_annotations(frames)
        write_table(gt, root / "gt.parquet", GroundTruth)
        seq = SequenceInfo(
            dataset=self.id,
            sequence=SEQUENCE,
            root=root,
            cameras=cameras,
            meta=SequenceMeta(
                overlap="overlap",
                splits=SPLITS,
                duration_s=len(stems) / FPS,
                n_persons=int(gt["person_id"].nunique()),
                notes="Annotated frames only (2 fps); world = positionID grid, metres.",
            ),
        )
        seq.save()
        progress(1.0, f"{self.id}: done")
        return [seq]


register_adapter(Wildtrack())
