"""CHIRLA (Univ. Alicante): 7 mostly non-overlapping 1080x720 @ 30 fps indoor cameras, person IDs shared across cameras.

Raw layout: `videos/seq_XXX/camera_N_<timestamp>.avi`, `annotations/seq_XXX/camera_N_<timestamp>.json`
with `{"<frame, 1-based>": [{"id": person, "BboxP": [x1, y1, x2, y2]}]}`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mtmct.associate.topology import fit_topology
from mtmct.data.base import DatasetMeta, Progress, SequenceInfo, register_adapter
from mtmct.data.common import build_sequence

REPO = "bdager/CHIRLA"
# sequence → split: the first is tuned on (and gives the transition windows), the second is the final test
SEQUENCES = {"seq_001": "tune", "seq_002": "test"}
ACCESS_MESSAGE = "Нужен доступ: примите условия на huggingface.co/datasets/bdager/CHIRLA и выполните `hf auth login`"


def read_gt(path: Path, camera_id: str) -> pd.DataFrame:
    """One annotation JSON → canonical boxes (`frame_id` 0-based: JSON frame numbers start at 1)."""
    rows = [
        (camera_id, int(frame) - 1, box["id"], *box["BboxP"])
        for frame, boxes in json.loads(path.read_text()).items()
        for box in boxes
    ]
    df = pd.DataFrame(rows, columns=["camera_id", "frame_id", "person_id", "x1", "y1", "x2", "y2"])
    df = df[(df["x2"] > df["x1"]) & (df["y2"] > df["y1"])]
    return pd.DataFrame(
        {
            "camera_id": df["camera_id"],
            "frame_id": df["frame_id"].astype("int32"),
            "person_id": df["person_id"].astype("int64"),
            "x": df["x1"].astype("float32"),
            "y": df["y1"].astype("float32"),
            "w": (df["x2"] - df["x1"]).astype("float32"),
            "h": (df["y2"] - df["y1"]).astype("float32"),
        }
    ).reset_index(drop=True)


def _camera_files(raw_dir: Path, sequence: str) -> dict[str, tuple[Path, Path]]:
    """{camera_id: (video, annotation)}; camera_id is the `camera_N` prefix of the file names."""
    out = {}
    for video in sorted((raw_dir / "videos" / sequence).glob("camera_*.avi")):
        camera = "_".join(video.stem.split("_")[:2])
        (annotation,) = (raw_dir / "annotations" / sequence).glob(f"{camera}_*.json")
        out[camera] = (video, annotation)
    return out


class Chirla:
    id = "chirla"

    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            id=self.id,
            title="CHIRLA",
            licence="CC BY 4.0 (Dominguez-Dager et al., Sci. Data 2026); не для слежения за реальными людьми",
            source_url="https://huggingface.co/datasets/bdager/CHIRLA",
            overlap="nonoverlap",
            real=True,
            n_cameras="7",
            download_gb=0.7,
            requires_auth=True,
            sequences=list(SEQUENCES),
            description=(
                "Семь камер в офисных коридорах и лабораториях университета Аликанте, 1080x720, 30 к/с, 22 человека "
                "с общими ID между камерами, камеры почти не перекрываются. Калибровки нет. Берутся две "
                "последовательности: seq_001 для подбора параметров и окон переходов (tune), seq_002 для оценки "
                "(test). Доступ закрытый: нужно принять условия на Hugging Face."
            ),
        )

    def fetch(self, raw_dir: Path, progress: Progress) -> None:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

        progress(0.0, f"{self.id}: {', '.join(SEQUENCES)}")
        patterns = [f"{kind}/{seq}/*" for kind in ("videos", "annotations") for seq in SEQUENCES]
        try:
            snapshot_download(REPO, repo_type="dataset", local_dir=raw_dir, allow_patterns=patterns)
        except (GatedRepoError, HfHubHTTPError) as e:
            raise RuntimeError(ACCESS_MESSAGE) from e
        progress(1.0, f"{self.id}: done")

    def prepare(self, raw_dir: Path, data_root: Path, progress: Progress) -> list[SequenceInfo]:
        out = []
        for i, (sequence, split) in enumerate(SEQUENCES.items()):
            progress(i / len(SEQUENCES), f"{self.id}: building {sequence}")
            files = _camera_files(raw_dir, sequence)
            gt = pd.concat([read_gt(ann, cam) for cam, (_, ann) in files.items()], ignore_index=True)
            out.append(
                build_sequence(
                    data_root,
                    self.id,
                    sequence,
                    {cam: video for cam, (video, _) in files.items()},
                    gt,
                    overlap="nonoverlap",
                    splits={split: (0.0, 1.0), "all": (0.0, 1.0)},
                    notes="No calibration. topology.json is fitted on the tune sequence.",
                )
            )
        # transition windows are learnt on the tune sequence only and shared by all sequences
        tune = next(s for s in out if SEQUENCES[s.sequence] == "tune")
        tune_gt = tune.gt()
        assert tune_gt is not None
        topology = json.dumps(fit_topology(tune_gt, tune), indent=2)
        for seq in out:
            (seq.root / "topology.json").write_text(topology)
        progress(1.0, f"{self.id}: done")
        return out


register_adapter(Chirla())
