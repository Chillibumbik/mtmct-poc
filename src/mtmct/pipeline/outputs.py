"""Run directory files (spec §7): config.yaml, git_sha, global_tracks.csv, per_camera/*.txt, *.json."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mtmct.config import RunConfig
from mtmct.data.base import SequenceInfo
from mtmct.eval.report import json_safe
from mtmct.io import GlobalTracks, write_table


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"


def write_run_header(run_dir: Path, cfg: RunConfig) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    (run_dir / "git_sha").write_text(git_sha() + "\n")


def write_tracks(run_dir: Path, tracks: pd.DataFrame, seq: SequenceInfo) -> None:
    """`global_tracks.csv` + one MOT file per camera (`frame+1,id,x,y,w,h,1,-1,-1,-1`)."""
    write_table(tracks, run_dir / "global_tracks.csv", GlobalTracks)
    mot_dir = run_dir / "per_camera"
    mot_dir.mkdir(exist_ok=True)
    for cam in seq.cameras:
        t = tracks[tracks["camera_id"] == cam.camera_id].sort_values(["frame_id", "global_id"])
        lines = [
            f"{f + 1},{g},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
            for f, g, x, y, w, h in zip(t.frame_id, t.global_id, t.x, t.y, t.w, t.h, strict=True)
        ]
        (mot_dir / f"{cam.camera_id}.txt").write_text("".join(line + "\n" for line in lines))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(obj), indent=2, allow_nan=False))
