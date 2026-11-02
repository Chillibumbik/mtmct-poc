"""Global tracks from labelled tracklets (spec §8.8)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mtmct.config import PostprocessCfg
from mtmct.io import GlobalTracks


def postprocess(tracklets: pd.DataFrame, labels: pd.Series, cfg: PostprocessCfg) -> pd.DataFrame:
    """Attach global ids, keep one box per (camera, frame, id) by score, drop short single-camera ids, renumber."""
    df = tracklets.assign(global_id=tracklets["tracklet_id"].map(labels))
    df = df[df["global_id"].notna()]
    df = df.sort_values("score", ascending=False, kind="stable").drop_duplicates(["camera_id", "frame_id", "global_id"])
    if cfg.drop_short_single_camera:
        per_id = df.groupby("global_id")
        long_or_multi = (per_id["camera_id"].transform("nunique") > 1) | (
            per_id["frame_id"].transform("size") >= cfg.min_global_len
        )
        df = df[long_or_multi]
    first_seen = df.sort_values(["ts", "camera_id", "frame_id"], kind="stable")["global_id"].unique()
    df = df.assign(global_id=df["global_id"].map({g: i for i, g in enumerate(first_seen)}).astype(np.int64))
    out = df.sort_values(["camera_id", "frame_id", "global_id"])[list(GlobalTracks.to_schema().columns)]
    return GlobalTracks.validate(out.reset_index(drop=True))
