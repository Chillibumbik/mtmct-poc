"""Camera transition-time windows fitted on ground truth (spec §8.7, `topology.json`)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from mtmct.data.base import SequenceInfo


def fit_topology(
    gt: pd.DataFrame, seq: SequenceInfo, q: tuple[float, float] = (0.05, 0.95), margin_s: float = 2.0
) -> dict[str, Any]:
    """Per person, consecutive visits to different cameras give Δt = next start − previous end per ordered pair.

    Window = [quantile(q0) − margin, quantile(q1) + margin] → `{"pairs": {"c0->c1": [lo, hi]}}`.
    """
    fps = {c.camera_id: c.fps for c in seq.cameras}
    offset = {c.camera_id: c.frame_offset for c in seq.cameras}
    df = gt[["person_id", "camera_id", "frame_id"]].copy()
    df["ts"] = (df["frame_id"] + df["camera_id"].map(offset)) / df["camera_id"].map(fps)
    df = df.sort_values(["person_id", "ts", "camera_id"])
    # A visit = maximal run of one person's observations (in time order) in the same camera.
    new_visit = (df["camera_id"] != df["camera_id"].shift()) | (df["person_id"] != df["person_id"].shift())
    visits = (
        df.assign(visit=new_visit.cumsum())
        .groupby("visit")
        .agg(person_id=("person_id", "first"), camera_id=("camera_id", "first"), t0=("ts", "min"), t1=("ts", "max"))
    )
    samples: dict[str, list[float]] = defaultdict(list)
    for _, v in visits.groupby("person_id"):
        cams, t0, t1 = v["camera_id"].to_numpy(), v["t0"].to_numpy(), v["t1"].to_numpy()
        for k in range(len(v) - 1):
            samples[f"{cams[k]}->{cams[k + 1]}"].append(float(t0[k + 1] - t1[k]))
    pairs = {}
    for key, dts in sorted(samples.items()):
        a, b = np.quantile(dts, q)
        pairs[key] = [float(a) - margin_s, float(b) + margin_s]
    return {"pairs": pairs}
