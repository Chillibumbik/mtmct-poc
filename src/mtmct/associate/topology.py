"""Camera transition-time windows fitted on ground truth (spec §8.7, `topology.json`)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from mtmct.data.base import SequenceInfo


def fit_topology(
    gt: pd.DataFrame,
    seq: SequenceInfo,
    q: tuple[float, float] = (0.05, 0.95),
    margin_s: float = 2.0,
    max_gap_s: float = 2.0,
) -> dict[str, Any]:
    """Transition windows from GT: Δt = next visit start − previous visit end, per ordered camera pair.

    A visit = continuous presence of a person in one camera (breaks longer than `max_gap_s` start a new one).
    A visit's predecessor is the person's visit that ended last before it started; views overlapping in time
    (a person seen by two cameras at once) are not transitions. Window = [quantile(q0) − margin,
    quantile(q1) + margin] → `{"pairs": {"c0->c1": [lo, hi]}}`.
    """
    fps = {c.camera_id: c.fps for c in seq.cameras}
    offset = {c.camera_id: c.frame_offset for c in seq.cameras}
    df = gt[["person_id", "camera_id", "frame_id"]].copy()
    df["ts"] = (df["frame_id"] + df["camera_id"].map(offset)) / df["camera_id"].map(fps)
    df = df.sort_values(["person_id", "camera_id", "ts"])
    new_visit = (
        (df["person_id"] != df["person_id"].shift())
        | (df["camera_id"] != df["camera_id"].shift())
        | (df["ts"].diff() > max_gap_s)
    )
    visits = (
        df.assign(visit=new_visit.cumsum())
        .groupby("visit")
        .agg(person_id=("person_id", "first"), camera_id=("camera_id", "first"), t0=("ts", "min"), t1=("ts", "max"))
        .sort_values(["person_id", "t0"])
    )
    samples: dict[str, list[float]] = defaultdict(list)
    for _, v in visits.groupby("person_id"):
        cams, t0, t1 = v["camera_id"].to_numpy(), v["t0"].to_numpy(), v["t1"].to_numpy()
        for k in range(1, len(v)):
            prev = int(np.argmax(t1[:k]))  # the earlier-started visit that ended last
            if t1[prev] < t0[k] and cams[prev] != cams[k]:
                samples[f"{cams[prev]}->{cams[k]}"].append(float(t0[k] - t1[prev]))
    pairs = {}
    for key, dts in sorted(samples.items()):
        a, b = np.quantile(dts, q)
        pairs[key] = [float(a) - margin_s, float(b) + margin_s]
    return {"pairs": pairs}
