"""ReID retrieval sanity: CMC rank-1/5 and mAP, query = tracklet portrait, gallery = tracklets of other cameras."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def tracklet_labels(matches: pd.DataFrame) -> pd.Series:
    """tracklet_id → majority GT person of its matched boxes (ties → smaller person id)."""
    cnt = matches[["tracklet_id", "person_id"]].value_counts().reset_index(name="n")
    best = cnt.sort_values(["tracklet_id", "n", "person_id"], ascending=[True, False, True]).drop_duplicates(
        "tracklet_id"
    )
    return best.set_index("tracklet_id")["person_id"]


def reid_sanity(emb: np.ndarray, labels: np.ndarray, cams: np.ndarray) -> dict[str, Any]:
    """Cosine retrieval; queries without a positive in other cameras are skipped. None if no query is valid."""
    x = np.asarray(emb, dtype=np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    sim = x @ x.T
    r1, r5, aps = [], [], []
    for q in range(len(x)):
        g = cams != cams[q]
        hit = (labels[g] == labels[q])[np.argsort(-sim[q, g], kind="stable")]
        if not hit.any():
            continue
        r1.append(hit[0])
        r5.append(hit[:5].any())
        pos = np.flatnonzero(hit)
        aps.append(np.mean(np.arange(1, len(pos) + 1) / (pos + 1)))
    if not aps:
        return {"rank1": None, "rank5": None, "mAP": None, "n_queries": 0}
    return {"rank1": float(np.mean(r1)), "rank5": float(np.mean(r5)), "mAP": float(np.mean(aps)), "n_queries": len(aps)}
