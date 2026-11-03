import numpy as np
import pandas as pd

from mtmct.associate.run import associate, build_context
from mtmct.config import AssociateCfg, CostSpec


def test_zero_weight_cost_is_fully_disabled():
    """Two co-visible tracklets of one person 10 m apart: hard geometry forbids the link; weight 0 must not."""
    ts = np.arange(10) / 10
    rows = [
        {"tracklet_id": tid, "camera_id": cam, "ts": t, "world_x": t + off, "world_y": 0.0}
        for tid, cam, off in [(1, "c0", 0.0), (10_000_001, "c1", 10.0)]
        for t in ts
    ]
    portraits = pd.DataFrame({"tracklet_id": [1, 10_000_001], "camera_id": ["c0", "c1"],
                              "t_start": [0.0, 0.0], "t_end": [0.9, 0.9]})
    emb = np.tile(np.eye(4)[:1], (2, 1))  # identical appearance
    ctx = build_context(None, pd.DataFrame(rows), portraits, emb, [emb[:1], emb[:1]])  # type: ignore[arg-type]
    on = AssociateCfg(costs={"appearance": CostSpec(), "geometry": CostSpec(weight=1.0, params={"hard": True})})
    off = AssociateCfg(costs={"appearance": CostSpec(), "geometry": CostSpec(weight=0.0, params={"hard": True})})
    assert associate(ctx, on).nunique() == 2
    assert associate(ctx, off).nunique() == 1
