import numpy as np
import pandas as pd

from mtmct.associate.costs.geometry import Geometry
from mtmct.components import AssocContext


def _ctx(dx: float) -> AssocContext:
    ts = np.arange(10) / 10
    rows = []
    for tid, cam, off in [(1, "c0", 0.0), (10_000_001, "c1", dx)]:
        for t in ts:
            rows.append({"tracklet_id": tid, "camera_id": cam, "ts": t, "world_x": t + off, "world_y": 0.0})
    tr = pd.DataFrame(rows)
    portraits = pd.DataFrame({"tracklet_id": [1, 10_000_001], "camera_id": ["c0", "c1"]})
    return AssocContext(seq=None, tracklets=tr, portraits=portraits, portrait_emb=np.zeros((2, 4)), gallery_emb=[])  # type: ignore[arg-type]


def test_soft_geometry_never_cannot_links():
    far = _ctx(dx=10.0)
    hard = Geometry(Geometry.Params(gate_m=1.5))(far)
    soft = Geometry(Geometry.Params(gate_m=1.5, hard=False))(far)
    assert hard.cannot_link[0, 1] and not soft.cannot_link.any()
    assert soft.dist[0, 1] == 1.0  # still a (maximal) soft penalty


def test_geometry_median_ignores_outlier_frames():
    ctx = _ctx(dx=0.5)
    ctx.tracklets.loc[(ctx.tracklets.tracklet_id == 1) & (ctx.tracklets.ts < 0.25), "world_x"] += 40.0  # 3 bad frames
    cm = Geometry(Geometry.Params(gate_m=1.5))(ctx)
    assert not cm.cannot_link[0, 1] and cm.dist[0, 1] < 0.5
