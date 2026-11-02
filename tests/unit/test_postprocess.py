import numpy as np
import pandas as pd

from mtmct.config import PostprocessCfg
from mtmct.io import GlobalTracks
from mtmct.postprocess.postprocess import postprocess


def _rows(tid, cam, frames, score=0.9):
    f = np.asarray(frames)
    return pd.DataFrame(
        {
            "camera_id": cam, "frame_id": f, "ts": f / 10.0, "tracklet_id": tid,
            "x": 1.0, "y": 2.0, "w": 10.0, "h": 20.0, "score": score, "quality": 0.5,
            "foot_x": 6.0, "foot_y": 22.0, "world_x": np.nan, "world_y": np.nan, "emb_idx": -1,
        }
    )  # fmt: skip


def test_postprocess_dedup_same_frame():
    # Tracklets 1 and 2 of one camera overlap in frames 5..9 but got the same global id.
    tr = pd.concat([_rows(1, "c0", range(10), 0.6), _rows(2, "c0", range(5, 15), 0.8)], ignore_index=True)
    out = postprocess(tr, pd.Series({1: 3, 2: 3}), PostprocessCfg(min_global_len=1))
    GlobalTracks.validate(out)
    assert not out.duplicated(["camera_id", "frame_id", "global_id"]).any()
    assert len(out) == 15 and (out.global_id == 0).all()
    overlap = out[out.frame_id.between(5, 9)]
    assert (overlap.tracklet_id == 2).all()  # higher score kept


def test_postprocess_drops_short_single_camera_and_renumbers():
    tr = pd.concat(
        [
            _rows(10, "c0", range(20, 40)),  # long, single camera → kept
            _rows(11, "c0", range(0, 5)),  # short, single camera → dropped
            _rows(12, "c0", range(5, 9)),  # short but its id is also seen in c1 → kept
            _rows(13, "c1", range(10, 13)),
            _rows(14, "c1", range(50, 60)),  # not in labels → dropped
        ],
        ignore_index=True,
    )
    labels = pd.Series({10: 7, 11: 5, 12: 9, 13: 9})
    out = postprocess(tr, labels, PostprocessCfg(min_global_len=10, drop_short_single_camera=True))
    assert set(out.tracklet_id) == {10, 12, 13}
    first = out.sort_values("frame_id").drop_duplicates("global_id")
    assert dict(zip(first.tracklet_id, first.global_id, strict=True)) == {12: 0, 10: 1}  # order of first appearance
    assert out.world_x.isna().all()

    keep_all = postprocess(tr, labels, PostprocessCfg(min_global_len=10, drop_short_single_camera=False))
    assert set(keep_all.tracklet_id) == {10, 11, 12, 13}


def test_postprocess_empty():
    tr = _rows(1, "c0", range(3))
    out = postprocess(tr.iloc[:0], pd.Series(dtype=np.int64), PostprocessCfg())
    assert len(out) == 0 and list(out.columns) == list(GlobalTracks.to_schema().columns)
