import json

import pandas as pd
import pandera.errors
import pytest

from mtmct.io import Detections, GlobalTracks, config_hash, read_table, stage_dir, write_table


def _dets(n=3):
    return pd.DataFrame(
        {
            "camera_id": ["c0"] * n,
            "frame_id": range(n),
            "ts": [i / 10 for i in range(n)],
            "x": [1.0] * n,
            "y": [2.0] * n,
            "w": [10.0] * n,
            "h": [20.0] * n,
            "score": [0.9] * n,
        }
    )


def test_write_read_roundtrip_validates(tmp_path):
    p = tmp_path / "d.parquet"
    write_table(_dets(), p, Detections)
    back = read_table(p, Detections)
    assert len(back) == 3 and back["frame_id"].tolist() == [0, 1, 2]


def test_write_rejects_bad_schema(tmp_path):
    bad = _dets().assign(score=1.5)
    with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
        write_table(bad, tmp_path / "d.parquet", Detections)
    with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
        write_table(_dets().assign(extra=1), tmp_path / "d.parquet", Detections)


def test_global_tracks_csv_roundtrip_with_null_world(tmp_path):
    df = pd.DataFrame(
        {
            "camera_id": ["c0"],
            "global_id": [3],
            "frame_id": [7],
            "x": [1.0],
            "y": [1.0],
            "w": [5.0],
            "h": [9.0],
            "world_x": [None],
            "world_y": [None],
        }
    )
    write_table(df, tmp_path / "g.csv", GlobalTracks)
    assert read_table(tmp_path / "g.csv", GlobalTracks)["global_id"].tolist() == [3]


def test_config_hash_stable_and_order_independent():
    assert config_hash({"a": 1, "b": {"c": 2}}) == config_hash({"b": {"c": 2}, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    assert len(config_hash({})) == 8


def test_stage_dir_atomic_commit(tmp_path):
    sd = stage_dir(tmp_path, "detect", {"k": 1})
    assert not sd.done
    work = sd.open()
    (work / "x.txt").write_text("hi")
    assert not sd.path.exists()  # nothing visible before commit
    sd.commit({"duration_s": 1.0})
    assert sd.done and (sd.path / "x.txt").read_text() == "hi"
    assert json.loads((sd.path / "manifest.json").read_text())["stage"] == "detect"
    assert not sd.tmp.exists()


def test_stage_dir_open_discards_leftovers(tmp_path):
    sd = stage_dir(tmp_path, "track", {"k": 1})
    (sd.open() / "partial").write_text("x")
    sd2 = stage_dir(tmp_path, "track", {"k": 1})
    assert not any(sd2.open().iterdir())
