from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import write_video

from mtmct.data import chirla
from mtmct.data.base import load_adapters, load_sequence

FIXTURES = Path(__file__).parents[1] / "fixtures" / "data" / "chirla"


def test_chirla_gt_parse():
    gt = chirla.read_gt(FIXTURES / "camera_5.json", "camera_5")
    assert list(gt.columns) == ["camera_id", "frame_id", "person_id", "x", "y", "w", "h"]
    first = gt.iloc[0]  # raw: frame "3533": {"id": 18, "BboxP": [513, 355, 551, 451]}; frame numbers are 1-based
    assert (first.frame_id, first.person_id, first.x, first.y, first.w, first.h) == (3532, 18, 513, 355, 38, 96)
    assert len(gt) == 4
    assert chirla.read_gt(FIXTURES / "camera_6.json", "camera_6").empty  # a camera without people stays valid


def _raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    for seq in chirla.SEQUENCES:
        for n in (3, 5, 6):
            stem = f"camera_{n}_2023-06-02-12:03:33"
            write_video(raw / "videos" / seq / f"{stem}.avi", 3600 if n == 5 else 6, 30.0, size=(54, 36))
            (raw / "annotations" / seq).mkdir(parents=True, exist_ok=True)
            (raw / "annotations" / seq / f"{stem}.json").write_text((FIXTURES / f"camera_{n}.json").read_text())
    return raw


def test_chirla_prepare_writes_splits_and_topology(tmp_path):
    raw, data = _raw(tmp_path), tmp_path / "data"
    tune, test = chirla.Chirla().prepare(raw, data, lambda f, m: None)
    assert (tune.sequence, test.sequence) == ("seq_001", "seq_002")
    tune = load_sequence(data, "chirla", "seq_001")
    assert tune.overlap == "nonoverlap" and tune.meta.splits["tune"] == (0.0, 1.0)
    assert load_sequence(data, "chirla", "seq_002").meta.splits["test"] == (0.0, 1.0)
    assert [c.camera_id for c in tune.cameras] == ["camera_3", "camera_5", "camera_6"]
    assert tune.camera("camera_5").H() is None and tune.camera("camera_5").fps == 30.0
    gt = tune.gt()
    assert gt is not None and gt.world_x.isna().all()
    # the same topology file, learnt on the tune sequence, lands in every sequence directory
    assert tune.topology() == load_sequence(data, "chirla", "seq_002").topology()
    assert tune.topology() is not None and "pairs" in tune.topology()


def test_chirla_fetch_without_access_explains_what_to_do(tmp_path, monkeypatch):
    from huggingface_hub.errors import GatedRepoError

    def deny(*args, **kwargs):
        raise GatedRepoError("401", response=SimpleNamespace(headers={}, request=None, status_code=401))

    monkeypatch.setattr("huggingface_hub.snapshot_download", deny)
    with pytest.raises(RuntimeError, match="hf auth login"):
        chirla.Chirla().fetch(tmp_path, lambda f, m: None)


def test_chirla_adapter_registered():
    meta = load_adapters()["chirla"].meta()
    assert meta.overlap == "nonoverlap" and meta.requires_auth and meta.sequences == ["seq_001", "seq_002"]


@pytest.mark.data
def test_chirla_real_prepared_sequences():
    from mtmct.config import paths

    for sequence in chirla.SEQUENCES:
        if not (paths().data / "chirla" / sequence / "cameras.json").exists():
            pytest.skip("chirla is not prepared")
        seq = load_sequence(paths().data, "chirla", sequence)
        assert len(seq.cameras) == 7 and all((seq.root / c.source).exists() for c in seq.cameras)
        assert all((c.width, c.height, c.fps) == (1080, 720, 30.0) for c in seq.cameras)
        gt = seq.gt()
        assert gt is not None and gt.person_id.nunique() > 3
        for cam in seq.cameras:
            g = gt[gt.camera_id == cam.camera_id]
            assert g.empty or g.frame_id.max() < cam.n_frames
        assert seq.topology() is not None
