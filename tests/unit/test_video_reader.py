import numpy as np

from mtmct.data.base import load_sequence
from mtmct.video.reader import iter_frames, selected_frame_ids


def test_iter_frames_subsamples_to_target_fps(tiny_sequence):
    cam = tiny_sequence.camera("c0")
    frames = list(iter_frames(tiny_sequence, cam, target_fps=10))
    assert len(frames) == 20
    ts = np.array([f.ts for f in frames])
    assert np.all(np.diff(ts) > 0) and np.allclose(np.diff(ts), 0.1, atol=0.021)
    assert [f.frame_id for f in frames] == selected_frame_ids(cam, cam.n_frames, 10).tolist()
    assert abs(int(frames[3].image.mean()) - frames[3].frame_id) <= 2  # decoded the right frame


def test_iter_frames_time_range_on_common_clock(tiny_sequence):
    cam = tiny_sequence.camera("c1")  # offset 5 frames = 0.2 s
    frames = list(iter_frames(tiny_sequence, cam, target_fps=25, t_range=(0.5, 1.0)))
    assert frames[0].frame_id == 8 and abs(frames[0].ts - 0.52) < 1e-9
    assert all(0.5 <= f.ts < 1.0 for f in frames)


def test_load_sequence_roundtrip(tiny_sequence):
    seq = load_sequence(tiny_sequence.root.parents[1], "toy", "seq")
    assert [c.camera_id for c in seq.cameras] == ["c0", "c1"]
    assert seq.split_range("tune") == (0.0, 1.0)
    assert not seq.has_gt and seq.gt() is None
