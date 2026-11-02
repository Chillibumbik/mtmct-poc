import numpy as np
import pytest

from mtmct import registry

N_FRAMES, MISSED = 50, 20
START = np.array([[50.0, 100.0], [300.0, 120.0], [550.0, 90.0]])
VEL = np.array([[4.0, 1.0], [-3.0, 0.5], [2.0, -1.0]])
IMAGE = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)


def _frame_dets(rng: np.random.Generator, t: int) -> np.ndarray:
    xy = START + VEL * t + rng.normal(0, 1.0, START.shape)
    wh = np.tile([40.0, 100.0], (3, 1)) + rng.normal(0, 1.0, (3, 2))
    score = rng.uniform(0.7, 0.95, (3, 1))
    return np.hstack([xy, wh, score]).astype(np.float32)


@pytest.mark.parametrize("name", ["bytetrack", "botsort", "ocsort"])
def test_tracker_keeps_id_on_linear_motion(name):
    tracker = registry.create("tracker", name, {"frame_rate": 10.0})
    rng = np.random.default_rng(0)
    ids_per_person: list[set[int]] = [set() for _ in range(3)]
    for t in range(N_FRAMES):
        dets = _frame_dets(rng, t)
        out = tracker.update(np.zeros((0, 5), np.float32) if t == MISSED else dets, IMAGE)
        assert out.shape[1] == 6 and (out[:, 5] >= 0).all()
        for row in out:
            person = int(np.argmin(np.abs(START[:, 0] + VEL[:, 0] * t - row[0])))
            ids_per_person[person].add(int(row[5]))
    assert all(len(ids) == 1 for ids in ids_per_person), ids_per_person
    assert len({next(iter(s)) for s in ids_per_person}) == 3


def test_tracker_output_is_in_original_pixels_and_reset_restarts_ids():
    tracker = registry.create("tracker", "bytetrack")
    dets = np.array([[100.0, 50.0, 40.0, 120.0, 0.9]], np.float32)
    for _ in range(4):
        out = tracker.update(dets, IMAGE)
    np.testing.assert_allclose(out[0, :5], dets[0], atol=1e-3)
    first_id = out[0, 5]
    tracker.reset()
    for _ in range(4):
        out = tracker.update(dets, IMAGE)
    assert out[0, 5] == first_id == 0


def test_tracker_empty_input():
    tracker = registry.create("tracker", "bytetrack")
    assert tracker.update(np.zeros((0, 5), np.float32), IMAGE).shape == (0, 6)
