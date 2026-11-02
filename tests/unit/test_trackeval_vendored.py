import numpy as np
import trackeval


def test_trackeval_imports_and_perfect_score():
    """Vendored TrackEval works with numpy>=2 (patched aliases): identical GT and prediction → HOTA = 1."""
    n_t = 5
    boxes = [np.array([[0, 0, 10, 10], [20, 20, 10, 10]], dtype=float) for _ in range(n_t)]
    ids = [np.array([0, 1]) for _ in range(n_t)]
    data = {
        "num_timesteps": n_t,
        "num_gt_ids": 2,
        "num_tracker_ids": 2,
        "num_gt_dets": 2 * n_t,
        "num_tracker_dets": 2 * n_t,
        "gt_ids": ids,
        "tracker_ids": ids,
        "similarity_scores": [np.eye(2) for _ in range(n_t)],
    }
    hota = trackeval.metrics.HOTA().eval_sequence(data)
    ident = trackeval.metrics.Identity().eval_sequence(data)
    assert np.isclose(hota["HOTA"].mean(), 1.0)
    assert np.isclose(ident["IDF1"], 1.0)
    del boxes
