import json
import time

import numpy as np

from mtmct.runtime.profiling import ResourceSampler, StageTimer, runtime_report


def test_sampler_records_peak_and_limit():
    with ResourceSampler(interval=0.05) as s:
        block = np.ones(20_000_000, dtype=np.uint8)  # ~20 MB
        time.sleep(0.15)
        del block
    assert s.peak_rss_gb > 0.02
    assert s.exceeded(0.001) and not s.exceeded(1000.0)
    assert s.mean_cpu >= 0.0


def test_stage_timer_and_report_shape():
    timer = StageTimer()
    with timer.stage("detect", unit="frames") as st:
        for _ in range(4):
            st.add(2, 0.02)  # a batch of 2 frames in 20 ms → 10 ms per frame
    timer.cached("reid")
    rep = runtime_report(timer, sampler=None, wall_time_s=2.0, frames_processed=8, cache_bytes=123)
    det = rep["stages"]["detect"]
    assert det["items"] == 8 and not det["cached"] and det["fps"] > 0
    assert abs(det["p50_ms"] - 10.0) < 1e-6 and abs(det["p95_ms"] - 10.0) < 1e-6
    assert rep["stages"]["reid"] == {"wall_s": 0.0, "items": 0, "items_per_s": None, "p50_ms": None,
                                     "p95_ms": None, "cached": True}  # fmt: skip
    assert rep["fps"] == 4.0 and rep["wall_time_s"] == 2.0 and rep["cache_bytes"] == 123
    assert {"peak_rss_gb", "mean_cpu_pct", "frames_processed"} <= rep.keys()
    json.dumps(rep, allow_nan=False)
