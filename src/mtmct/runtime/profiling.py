"""Run resource profiling (spec §9.6): RSS/CPU sampler, per-stage timings, `runtime.json` payload."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psutil

GB = 1024**3


class ResourceSampler:
    """Background thread sampling RSS of this process tree and CPU% of this process every `interval` seconds."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.peak_rss_gb = 0.0
        self._cpu: list[float] = []
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _rss_gb(self) -> float:
        rss = self._proc.memory_info().rss
        for child in self._proc.children(recursive=True):
            with suppress(psutil.Error):
                rss += child.memory_info().rss
        return rss / GB

    def sample(self) -> float:
        rss = self._rss_gb()
        self.peak_rss_gb = max(self.peak_rss_gb, rss)
        return rss

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()
            self._cpu.append(self._proc.cpu_percent())

    @property
    def mean_cpu(self) -> float:
        return float(np.mean(self._cpu)) if self._cpu else 0.0

    def exceeded(self, limit_gb: float) -> bool:
        self.sample()
        return self.peak_rss_gb > limit_gb

    def __enter__(self) -> ResourceSampler:
        self._proc.cpu_percent()  # first call only primes the counter
        self.sample()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.sample()


@dataclass
class StageStats:
    unit: str = "items"
    wall_s: float = 0.0
    items: int = 0
    cached: bool = False
    latencies_s: list[float] = field(default_factory=list)  # per-item latency samples

    def add(self, n_items: int, seconds: float) -> None:
        """A batch of `n_items` took `seconds` → one per-item latency sample per item."""
        self.items += n_items
        if n_items:
            self.latencies_s.extend([seconds / n_items] * n_items)


class StageTimer:
    def __init__(self) -> None:
        self.stages: dict[str, StageStats] = {}

    @contextmanager
    def stage(self, name: str, unit: str = "items") -> Iterator[StageStats]:
        st = self.stages[name] = StageStats(unit=unit)
        t0 = time.perf_counter()
        try:
            yield st
        finally:
            st.wall_s = time.perf_counter() - t0

    def cached(self, name: str, unit: str = "items") -> None:
        self.stages[name] = StageStats(unit=unit, cached=True)


def _stage_report(st: StageStats) -> dict[str, Any]:
    rate = st.items / st.wall_s if st.wall_s > 0 and st.items else None
    lat = np.asarray(st.latencies_s) * 1000
    return {
        "wall_s": round(st.wall_s, 4),
        "items": st.items,
        "fps" if st.unit == "frames" else "items_per_s": rate,
        "p50_ms": float(np.percentile(lat, 50)) if len(lat) else None,
        "p95_ms": float(np.percentile(lat, 95)) if len(lat) else None,
        "cached": st.cached,
    }


def runtime_report(
    timer: StageTimer,
    sampler: ResourceSampler | None,
    wall_time_s: float,
    frames_processed: int,
    cache_bytes: int,
) -> dict[str, Any]:
    return {
        "stages": {name: _stage_report(st) for name, st in timer.stages.items()},
        "wall_time_s": wall_time_s,
        "frames_processed": frames_processed,
        "fps": frames_processed / wall_time_s if wall_time_s > 0 else None,
        "peak_rss_gb": sampler.peak_rss_gb if sampler else None,
        "mean_cpu_pct": sampler.mean_cpu if sampler else None,
        "cache_bytes": cache_bytes,
    }
