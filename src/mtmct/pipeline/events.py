"""Run events: overall progress, `log.jsonl`, cancellation and RAM guard checks, per-stage timing."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from mtmct.api.schemas import JobEventIn
from mtmct.runtime.profiling import ResourceSampler, StageStats, StageTimer

PROGRESS_EVERY_S = 0.5


class RunCancelled(Exception):
    pass


class MemoryLimitExceeded(RuntimeError):
    pass


class Reporter:
    """Emits JobEventIn to `on_event` and `log.jsonl`; `weights` = share of overall progress per stage."""

    def __init__(
        self,
        on_event: Callable[[JobEventIn], None],
        log_path: Path,
        weights: dict[str, float],
        should_cancel: Callable[[], bool],
        sampler: ResourceSampler,
        max_rss_gb: float,
    ) -> None:
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        self.on_event, self.log_path = on_event, log_path
        self.should_cancel, self.sampler, self.max_rss_gb = should_cancel, sampler, max_rss_gb
        self.timer = StageTimer()
        self.warnings: list[str] = []
        self.stage: str | None = None
        self._done = 0.0  # overall progress of finished stages
        self._last_progress = 0.0

    def log(self, ev: JobEventIn) -> None:
        with self.log_path.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(UTC).isoformat(), **ev.model_dump()}, default=str) + "\n")

    def emit(
        self,
        message: str,
        level: Literal["debug", "info", "warning", "error"] = "info",
        stage_progress: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        progress = self._done + self.weights.get(self.stage or "", 0.0) * (stage_progress or 0.0)
        ev = JobEventIn(
            level=level,
            stage=self.stage,
            progress=min(progress, 1.0),
            stage_progress=stage_progress,
            message=message,
            data=data or {},
        )
        self.log(ev)
        self.on_event(ev)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self.emit(message, level="warning")

    def progress(self, fraction: float, message: str = "") -> None:
        """Throttled stage progress (at most every PROGRESS_EVERY_S)."""
        now = time.monotonic()
        if now - self._last_progress >= PROGRESS_EVERY_S:
            self._last_progress = now
            self.emit(message or f"{self.stage} {fraction:.0%}", stage_progress=fraction)

    def check(self) -> None:
        """Called at batch boundaries: honour cancellation and the RAM guard."""
        if self.should_cancel():
            raise RunCancelled(f"run cancelled during stage {self.stage}")
        if self.sampler.exceeded(self.max_rss_gb):
            raise MemoryLimitExceeded(
                f"peak RSS {self.sampler.peak_rss_gb:.2f} GB exceeded data.max_rss_gb={self.max_rss_gb} GB "
                f"during stage {self.stage}; lower data.max_seconds / data.target_fps or raise the limit"
            )

    @contextmanager
    def stage_ctx(self, name: str, unit: str = "items") -> Iterator[StageStats]:
        self.stage = name
        self.check()
        self.emit(f"{name}: start", stage_progress=0.0)
        with self.timer.stage(name, unit) as st:
            yield st
        self.emit(f"{name}: done", stage_progress=1.0, data={"wall_s": round(st.wall_s, 3), "items": st.items})
        self._finish(name)

    def cached(self, name: str, unit: str = "items") -> None:
        self.stage = name
        self.timer.cached(name, unit)
        self.emit(f"{name}: cached", stage_progress=1.0)
        self._finish(name)

    def _finish(self, name: str) -> None:
        self._done += self.weights.get(name, 0.0)
        self.stage = None
