"""Pipeline orchestrator (spec §6.1, §8): cached heavy stages → cheap association/eval → run directory."""

from __future__ import annotations

import contextlib
import importlib.metadata
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mtmct.api.schemas import JobEventIn
from mtmct.associate.run import associate, build_context, threshold_labels
from mtmct.config import RunConfig, paths
from mtmct.data.base import SequenceInfo
from mtmct.eval.report import evaluate_run
from mtmct.geometry.homography import to_world
from mtmct.io import CACHE_VERSION, StageDir, Tracklets, read_table, stage_dir
from mtmct.pipeline import stages
from mtmct.pipeline.events import MemoryLimitExceeded, Reporter, RunCancelled
from mtmct.pipeline.outputs import write_json, write_run_header, write_tracks
from mtmct.portrait.portrait import build_portraits
from mtmct.postprocess.postprocess import postprocess
from mtmct.runtime.profiling import ResourceSampler, StageStats, runtime_report
from mtmct.video.reader import selected_frame_ids

__all__ = ["MemoryLimitExceeded", "RunCancelled", "RunResult", "run_pipeline"]

Mode = Literal["full", "oracle"]
WEIGHTS = {
    "full": {"detect": 0.45, "track": 0.05, "reid": 0.30, "refine": 0.05, "associate": 0.10, "eval": 0.05},
    "oracle": {"oracle": 0.05, "reid": 0.70, "associate": 0.10, "eval": 0.05},
}
UNITS = {"detect": "frames", "track": "frames", "oracle": "boxes", "reid": "boxes", "refine": "boxes"}
VERSIONED_PACKAGES = ("mtmct", "numpy", "openvino", "onnxruntime", "trackers")


@dataclass
class RunResult:
    run_dir: Path
    metrics: dict[str, Any]
    runtime: dict[str, Any]


def time_range(cfg: RunConfig, seq: SequenceInfo, split: str) -> tuple[float, float]:
    a, b = seq.split_range(split)
    t0 = a + cfg.data.t_start
    return t0, b if cfg.data.max_seconds is None else min(b, t0 + cfg.data.max_seconds)


def heavy_plan(cfg: RunConfig, mode: Mode) -> list[str]:
    if mode == "oracle":
        return ["oracle", "reid"]  # oracle tracklets are clean by construction: no refine
    return ["detect", "track", "reid"] + (["refine"] if cfg.refine.enabled else [])


def stage_key(cfg: RunConfig, stage: str, mode: Mode, t_range: tuple[float, float]) -> dict[str, Any]:
    key = {"data": cfg.stage_key("detect")["data"]} if stage == "oracle" else cfg.stage_key(stage)
    if mode == "oracle":  # detector/tracker settings do not affect GT tracklets
        key = {k: v for k, v in key.items() if k not in ("detect", "track")}
    return key | {"t_range": list(t_range), "mode": mode}


def _versions() -> dict[str, str]:
    out = {}
    for pkg in VERSIONED_PACKAGES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            out[pkg] = importlib.metadata.version(pkg)
    return out


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def _compute(
    name: str,
    cfg: RunConfig,
    seq: SequenceInfo,
    t_range: tuple[float, float],
    frames: dict[str, np.ndarray],
    prev: Path,
    out: Path,
    rep: Reporter,
    st: StageStats,
) -> None:
    """Run heavy stage `name`, reading its input from the upstream stage dir `prev`, writing into `out`."""
    if name == "detect":
        stages.detect(seq, cfg, t_range, frames, out, rep, st)
    elif name == "track":
        stages.track(seq, cfg, t_range, frames, prev, out, rep, st)
    elif name == "oracle":
        stages.oracle(seq, cfg, frames, out, st)
    elif name == "reid":
        stages.reid(seq, cfg, t_range, prev, out, rep, st)
    elif name == "refine":
        stages.refine_stage(cfg, stages.processed_fps(cfg, seq), prev, out, st)
    else:
        raise KeyError(name)


def _run_heavy(
    cfg: RunConfig,
    seq: SequenceInfo,
    mode: Mode,
    t_range: tuple[float, float],
    frames: dict[str, np.ndarray],
    rep: Reporter,
) -> dict[str, StageDir]:
    """Run (or reuse) the cached stages; returns their committed directories."""
    plan = heavy_plan(cfg, mode)
    root = paths().cache / CACHE_VERSION / seq.dataset / seq.sequence
    dirs = {s: stage_dir(root, s, stage_key(cfg, s, mode, t_range)) for s in plan}
    # Keys chain upstream configs, so the newest finished stage makes all earlier ones unnecessary
    # (embeddings always come from the reid dir, so it must exist when a later stage is reused).
    last_done = max((i for i, s in enumerate(plan) if dirs[s].done), default=-1)
    if not dirs["reid"].done:
        last_done = min(last_done, plan.index("reid") - 1)
    for i, name in enumerate(plan):
        if i <= last_done:
            rep.cached(name, UNITS[name])
            continue
        prev = dirs[plan[i - 1]].path if i else Path()
        sd = dirs[name]
        with rep.stage_ctx(name, UNITS[name]) as st:
            t0 = time.perf_counter()
            tmp = sd.open()
            try:
                _compute(name, cfg, seq, t_range, frames, prev, tmp, rep, st)
            except BaseException:
                sd.abort()
                raise
            sd.commit(
                {
                    "status": "done",
                    "created_at": datetime.now(UTC).isoformat(),
                    "duration_s": time.perf_counter() - t0,
                    "inputs": {plan[i - 1]: dirs[plan[i - 1]].hash} if i else {},
                    "versions": _versions(),
                }
            )
    return dirs


def run_pipeline(
    cfg: RunConfig,
    seq: SequenceInfo,
    mode: Mode,
    split: str,
    out_dir: Path | None,
    on_event: Callable[[JobEventIn], None],
    should_cancel: Callable[[], bool],
) -> RunResult:
    """Run one sequence end to end. Raises RunCancelled / MemoryLimitExceeded; partial stage dirs never remain."""
    run_dir = out_dir or paths().outputs / f"cli-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_header(run_dir, cfg)
    t_range = time_range(cfg, seq, split)
    frames = {c.camera_id: selected_frame_ids(c, c.n_frames, cfg.data.target_fps, t_range) for c in seq.cameras}
    n_frames = int(sum(len(f) for f in frames.values()))
    weights = {k: v for k, v in WEIGHTS[mode].items() if k in heavy_plan(cfg, mode) or k == "associate"}
    if seq.has_gt:
        weights["eval"] = WEIGHTS[mode]["eval"]
    t_start = time.perf_counter()
    with ResourceSampler() as sampler:
        rep = Reporter(on_event, run_dir / "log.jsonl", weights, should_cancel, sampler, cfg.data.max_rss_gb)
        rep.emit(
            f"run started: {seq.dataset}/{seq.sequence} split={split} mode={mode}, {n_frames} frames",
            data={"t_range": list(t_range), "frames": n_frames},
        )
        try:
            dirs = _run_heavy(cfg, seq, mode, t_range, frames, rep)
            metrics = _associate_and_eval(cfg, seq, mode, split, run_dir, frames, dirs, rep)
        except BaseException as e:
            rep.log(JobEventIn(level="error", stage=rep.stage, message=f"{type(e).__name__}: {e}"))
            raise
    runtime = runtime_report(
        rep.timer, sampler, time.perf_counter() - t_start, n_frames, sum(_dir_bytes(d.path) for d in dirs.values())
    )
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "runtime.json", runtime)
    rep.emit("run finished", data={"run_dir": str(run_dir)})
    return RunResult(run_dir=run_dir, metrics=metrics, runtime=runtime)


def _associate_and_eval(
    cfg: RunConfig,
    seq: SequenceInfo,
    mode: Mode,
    split: str,
    run_dir: Path,
    frames: dict[str, np.ndarray],
    dirs: dict[str, StageDir],
    rep: Reporter,
) -> dict[str, Any]:
    final = dirs[heavy_plan(cfg, mode)[-1]].path
    with rep.stage_ctx("portrait", "tracklets") as st:
        tracklets = to_world(read_table(final / "tracklets.parquet", Tracklets), seq, cfg.geometry.smooth_window)
        emb = np.load(dirs["reid"].path / "embeddings.npy").astype(np.float32)
        portraits, portrait_emb, galleries = build_portraits(tracklets, emb, cfg.portrait.top_k)
        st.items = len(portraits)
    with rep.stage_ctx("associate", "tracklets") as st:
        ctx = build_context(seq, tracklets, portraits, portrait_emb, galleries, seq.topology(), rep.warn)
        labels = associate(ctx, cfg.associate)
        curve = threshold_labels(ctx, cfg.associate, cfg.eval.thresholds) if seq.has_gt else None
        st.items = len(portraits)
    with rep.stage_ctx("postprocess", "boxes") as st:
        tracks = postprocess(tracklets, labels, cfg.postprocess)
        write_tracks(run_dir, tracks, seq)
        st.items = len(tracks)
    metrics: dict[str, Any] = {
        "run_id": run_dir.name,
        "dataset": seq.dataset,
        "sequence": seq.sequence,
        "split": split,
        "mode": mode,
    }
    gt = seq.gt()
    if gt is not None:
        with rep.stage_ctx("eval", "boxes") as st:
            step = seq.meta.gt_frame_step
            eval_frames = {cam: f[f % step == 0] for cam, f in frames.items()}
            metrics |= evaluate_run(gt, tracks, seq, eval_frames, cfg.eval.iou, portraits, portrait_emb, curve)
            st.items = len(tracks)
    return metrics | {"warnings": rep.warnings}
