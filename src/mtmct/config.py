"""Run configuration: base ⊕ profile ⊕ dataset ⊕ dotted overrides (spec §6.3), plus project paths."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.root / "data" / "_raw"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def db(self) -> Path:
        return self.root / "db"


@cache
def paths() -> Paths:
    """Project root = $MTMCT_HOME, else the nearest ancestor of this file containing `configs/base.yaml`."""
    if env := os.environ.get("MTMCT_HOME"):
        return Paths(Path(env).resolve())
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs" / "base.yaml").exists():
            return Paths(parent)
    return Paths(Path.cwd())


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentSpec(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class CostSpec(_Strict):
    weight: float = 1.0
    params: dict[str, Any] = Field(default_factory=dict)


class DataCfg(_Strict):
    target_fps: float = 10.0
    t_start: float = 0.0  # seconds from sequence start (applied after split range)
    max_seconds: float | None = None  # cap processed duration (smoke tests)
    max_rss_gb: float = 9.0  # RAM guard: abort the run above this peak RSS


class RefineCfg(_Strict):
    enabled: bool = True
    split_thr: float = 0.75
    window: int = 7
    max_gap_s: float = 1.0
    stitch_dist_px: float = 80.0
    stitch_app_thr: float = 0.75
    min_len: int = 8
    min_score: float = 0.4
    interp_gap: int = 10


class PortraitCfg(_Strict):
    reid_stride: int = 3  # embed every n-th box of a tracklet ...
    top_k: int = 10  # ... plus the top-k boxes by quality; portrait = mean of top-k embedded


class GeometryCfg(_Strict):
    foot_point: ComponentSpec = ComponentSpec(name="bottom_center")
    smooth_window: int = 5


class AssociateCfg(_Strict):
    costs: dict[str, CostSpec] = Field(default_factory=lambda: {"appearance": CostSpec()})
    constraints: dict[str, Any] = Field(default_factory=lambda: {"same_camera_overlap": "cannot_link"})
    rerank: dict[str, Any] | None = None  # e.g. {"k1": 20, "k2": 6, "lambda_": 0.3}
    solver: ComponentSpec = ComponentSpec(name="agglomerative", params={"linkage": "average", "threshold": 0.45})


class PostprocessCfg(_Strict):
    min_global_len: int = 10
    drop_short_single_camera: bool = True


class EvalCfg(_Strict):
    iou: float = 0.5
    thresholds: list[float] = Field(default_factory=lambda: [round(0.1 + 0.05 * i, 2) for i in range(15)])


class RunConfig(_Strict):
    profile: Literal["overlap", "nonoverlap"] = "overlap"
    data: DataCfg = DataCfg()
    detect: ComponentSpec = ComponentSpec(name="yolox_s")
    track: ComponentSpec = ComponentSpec(name="bytetrack")
    reid: ComponentSpec = ComponentSpec(name="osnet_ain_x1_0")
    refine: RefineCfg = RefineCfg()
    portrait: PortraitCfg = PortraitCfg()
    geometry: GeometryCfg = GeometryCfg()
    associate: AssociateCfg = AssociateCfg()
    postprocess: PostprocessCfg = PostprocessCfg()
    eval: EvalCfg = EvalCfg()

    def stage_key(self, stage: str) -> dict[str, Any]:
        """Cache key of a heavy stage = its own config + keys of upstream stages."""
        own = {
            "detect": {"data": self.data.model_dump(exclude={"max_rss_gb"}), "detect": self.detect.model_dump()},
            "track": {"track": self.track.model_dump()},
            "reid": {"reid": self.reid.model_dump(), "portrait": self.portrait.model_dump()},
            "refine": {"refine": self.refine.model_dump()},
        }
        order = list(own)
        if stage not in own:
            raise KeyError(f"no cache key for stage {stage!r}; heavy stages: {order}")
        key: dict[str, Any] = {}
        for s in order[: order.index(stage) + 1]:
            key.update(own[s])
        return key


def deep_merge(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """`{"associate.solver.params.threshold": 0.5}` → nested assignment (creates missing dicts)."""
    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = out
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {} if path.exists() else {}


def load_config(
    dataset: str | None = None,
    profile: str | None = None,
    overrides: dict[str, Any] | None = None,
    configs_dir: Path | None = None,
) -> RunConfig:
    root = configs_dir or paths().configs
    merged = _read_yaml(root / "base.yaml")
    ds = _read_yaml(root / "datasets" / f"{dataset}.yaml") if dataset else {}
    profile = profile or ds.get("profile") or merged.get("profile") or "overlap"
    merged = deep_merge(merged, _read_yaml(root / "profiles" / f"{profile}.yaml"))
    merged = deep_merge(merged, {k: v for k, v in ds.items() if k in RunConfig.model_fields})
    merged["profile"] = profile
    merged = apply_overrides(merged, overrides or {})
    return RunConfig.model_validate(merged)
