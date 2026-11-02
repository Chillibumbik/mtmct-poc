from pathlib import Path

import pytest
from pydantic import ValidationError

from mtmct.config import load_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_load_config_merge_order_and_dotted_overrides():
    cfg = load_config(
        "campus", None, {"associate.solver.params.threshold": 0.6, "data.target_fps": 5}, configs_dir=CONFIGS
    )
    assert cfg.profile == "overlap"
    assert set(cfg.associate.costs) == {"appearance", "geometry"}  # base + overlap profile
    assert cfg.associate.solver.params["threshold"] == 0.6
    assert cfg.data.target_fps == 5


def test_nonoverlap_profile_adds_time_cost():
    cfg = load_config(None, "nonoverlap", configs_dir=CONFIGS)
    assert set(cfg.associate.costs) == {"appearance", "time"}


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        load_config(None, "overlap", {"data.nope": 1}, configs_dir=CONFIGS)


def test_stage_key_includes_upstream_only():
    a = load_config(None, "overlap", configs_dir=CONFIGS)
    b = load_config(
        None, "overlap", {"associate.solver.params.threshold": 0.9, "data.max_rss_gb": 4}, configs_dir=CONFIGS
    )
    c = load_config(None, "overlap", {"detect.params.conf": 0.5}, configs_dir=CONFIGS)
    for stage in ("detect", "track", "reid", "refine"):
        assert a.stage_key(stage) == b.stage_key(stage)  # association/RAM guard do not invalidate caches
        assert a.stage_key(stage) != c.stage_key(stage)  # detector change invalidates everything downstream
    assert "refine" not in a.stage_key("track")
