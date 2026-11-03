from pathlib import Path

from mtmct.config import load_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_max_rss_env_overrides_base_but_not_explicit_override(monkeypatch):
    monkeypatch.setenv("MTMCT_MAX_RSS_GB", "2.5")
    assert load_config(None, "overlap", configs_dir=CONFIGS).data.max_rss_gb == 2.5
    assert load_config(None, "overlap", {"data.max_rss_gb": 4}, configs_dir=CONFIGS).data.max_rss_gb == 4
