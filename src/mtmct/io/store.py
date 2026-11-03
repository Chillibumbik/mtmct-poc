"""Validated table IO and content-addressed, atomic stage cache directories (spec §6.4)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa

CACHE_VERSION = "v1"


def write_table(df: pd.DataFrame, path: Path, schema: type[pa.DataFrameModel]) -> None:
    validated = schema.validate(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        validated.to_csv(path, index=False)
    else:
        validated.to_parquet(path, index=False)


def read_table(path: Path, schema: type[pa.DataFrameModel]) -> pd.DataFrame:
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    return schema.validate(df)


def config_hash(obj: Any) -> str:
    """8 hex chars of sha256 over canonical JSON; independent of dict key order."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


@dataclass
class StageDir:
    """`<root>/<stage>-<hash8>/`. Work goes into `tmp`; `commit` atomically publishes it with a manifest."""

    root: Path
    stage: str
    key: dict[str, Any]

    @property
    def hash(self) -> str:
        return config_hash(self.key)

    @property
    def path(self) -> Path:
        return self.root / f"{self.stage}-{self.hash}"

    @property
    def tmp(self) -> Path:
        return self.root / f".{self.stage}-{self.hash}.{os.getpid()}.tmp"  # per process: no shared scratch dir

    @property
    def done(self) -> bool:
        return (self.path / "manifest.json").exists()

    def open(self) -> Path:
        """Fresh empty tmp dir (discards leftovers of an interrupted attempt)."""
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp.mkdir(parents=True)
        return self.tmp

    def commit(self, manifest: dict[str, Any]) -> Path:
        body = {"stage": self.stage, "key": self.key, "hash": self.hash, "cache_version": CACHE_VERSION, **manifest}
        (self.tmp / "manifest.json").write_text(json.dumps(body, indent=2, default=str))
        shutil.rmtree(self.path, ignore_errors=True)
        os.replace(self.tmp, self.path)
        return self.path

    def abort(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


def stage_dir(root: Path, stage: str, key: dict[str, Any]) -> StageDir:
    root.mkdir(parents=True, exist_ok=True)
    return StageDir(root=root, stage=stage, key=key)
