"""System info, component registry and profiles (spec §10.1)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from functools import cache
from importlib import metadata
from typing import Literal, cast

import psutil
from fastapi import APIRouter

from mtmct import registry
from mtmct.api.schemas import ComponentOut, LicenceItem, ProfileOut, RegistryOut, SystemOut
from mtmct.config import load_config, paths
from mtmct.data.base import load_adapters

router = APIRouter()

GB = 1024**3
PROFILE_DESCRIPTIONS = {
    "overlap": "Перекрывающиеся калиброванные камеры: геометрия на плоскости земли — основной признак.",
    "nonoverlap": "Непересекающиеся камеры: внешний вид + окна времени переходов из topology.json.",
}


@cache
def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=paths().root, capture_output=True, text=True, timeout=3
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


@cache
def _openvino_devices() -> list[str]:
    try:
        from openvino import Core

        return list(Core().available_devices)
    except Exception:
        return []


@cache
def _cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):  # noqa: SIM115
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def warmup() -> None:
    """Pay the slow first-call costs (OpenVINO init, plugin imports) at startup, not on the first request."""
    _openvino_devices()
    registry.catalog()


@router.get("/system", response_model=SystemOut)
def system() -> SystemOut:
    mem = psutil.virtual_memory()
    licences = [
        LicenceItem(
            kind="model" if c.category in ("detector", "reid") else "library",
            name=f"{c.category}/{c.name}",
            licence=c.licence,
            note="опциональный компонент" if c.optional else "",
        )
        for items in registry.catalog().values()
        for c in items
    ]
    licences += [
        LicenceItem(kind="dataset", name=a.meta().title, licence=a.meta().licence) for a in load_adapters().values()
    ]
    return SystemOut(
        version=metadata.version("mtmct"),
        git_sha=_git_sha(),
        cpu=_cpu_name(),
        cpu_threads=os.cpu_count() or 1,
        ram_total_gb=round(mem.total / GB, 2),
        ram_available_gb=round(mem.available / GB, 2),
        disk_free_gb=round(shutil.disk_usage(paths().root).free / GB, 2),
        openvino_devices=_openvino_devices(),
        licences=licences,
    )


@router.get("/registry", response_model=RegistryOut)
def get_registry() -> RegistryOut:
    return RegistryOut(
        components={cat: [ComponentOut(**c.model_dump()) for c in items] for cat, items in registry.catalog().items()}
    )


@router.get("/profiles", response_model=list[ProfileOut])
def profiles() -> list[ProfileOut]:
    return [
        ProfileOut(
            name=cast(Literal["overlap", "nonoverlap"], name),
            description=desc,
            config=load_config(None, name).model_dump(mode="json"),
        )
        for name, desc in PROFILE_DESCRIPTIONS.items()
    ]
