"""Registry of interchangeable components (spec §6.3).

A component is a class decorated with `@register(category, name)` that defines an inner
`Params(pydantic.BaseModel)` and a `LICENCE` string, and is constructed as `cls(params)`.
"""

from __future__ import annotations

import importlib
from typing import Any, TypeVar

from pydantic import BaseModel

CATEGORIES = ("detector", "tracker", "reid", "foot_point", "cost", "solver")

# Modules that register components when imported. Missing modules are skipped so that
# zones can land independently; optional extras (AGPL) are imported only if installed.
PLUGIN_MODULES = (
    "mtmct.detect.yolox",
    "mtmct.detect.ultralytics_yolo",
    "mtmct.reid.osnet",
    "mtmct.reid.omz",
    "mtmct.track.trackers_lib",
    "mtmct.geometry.foot",
    "mtmct.associate.costs.appearance",
    "mtmct.associate.costs.geometry",
    "mtmct.associate.costs.time",
    "mtmct.associate.solvers.agglomerative",
    "mtmct.associate.solvers.anchor",
    "mtmct.associate.solvers.hungarian",
)

_REGISTRY: dict[str, dict[str, Any]] = {c: {} for c in CATEGORIES}
T = TypeVar("T", bound=type)


class ComponentInfo(BaseModel):
    category: str
    name: str
    params_schema: dict[str, Any]
    licence: str
    optional: bool
    description: str


def register(category: str, name: str):
    if category not in _REGISTRY:
        raise ValueError(f"unknown category {category!r}; expected one of {CATEGORIES}")

    def deco(cls: T) -> T:
        params = getattr(cls, "Params", None)
        if not (isinstance(params, type) and issubclass(params, BaseModel)):
            raise TypeError(f"{cls.__name__} must define inner `Params(BaseModel)`")
        if not isinstance(getattr(cls, "LICENCE", None), str):
            raise TypeError(f"{cls.__name__} must define `LICENCE: str`")
        if name in _REGISTRY[category] and _REGISTRY[category][name] is not cls:
            raise ValueError(f"duplicate component {category}/{name}")
        cls.CATEGORY, cls.NAME = category, name  # type: ignore[attr-defined]
        _REGISTRY[category][name] = cls
        return cls

    return deco


def load_plugins() -> None:
    for module in PLUGIN_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as e:
            # Skip only when the plugin module itself (or its optional extra) is absent.
            if e.name is None or not (module.startswith(e.name) or e.name in {"ultralytics", "torch"}):
                raise


def get(category: str, name: str) -> Any:
    if name not in _REGISTRY.get(category, {}):
        load_plugins()
    try:
        return _REGISTRY[category][name]
    except KeyError:
        known = sorted(_REGISTRY.get(category, {}))
        raise KeyError(f"unknown component {category}/{name}; known: {known}") from None


def create(category: str, name: str, params: dict[str, Any] | None = None) -> Any:
    cls = get(category, name)
    return cls(cls.Params(**(params or {})))


def catalog() -> dict[str, list[ComponentInfo]]:
    load_plugins()
    return {
        category: [
            ComponentInfo(
                category=category,
                name=name,
                params_schema=cls.Params.model_json_schema(),
                licence=cls.LICENCE,
                optional=bool(getattr(cls, "OPTIONAL", False)),
                description=(cls.__doc__ or "").strip().split("\n")[0],
            )
            for name, cls in sorted(items.items())
        ]
        for category, items in _REGISTRY.items()
    }
