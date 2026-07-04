from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Final


@dataclass(frozen=True)
class UltralyticsModelAliasSpec:
    provider: str
    supported_aliases: tuple[str, ...]


_ULTRALYTICS_DEFAULT_FALLBACK_ALIASES: Final[tuple[str, ...]] = (
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
)


def _aliases_from_yaml_names(yaml_names: list[str]) -> tuple[str, ...]:
    aliases: set[str] = set()
    for name in yaml_names:
        if not name.endswith(".yaml"):
            continue
        alias = name[:-5].strip()
        if not alias:
            continue
        aliases.add(alias)
        m = re.match(r"^(yolo(?:v)?\d+)(-[a-z0-9]+)?$", alias)
        if m:
            base = m.group(1)
            tail = m.group(2) or ""
            for scale in ("n", "s", "m", "l", "x"):
                aliases.add(f"{base}{scale}{tail}")
    return tuple(sorted(aliases))


@lru_cache(maxsize=1)
def _discover_ultralytics_supported_aliases() -> tuple[str, ...]:
    """
    Read model aliases from installed ultralytics package configs.

    Source: ultralytics/cfg/models/**/*.yaml
    """
    try:
        import ultralytics  # type: ignore
    except Exception:
        return _ULTRALYTICS_DEFAULT_FALLBACK_ALIASES

    pkg_root = Path(getattr(ultralytics, "__file__", "")).resolve().parent
    models_cfg_root = pkg_root / "cfg" / "models"
    if not models_cfg_root.is_dir():
        return _ULTRALYTICS_DEFAULT_FALLBACK_ALIASES

    yaml_names = [p.name.lower() for p in models_cfg_root.rglob("*.yaml")]
    aliases = _aliases_from_yaml_names(yaml_names)
    if not aliases:
        return _ULTRALYTICS_DEFAULT_FALLBACK_ALIASES
    return aliases


@lru_cache(maxsize=1)
def _ultralytics_spec() -> UltralyticsModelAliasSpec:
    return UltralyticsModelAliasSpec(
        provider="ultralytics",
        supported_aliases=_discover_ultralytics_supported_aliases(),
    )


def _alias_registry() -> dict[str, UltralyticsModelAliasSpec]:
    spec = _ultralytics_spec()
    return {spec.provider: spec}


def default_train_provider() -> str:
    return _ultralytics_spec().provider


def get_ultralytics_model_alias_spec(provider: str | None = None) -> UltralyticsModelAliasSpec:
    registry = _alias_registry()
    key = (provider or default_train_provider()).strip().lower()
    if key not in registry:
        known = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown training provider: {provider!r}. Known providers: {known}")
    return registry[key]


def list_train_providers() -> tuple[str, ...]:
    return tuple(sorted(_alias_registry().keys()))
