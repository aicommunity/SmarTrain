"""Portable normalization for dataset ``data.yaml`` files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml


def _strip_dot_slash(value: str) -> str:
    normalized = " ".join(((value or "").replace("\\", "/")).split())
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _to_rel_split(root: str, raw: str) -> str:
    value = _strip_dot_slash(raw)
    if not value:
        return value
    root_abs = os.path.abspath(root)
    joined = os.path.abspath(os.path.join(root_abs, value.replace("/", os.sep)))
    try:
        relative = os.path.relpath(joined, root_abs)
        if not relative.startswith(".."):
            return relative.replace("\\", "/")
    except ValueError:
        pass
    absolute = os.path.abspath(value.replace("/", os.sep))
    try:
        relative = os.path.relpath(absolute, root_abs)
        if not relative.startswith(".."):
            return relative.replace("\\", "/")
    except ValueError:
        pass
    return value.replace("\\", "/")


def _foreign_absolute_to_split_relative(dataset_root: str, value: str) -> str | None:
    root = Path(dataset_root)
    normalized = value.replace("\\", "/").strip()
    if not normalized or not (normalized.startswith("/") or (len(normalized) > 2 and normalized[1] == ":")):
        return None
    for relative in ("train/images", "val/images", "valid/images", "test/images"):
        if (root / relative.replace("/", os.sep)).is_dir() and normalized.endswith("/" + relative):
            return relative
    if (root / "images").is_dir():
        if any(normalized.endswith(tail) for tail in ("/train/images", "/val/images", "/valid/images", "/test/images")):
            return None
        if normalized.endswith("/images"):
            return "images"
    return None


def _rewrite_split_string(dataset_root: str, raw: str) -> str:
    value = _to_rel_split(dataset_root, raw)
    return _foreign_absolute_to_split_relative(dataset_root, value) or value


def _normalize_split_field(root: str, value: Any) -> Any:
    if isinstance(value, list):
        return [_rewrite_split_string(root, str(item)) for item in value if isinstance(item, str)]
    if isinstance(value, str):
        return _rewrite_split_string(root, value)
    return value


def normalize_data_yaml_mapping(dataset_root: str, data: Mapping[str, Any]) -> dict[str, Any]:
    root = os.path.abspath(os.path.expanduser(dataset_root))
    normalized = dict(data)
    normalized.pop("path", None)
    for key in ("train", "val", "test", "minival"):
        if key in normalized and normalized[key] not in (None, ""):
            normalized[key] = _normalize_split_field(root, normalized[key])
    return normalized


def _canonical_dump(data: dict[str, Any]) -> str:
    ordered: dict[str, Any] = {}
    for key in ("train", "val", "test", "minival"):
        if key in data:
            ordered[key] = data[key]
    for key, value in data.items():
        if key not in ordered:
            ordered[key] = value
    return yaml.safe_dump(ordered, allow_unicode=True, default_flow_style=False, sort_keys=False)


def normalize_data_yaml_file(dataset_dir: str, *, dry_run: bool = False) -> tuple[bool, str]:
    path = Path(dataset_dir) / "data.yaml"
    if not path.is_file():
        return False, "no data.yaml"
    try:
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
    except Exception as error:
        return False, f"read error: {error}"
    if not isinstance(raw, dict):
        return False, "not a mapping"
    normalized = normalize_data_yaml_mapping(str(path.parent), raw)
    dumped = _canonical_dump(normalized)
    try:
        unchanged = yaml.safe_load(dumped) == yaml.safe_load(text)
    except Exception:
        unchanged = False
    if unchanged:
        return False, "already normalized"
    if dry_run:
        return True, "would update"
    path.write_text(dumped, encoding="utf-8")
    return True, "updated"


def iter_dataset_roots_with_data_yaml(datasets_root: str) -> list[str]:
    root = Path(os.path.abspath(os.path.expanduser(datasets_root)))
    if not root.is_dir():
        return []
    found: set[str] = set()
    for path in root.rglob("data.yaml"):
        if path.is_file() and path.parent.resolve() != root.resolve():
            found.add(str(path.parent.resolve()))
    return sorted(found)
