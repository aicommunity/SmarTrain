from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SPLIT_PREFERENCE: tuple[str, ...] = ("test", "val", "train")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_data_yaml_cfg(data_yaml_path: str) -> dict[str, Any]:
    with open(data_yaml_path, "r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML: {data_yaml_path}")
    return payload


def resolve_data_yaml_root(cfg: dict[str, Any], yaml_path: str | Path) -> Path:
    path_value = cfg.get("path")
    if isinstance(path_value, str) and path_value.strip():
        expanded = os.path.expanduser(path_value.strip())
        if os.path.isabs(expanded):
            return Path(expanded).resolve()
        yaml_dir = Path(yaml_path).resolve().parent
        return (yaml_dir / expanded).resolve()
    return Path(yaml_path).resolve().parent


def resolve_split_dir_path(data_yaml_path: str, split_name: str) -> Path | None:
    try:
        cfg = load_data_yaml_cfg(data_yaml_path)
    except Exception:
        return None
    split_rel = cfg.get(split_name)
    if not isinstance(split_rel, str) or not split_rel.strip():
        return None
    root = resolve_data_yaml_root(cfg, data_yaml_path)
    split_path = (root / split_rel.strip().replace("\\", "/").lstrip("./")).resolve()
    if os.path.isdir(split_path):
        return split_path
    return None


def split_dir_exists(data_yaml_path: str, split_name: str) -> bool:
    return resolve_split_dir_path(data_yaml_path, split_name) is not None


def resolve_best_split(
    data_yaml_path: str,
    preference: tuple[str, ...] = DEFAULT_SPLIT_PREFERENCE,
) -> str | None:
    for split_name in preference:
        if split_dir_exists(data_yaml_path, split_name):
            return split_name
    return None


def collect_split_images_for_split(data_yaml_path: str, split_name: str, limit: int) -> list[str]:
    split_path = resolve_split_dir_path(data_yaml_path, split_name)
    if split_path is None:
        raise FileNotFoundError(
            f"Split directory not found for split={split_name!r} in {data_yaml_path}"
        )
    images = sorted(
        p
        for p in glob(os.path.join(str(split_path), "**", "*"), recursive=True)
        if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)
    )
    if limit and limit > 0:
        return images[:limit]
    return images


def collect_split_images_resolved(
    data_yaml_path: str,
    preference: tuple[str, ...] = DEFAULT_SPLIT_PREFERENCE,
    limit: int = 0,
) -> tuple[list[str], str]:
    split_name = resolve_best_split(data_yaml_path, preference=preference)
    if split_name is None:
        raise FileNotFoundError(f"No image split found in {data_yaml_path} (tried: {', '.join(preference)})")
    images = collect_split_images_for_split(data_yaml_path, split_name, limit)
    if not images:
        raise FileNotFoundError(f"No images found for split={split_name!r} in {data_yaml_path}")
    return images, split_name


def candidate_source_rank(source: str) -> int:
    src = str(source or "").lower()
    if "training_metadata.dataset.name" in src or "path_under_workspace" in src:
        return 0
    if "training_metadata.dataset" in src:
        return 1
    if "train/args.yaml" in src and "runtime" not in src:
        return 2
    if "_runtime_data_" in src or "runtime" in src:
        return 4
    return 3


def pick_best_data_yaml_candidate(
    candidates: list[tuple[str, str]],
    *,
    preferred_split: str | None = None,
    split_preference: tuple[str, ...] = DEFAULT_SPLIT_PREFERENCE,
) -> tuple[str, str] | None:
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda item: (candidate_source_rank(item[1]), item[0]),
    )
    if preferred_split:
        preferred_matches = [
            item for item in ranked if split_dir_exists(item[0], preferred_split)
        ]
        if preferred_matches:
            return preferred_matches[0]
    for split_name in split_preference:
        for path, source in ranked:
            if split_dir_exists(path, split_name):
                return path, source
    return ranked[0]
