"""Shared helpers for writing YOLO data.yaml files."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def class_map_to_names(class_map: dict[str, Any]) -> list[str]:
    return [k for k, _ in sorted(((str(k), int(v)) for k, v in class_map.items()), key=lambda kv: kv[1])]


def write_data_yaml_from_class_map(
    out_dir: str,
    class_map: dict[str, Any],
    *,
    val_rel: str | None = None,
    test_rel: str = "test/images",
    train_rel: str = "train/images",
) -> None:
    """Write a standard split-layout data.yaml from a class id map."""
    names = class_map_to_names(class_map)
    if val_rel is None:
        val_rel = "valid/images" if (Path(out_dir) / "valid" / "images").is_dir() else "val/images"
    Path(out_dir, "data.yaml").write_text(
        f"train: {train_rel}\nval: {val_rel}\ntest: {test_rel}\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )


def write_data_yaml_from_names(
    out_dir: str,
    names: list[str],
    *,
    train_rel: str,
    val_rel: str,
    test_rel: str,
) -> None:
    Path(out_dir, "data.yaml").write_text(
        f"train: {train_rel}\nval: {val_rel}\ntest: {test_rel}\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )
