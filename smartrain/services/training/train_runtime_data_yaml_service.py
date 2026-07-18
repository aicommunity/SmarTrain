from __future__ import annotations

import json
import os
from typing import Any

import yaml

from smartrain.core.runtime.path_portable import (
    is_abs_like,
    resolve_stored_path_under_workspace,
    store_path_under_workspace,
    to_posix,
)


def resolve_training_data_path(
    layout,
    data_arg: str,
    *,
    datasets_info_file: str,
    resolve_dataset_root_cb,
) -> str:
    expanded = os.path.abspath(os.path.expanduser(data_arg))
    yaml_here = os.path.join(expanded, "data.yaml")
    if os.path.isdir(expanded) and os.path.isfile(yaml_here):
        return expanded
    info_path = layout.work_datasets_info_path()
    if not os.path.isfile(info_path):
        raise FileNotFoundError(
            f"The directory with data.yaml for {data_arg!r} was not found and {info_path} is missing."
        )
    with open(info_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    if not isinstance(catalog, dict):
        raise ValueError(f"{info_path}: JSON object expected.")
    if data_arg not in catalog:
        names = ", ".join(sorted(catalog.keys()))
        hint = f" Known names: {names}." if names else ""
        raise ValueError(
            f"Dataset name {data_arg!r} is missing from datasets/{datasets_info_file}.{hint}"
        )
    entry = catalog[data_arg]
    if not isinstance(entry, dict):
        raise ValueError(f"The {data_arg!r} entry must be a JSON object.")
    return resolve_dataset_root_cb(layout.root, data_arg, entry, layout.work_datasets)


def split_dir_from_dataset_yaml(dataset_path: str, raw: dict, split_key: str) -> str | None:
    v = raw.get(split_key)
    if not isinstance(v, str) or not v.strip():
        return None
    rel = v.strip().replace("\\", "/").lstrip("./")
    if not rel:
        return None
    abs_p = os.path.normpath(os.path.join(dataset_path, rel))
    if os.path.isdir(abs_p):
        return rel
    return None


def pick_split_relative_dir(dataset_path: str, split_aliases: tuple[str, ...]) -> str | None:
    candidates: list[str] = []
    for split in split_aliases:
        candidates.extend([f"{split}/images", f"images/{split}", split])
    for rel in candidates:
        abs_p = os.path.join(dataset_path, rel)
        if os.path.isdir(abs_p):
            return rel
    return None


def resolve_runtime_split_dirs(dataset_path: str, raw: dict) -> tuple[str, str, str | None]:
    """Resolve train/val/test image dirs; val falls back to train when val is absent."""
    train_rel = pick_split_relative_dir(dataset_path, ("train",)) or split_dir_from_dataset_yaml(
        dataset_path, raw, "train"
    )
    if train_rel is None:
        raise FileNotFoundError(
            f"Required train split folder not found inside {dataset_path}."
        )
    val_rel = pick_split_relative_dir(dataset_path, ("val", "valid")) or split_dir_from_dataset_yaml(
        dataset_path, raw, "val"
    )
    if val_rel is None:
        val_rel = train_rel
        print(
            f"[WARN] val split folder not found in {dataset_path}; "
            f"using train images for validation ({train_rel})."
        )
    test_rel = pick_split_relative_dir(dataset_path, ("test",)) or split_dir_from_dataset_yaml(
        dataset_path, raw, "test"
    )
    if test_rel is None:
        test_rel = val_rel
    return train_rel, val_rel, test_rel


def coerce_dataset_root(dataset_path: str) -> tuple[str, str]:
    """Accept a dataset directory or a ``data.yaml`` path; return ``(root, yaml)``."""
    raw = str(dataset_path or "").strip()
    if not raw:
        raise ValueError("dataset_path is empty")
    abs_in = os.path.abspath(os.path.expanduser(raw))
    base = os.path.basename(abs_in).lower()
    if os.path.isfile(abs_in) and base.endswith((".yaml", ".yml")):
        return os.path.dirname(abs_in), abs_in
    if base in {"data.yaml", "data.yml"}:
        # Path written as …/data.yaml even if the file is missing momentarily.
        return os.path.dirname(abs_in), abs_in
    return abs_in, os.path.join(abs_in, "data.yaml")


def build_runtime_data_yaml(
    dataset_path: str,
    run_dir: str,
    *,
    stage: str,
    ensure_run_layout_cb,
    run_tmp_dir_cb,
    workspace_root: str | None = None,
) -> str:
    dataset_root, src_yaml = coerce_dataset_root(dataset_path)
    with open(src_yaml, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Incorrect YAML format data.yaml: {src_yaml}")

    train_rel, val_rel, test_rel = resolve_runtime_split_dirs(dataset_root, raw)

    runtime_cfg: dict[str, Any] = dict(raw)
    if workspace_root:
        stored = store_path_under_workspace(workspace_root, dataset_root)
        runtime_cfg["path"] = stored
    else:
        runtime_cfg["path"] = dataset_root
    runtime_cfg["train"] = to_posix(train_rel)
    runtime_cfg["val"] = to_posix(val_rel)
    runtime_cfg["test"] = to_posix(test_rel) if test_rel else test_rel

    ensure_run_layout_cb(run_dir)
    out_yaml = os.path.join(str(run_tmp_dir_cb(run_dir)), f"_runtime_data_{stage}.yaml")
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_cfg, f, allow_unicode=True, sort_keys=False)
    print(
        f"[INFO] Runtime data.yaml ({stage}) generated for the selected dataset: {out_yaml}"
    )
    return out_yaml


def materialize_ultralytics_data_yaml(
    portable_yaml_path: str,
    workspace_root: str,
) -> str:
    """
    Write a sibling ``*.ultralytics.yaml`` with an absolute ``path`` for Ultralytics.

    The portable file is left unchanged (workspace-relative ``path`` when under WS).
    """
    src = os.path.abspath(portable_yaml_path)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"portable runtime yaml missing: {src}")
    with open(src, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Incorrect YAML format: {src}")
    path_val = cfg.get("path")
    if isinstance(path_val, str) and path_val.strip():
        if is_abs_like(path_val) and os.path.isdir(os.path.abspath(os.path.expanduser(path_val))):
            abs_path = os.path.abspath(os.path.expanduser(path_val))
        else:
            abs_path = resolve_stored_path_under_workspace(workspace_root, path_val)
        cfg["path"] = abs_path
    stem, ext = os.path.splitext(src)
    if stem.endswith(".ultralytics"):
        out = src
    else:
        out = f"{stem}.ultralytics{ext or '.yaml'}"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out
