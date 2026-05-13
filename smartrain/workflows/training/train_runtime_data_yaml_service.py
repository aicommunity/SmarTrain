from __future__ import annotations

import json
import os
from typing import Any

import yaml


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


def build_runtime_data_yaml(
    dataset_path: str,
    run_dir: str,
    *,
    stage: str,
    ensure_run_layout_cb,
    run_tmp_dir_cb,
) -> str:
    src_yaml = os.path.join(dataset_path, "data.yaml")
    with open(src_yaml, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Incorrect YAML format data.yaml: {src_yaml}")

    train_rel = pick_split_relative_dir(dataset_path, ("train",)) or split_dir_from_dataset_yaml(
        dataset_path, raw, "train"
    )
    val_rel = pick_split_relative_dir(dataset_path, ("val", "valid")) or split_dir_from_dataset_yaml(
        dataset_path, raw, "val"
    )
    test_rel = pick_split_relative_dir(dataset_path, ("test",)) or split_dir_from_dataset_yaml(
        dataset_path, raw, "test"
    )
    if train_rel is None or val_rel is None:
        raise FileNotFoundError(
            f"Required train/val split folders not found inside {dataset_path}."
        )

    runtime_cfg: dict[str, Any] = dict(raw)
    runtime_cfg["path"] = dataset_path
    runtime_cfg["train"] = train_rel
    runtime_cfg["val"] = val_rel
    if test_rel is not None:
        runtime_cfg["test"] = test_rel

    ensure_run_layout_cb(run_dir)
    out_yaml = os.path.join(str(run_tmp_dir_cb(run_dir)), f"_runtime_data_{stage}.yaml")
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_cfg, f, allow_unicode=True, sort_keys=False)
    print(
        f"[INFO] Runtime data.yaml ({stage}) generated for the selected dataset: {out_yaml}"
    )
    return out_yaml

