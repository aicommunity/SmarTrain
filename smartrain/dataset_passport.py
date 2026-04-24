from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from smartrain.dataset_hash import calculate_dataset_hash
from smartrain.path_portable import relativize_abs_paths_in_obj, relativize_if_under


def next_dataset_name(base_dir: str, preferred: str) -> str:
    name = preferred.strip()
    if not name:
        name = "dataset"
    candidate = name
    idx = 2
    while os.path.exists(os.path.join(base_dir, candidate)):
        candidate = f"{name}_{idx}"
        idx += 1
    return candidate


def write_dataset_passport(
    *,
    output_dataset_dir: str,
    command: str,
    source_datasets: list[dict[str, Any]],
    parameters: dict[str, Any],
    transformations: list[dict[str, Any]],
    stats_before: dict[str, Any] | None = None,
    stats_after: dict[str, Any] | None = None,
    random_seed: int | None = None,
    workspace_root: str | None = None,
) -> str:
    os.makedirs(output_dataset_dir, exist_ok=True)
    output_hash = None
    try:
        output_hash = calculate_dataset_hash(output_dataset_dir)
    except Exception:
        output_hash = None
    out_path_abs = os.path.abspath(output_dataset_dir)
    created_path: str = out_path_abs
    if workspace_root:
        created_path = relativize_if_under(workspace_root, out_path_abs) or created_path
    norm_sources = source_datasets
    norm_parameters = parameters
    if workspace_root:
        norm_sources = relativize_abs_paths_in_obj(list(source_datasets), workspace_root)
        norm_parameters = relativize_abs_paths_in_obj(dict(parameters), workspace_root)
    passport = {
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": norm_sources,
        "created_dataset": {
            "name": os.path.basename(os.path.normpath(output_dataset_dir)),
            "path": created_path,
        },
        "tool_version": "smartrain-0.1.0",
        "parameters": norm_parameters,
        "transformations": transformations,
        "random_seed": random_seed,
        "input_hash": [x.get("dataset_hash") for x in source_datasets],
        "output_hash": output_hash,
        "stats_before": stats_before or {},
        "stats_after": stats_after or {},
    }
    out_path = os.path.join(output_dataset_dir, "dataset_passport.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(passport, f, ensure_ascii=False, indent=2)
    return out_path

