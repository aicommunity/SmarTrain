from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from smartrain.dataset_hash import calculate_dataset_hash


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
) -> str:
    os.makedirs(output_dataset_dir, exist_ok=True)
    output_hash = None
    try:
        output_hash = calculate_dataset_hash(output_dataset_dir)
    except Exception:
        output_hash = None
    passport = {
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": source_datasets,
        "created_dataset": {
            "name": os.path.basename(os.path.normpath(output_dataset_dir)),
            "path": os.path.abspath(output_dataset_dir),
        },
        "tool_version": "smartrain-0.1.0",
        "parameters": parameters,
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

