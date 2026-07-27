"""Registry helpers: load run training fields via gateway + training_metadata.json."""

from __future__ import annotations

import json
import os
from typing import Any

from smartrain.run_model_contract.gateway import load_target
from smartrain.run_model_contract.io.read.resolvers import infer_source_kind


def read_training_metadata(run_dir: str) -> dict[str, Any]:
    """Load ``training_metadata.json`` for a run directory.

    Contract-local reader used by registry (not ``metrics_reader.load_metadata``).
    """
    path = os.path.join(run_dir, "training_metadata.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No training_metadata.json: {run_dir}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"training_metadata.json is not an object: {path}")
    return data


def load_run_list_fields(run_dir: str) -> dict[str, str]:
    """Resolve model + dataset labels for ``registry runs-list`` via gateway first."""
    model = "?"
    dataset = "?"
    try:
        payload = load_target(run_dir, source_kind=infer_source_kind(run_dir))
        if payload.models:
            mid = str(payload.models[0].model_id or "").strip()
            if mid:
                model = mid
        if payload.runs:
            ds = str(payload.runs[0].dataset_ref or "").strip()
            if ds:
                dataset = ds
    except Exception:
        pass
    if model == "?" or dataset == "?":
        try:
            md = read_training_metadata(run_dir)
            ti = md.get("training_info") or {}
            if model == "?" and isinstance(ti, dict) and ti.get("model") is not None:
                model = str(ti["model"])
            if dataset == "?" and isinstance(ti, dict):
                ds_entry = ti.get("dataset") or {}
                if isinstance(ds_entry, dict) and ds_entry.get("name") is not None:
                    dataset = str(ds_entry["name"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError, FileNotFoundError):
            pass
    return {"model": model, "dataset": dataset}


def load_run_training_metadata(run_dir: str) -> dict[str, Any]:
    """Full training metadata for registry info / models-add.

    Prefer validating the run through the gateway first (fail-fast on broken refs),
    then return the on-disk training_metadata payload required for promotion naming.
    """
    try:
        load_target(run_dir, source_kind=infer_source_kind(run_dir))
    except Exception as exc:
        # Still allow promotion when metadata file exists but unified adapters are incomplete.
        meta_path = os.path.join(run_dir, "training_metadata.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"No training_metadata.json: {run_dir}") from exc
    return read_training_metadata(run_dir)
