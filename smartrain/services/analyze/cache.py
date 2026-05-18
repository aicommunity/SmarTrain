from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from smartrain.core.runtime.run_artifacts import preferred_run_model_path, materialize_preferred_run_model


def run_cache_root(run_dir: str) -> str:
    root = os.path.join(os.path.abspath(run_dir), ".smartrain_cache", "analyze")
    os.makedirs(root, exist_ok=True)
    return root


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_fingerprint(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def data_yaml_hash(path: str) -> str:
    if not os.path.isfile(path):
        return "missing"
    return _sha256_file(path)[:16]


def weights_hash(run_dir: str) -> str:
    best = preferred_run_model_path(run_dir, ".pt")
    if not os.path.isfile(best):
        materialized = materialize_preferred_run_model(run_dir, ext=".pt", move=True, normalize_metadata=True)
        if materialized is not None:
            best = str(materialized)
    if not os.path.isfile(best):
        return "missing"
    return _sha256_file(best)[:16]


def cache_manifest_path(run_dir: str) -> str:
    return os.path.join(run_cache_root(run_dir), "cache_manifest.json")


def load_cache_manifest(run_dir: str) -> dict[str, Any]:
    path = cache_manifest_path(run_dir)
    if not os.path.isfile(path):
        return {"entries": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if isinstance(data, dict):
            data.setdefault("entries", [])
            return data
    except Exception:
        pass
    return {"entries": []}


def append_cache_entry(run_dir: str, entry: dict[str, Any]) -> None:
    data = load_cache_manifest(run_dir)
    entries = data.setdefault("entries", [])
    if isinstance(entries, list):
        entries.append(entry)
    with open(cache_manifest_path(run_dir), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

