from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Any

from smartrain.core.runtime.path_portable import store_path_under_workspace
from smartrain.core.runtime.workspace_coordination import catalog_write_lock, get_active_session
from smartrain.core.runtime.workspace_paths import WorkspaceLayout

_CATALOG_RMW_LOCK = threading.Lock()


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
        os.replace(tmp, path)
        tmp = ""
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def load_dataset_catalog(layout: WorkspaceLayout) -> dict:
    info_path = layout.work_datasets_info_path()
    if not os.path.isfile(info_path):
        return {}
    with open(info_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def sorted_class_names_union_from_catalog(catalog: dict[str, Any]) -> list[str]:
    """All normalized class name keys across datasets_info entries (used for interactive prompts)."""
    names = {k for v in catalog.values() if isinstance(v, dict) for k in (v.get("classes") or {}).keys()}
    return sorted(names)


def sorted_class_names_for_dataset(catalog: dict[str, Any], dataset_key: str) -> list[str]:
    """Class name keys from a single datasets_info entry."""
    entry = catalog.get(dataset_key)
    if not isinstance(entry, dict):
        return []
    classes = entry.get("classes")
    if not isinstance(classes, dict):
        return []
    return sorted(str(k) for k in classes.keys())


def detect_split_from_path(images_path: str, *, prefer_valid_name: bool = False) -> str:
    low = str(images_path).lower()
    if "/train/" in low:
        return "train"
    if "/valid/" in low:
        return "valid" if prefer_valid_name else "val"
    if "/val/" in low:
        return "val"
    if "/test/" in low:
        return "test"
    return "train"


def update_datasets_sidecar(
    *,
    layout: WorkspaceLayout,
    output_key: str,
    class_map: dict[str, int],
    target_dir: str,
    output_hash: str,
    structure: str = "split",
) -> None:
    os.makedirs(layout.datasets, exist_ok=True)
    rel = store_path_under_workspace(layout.root, os.path.abspath(target_dir))
    entry = {
        "classes": {str(k): int(v) for k, v in sorted(class_map.items(), key=lambda kv: int(kv[1]))},
        "structure": str(structure),
        "elements_count": None,
        "data_path": rel,
        "dataset_hash": output_hash,
        "modified": False,
    }
    _merge_catalog_entry(layout, output_key, entry)


def update_datasets_sidecar_from_entry(
    *,
    layout: WorkspaceLayout,
    output_key: str,
    entry: dict[str, Any],
    target_dir: str,
    output_hash: str,
) -> None:
    os.makedirs(layout.datasets, exist_ok=True)
    rel = store_path_under_workspace(layout.root, os.path.abspath(target_dir))
    new_entry = dict(entry) if isinstance(entry, dict) else {}
    new_entry["data_path"] = rel
    new_entry["dataset_hash"] = output_hash
    new_entry["modified"] = False
    _merge_catalog_entry(layout, output_key, new_entry)


def _merge_catalog_entry(layout: WorkspaceLayout, output_key: str, entry: dict[str, Any]) -> None:
    info_path = layout.work_datasets_info_path()
    cn_path = layout.work_class_names_path()
    classes = entry.get("classes") if isinstance(entry.get("classes"), dict) else {}

    with _CATALOG_RMW_LOCK:
        with catalog_write_lock(layout, session=get_active_session()):
            previous: dict[str, Any] = {}
            if os.path.isfile(info_path):
                with open(info_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    previous = loaded
            previous[output_key] = entry
            _write_json_atomic(info_path, previous)

            class_names_out: dict[str, str] = {}
            if os.path.isfile(cn_path):
                with open(cn_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    class_names_out = {str(k): str(v) for k, v in loaded.items()}
            for c in classes.keys():
                class_names_out[str(c)] = str(c)
            _write_json_atomic(cn_path, class_names_out)
