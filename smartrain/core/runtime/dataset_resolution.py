"""Dataset path resolution primitives used by core workflows."""

from __future__ import annotations

import os
from typing import Any

from smartrain.core.runtime.workspace_paths import resolve_or_extract_dataset_root


def resolve_dataset_root_for_entry(
    dataset_name: str,
    info: dict[str, Any],
    *,
    workspace_root: str | None,
    source_catalog_dir: str,
    legacy_source_parent: str,
) -> str:
    """Resolve a dataset directory, extracting workspace-managed ZIP datasets as needed."""
    if workspace_root is not None:
        return resolve_or_extract_dataset_root(workspace_root, dataset_name, info, source_catalog_dir)
    if "data_path" in info:
        raw = info["data_path"]
        if not isinstance(raw, str):
            raise TypeError(f"data_path for {dataset_name!r} must be a string.")
        if os.path.isabs(raw):
            return os.path.abspath(raw)
        return os.path.abspath(os.path.join(legacy_source_parent, os.path.normpath(raw)))
    return os.path.join(legacy_source_parent, dataset_name)


def find_yaml_file(folder_path: str) -> str | None:
    """Find the first ``data.yaml`` or ``data.yml`` under a dataset directory."""
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower() in ("data.yaml", "data.yml"):
                return os.path.join(root, file_name)
    return None
