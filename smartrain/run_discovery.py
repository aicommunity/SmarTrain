from __future__ import annotations

import os

from smartrain.workspace_paths import WorkspaceLayout, resolve_workspace_root


def resolve_models_scan_root(workspace_cli: str | None, models_root_cli: str | None) -> str:
    if models_root_cli is not None:
        return os.path.abspath(os.path.expanduser(models_root_cli))
    try:
        ws = resolve_workspace_root(workspace_cli)
        return WorkspaceLayout(ws).runs
    except ValueError:
        return os.path.abspath(os.getcwd())


def find_run_directories(models_root: str) -> list[str]:
    runs: list[str] = []
    models_root = os.path.abspath(models_root)
    if not os.path.isdir(models_root):
        return runs
    for dirpath, _, filenames in os.walk(models_root):
        if "training_metadata.json" in filenames:
            runs.append(dirpath)
    return sorted(runs)


def is_run_directory(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "training_metadata.json"))

