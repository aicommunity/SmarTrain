from __future__ import annotations

import os

from smartrain.run_artifacts import canonical_run_model_path
from smartrain.workspace_paths import WorkspaceLayout, resolve_workspace_root


def _looks_like_run_dir(path: str, filenames: set[str]) -> bool:
    canonical_best = canonical_run_model_path(path, ".pt")
    train_dir = os.path.join(path, "train")
    has_train_artifacts = (
        os.path.isfile(os.path.join(train_dir, "args.yaml"))
        or os.path.isfile(os.path.join(train_dir, "results.csv"))
        or os.path.isfile(os.path.join(train_dir, "weights", "last.pt"))
        or os.path.isfile(canonical_best)
    )
    return "training_metadata.json" in filenames or has_train_artifacts


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
        if _looks_like_run_dir(dirpath, set(filenames)):
            runs.append(dirpath)
    return sorted(runs)


def is_run_directory(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        filenames = set(os.listdir(path))
    except OSError:
        return False
    return _looks_like_run_dir(path, filenames)

