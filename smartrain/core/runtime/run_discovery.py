from __future__ import annotations

import os

from smartrain.core.runtime.workspace_paths import WorkspaceLayout, resolve_workspace_root


def _looks_like_run_dir(path: str, filenames: set[str]) -> bool:
    root = os.path.abspath(path)
    canonical_best = os.path.join(root, "models", f"{os.path.basename(root)}.pt")
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


def discover_analysis_targets(
    *,
    workspace_cli: str | None = None,
    models_root_cli: str | None = None,
) -> list[str]:
    """Return run and promoted-model directories available for ``smartrain analyze``.

    Scans workspace ``runs/`` (or ``--models-root`` when set). Unless ``--models-root``
    overrides the default, also scans workspace ``models/`` for released model bundles.
    """
    runs_root = resolve_models_scan_root(workspace_cli, models_root_cli)
    seen: set[str] = set()
    out: list[str] = []
    for path in find_run_directories(runs_root):
        ap = os.path.abspath(path)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)
    if models_root_cli is None:
        try:
            ws = resolve_workspace_root(workspace_cli)
            models_root = WorkspaceLayout(ws).models
            if os.path.abspath(models_root) != os.path.abspath(runs_root):
                for path in find_run_directories(models_root):
                    ap = os.path.abspath(path)
                    if ap not in seen:
                        seen.add(ap)
                        out.append(ap)
        except ValueError:
            pass
    return sorted(out)


def is_run_directory(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        filenames = set(os.listdir(path))
    except OSError:
        return False
    return _looks_like_run_dir(path, filenames)

