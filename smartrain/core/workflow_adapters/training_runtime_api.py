from __future__ import annotations

from smartrain.core.runtime.dataset_hash import calculate_dataset_hash


def resolve_dataset_path_for_resume(run_dir: str, workspace_root: str) -> str | None:
    from smartrain.workflows.training.train_resume import resolve_dataset_path_for_resume as _impl

    return _impl(run_dir, workspace_root)


def diagnose_run(run_dir: str):
    from smartrain.workflows.training.train_resume import diagnose_run as _impl

    return _impl(run_dir)
