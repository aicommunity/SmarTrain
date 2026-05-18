from __future__ import annotations

def calculate_dataset_hash(dataset_path: str) -> str | None:
    from smartrain.services.datasets.dataset_hash import calculate_dataset_hash as _impl

    return _impl(dataset_path)


def resolve_dataset_path_for_resume(run_dir: str, workspace_root: str) -> str | None:
    from smartrain.workflows.training.train_resume import resolve_dataset_path_for_resume as _impl

    return _impl(run_dir, workspace_root)


def diagnose_run(run_dir: str):
    from smartrain.workflows.training.train_resume import diagnose_run as _impl

    return _impl(run_dir)
