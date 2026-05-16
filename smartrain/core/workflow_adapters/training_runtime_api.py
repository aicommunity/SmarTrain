from __future__ import annotations

from types import ModuleType


def calculate_dataset_hash(dataset_path: str) -> str | None:
    from smartrain.services.datasets.dataset_hash import calculate_dataset_hash as _impl

    return _impl(dataset_path)


def get_training_module_api() -> ModuleType:
    from smartrain.workflows.training import model_training_module as mtm

    return mtm
