from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from smartrain.external_providers.runner import run_external_infer, run_external_train
from smartrain.services.train_runtime_helpers import (
    build_run_name,
    json_safe_train_summary,
    load_batch_from_training_metadata,
    resolve_external_eval_source,
)
from smartrain.services.testing.model_test_service import sync_test_artifacts_manifest
from smartrain.services.training.train_metadata_io_service import save_training_metadata as _save_training_metadata_svc
from smartrain.services.training.train_system_profile_service import collect_system_profile
from smartrain.services.training.train_yolo_execution_service import test_yolo as _test_yolo
from smartrain.services.training.train_yolo_execution_service import train_yolo as _train_yolo
from smartrain.services.training.train_yolo_hooks import build_train_yolo_hooks


@dataclass(frozen=True)
class TrainRuntimeOps:
    """Runtime operations bundle for train/test execution."""

    train_yolo: Callable[..., Any]
    test_yolo: Callable[..., Any]
    save_training_metadata: Callable[..., Any]
    collect_system_profile: Callable[..., Any]
    build_run_name: Callable[..., Any]
    resolve_external_eval_source: Callable[..., Any]
    json_safe_train_summary: Callable[..., Any]
    load_batch_from_training_metadata: Callable[..., Any]
    run_external_train: Callable[..., Any]
    run_external_infer: Callable[..., Any]


def _train_yolo_with_hooks(**kwargs: Any) -> Any:
    return _train_yolo(**kwargs, hooks=build_train_yolo_hooks())


def _save_training_metadata(**kwargs: Any) -> None:
    _save_training_metadata_svc(
        **kwargs,
        sync_test_artifacts_manifest_cb=sync_test_artifacts_manifest,
    )


def build_train_runtime_ops() -> TrainRuntimeOps:
    return TrainRuntimeOps(
        train_yolo=_train_yolo_with_hooks,
        test_yolo=_test_yolo,
        save_training_metadata=_save_training_metadata,
        collect_system_profile=collect_system_profile,
        build_run_name=build_run_name,
        resolve_external_eval_source=resolve_external_eval_source,
        json_safe_train_summary=json_safe_train_summary,
        load_batch_from_training_metadata=load_batch_from_training_metadata,
        run_external_train=run_external_train,
        run_external_infer=run_external_infer,
    )
