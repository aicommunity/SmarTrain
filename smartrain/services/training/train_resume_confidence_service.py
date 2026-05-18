from __future__ import annotations

import os

from smartrain.core.runtime.run_artifacts import preferred_run_model_path
from smartrain.core.training.confidence_recommendation import (
    read_recommendation_file,
    recommendation_file_path,
    recommendations_complete,
)
from smartrain.core.workflow_adapters.training_runtime_api import resolve_dataset_path_for_resume
from smartrain.services.train_runtime_helpers import maybe_free_cuda_memory
from smartrain.services.training.train_yolo_execution_service import test_yolo


def ensure_resume_confidence_recommendations(
    run_dir: str,
    workspace_root: str,
    val_batch: int = 1,
) -> None:
    test_payload = read_recommendation_file(recommendation_file_path(run_dir, "test"))
    val_payload = read_recommendation_file(recommendation_file_path(run_dir, "val"))
    if recommendations_complete(test_payload) and recommendations_complete(val_payload):
        return

    best_pt = preferred_run_model_path(run_dir, ".pt")
    if not os.path.isfile(best_pt):
        print(
            "[WARN] Resume post-check: recommendations missing but canonical run model is absent; "
            "cannot recompute confidence recommendations."
        )
        return

    dataset_path = resolve_dataset_path_for_resume(run_dir, workspace_root)
    if not dataset_path:
        print(
            "[WARN] Resume post-check: recommendations missing but dataset path is unresolved; "
            "cannot recompute confidence recommendations."
        )
        return

    print("[INFO] Resume post-check: recomputing missing confidence recommendations (val/test).")
    maybe_free_cuda_memory()
    test_yolo(run_dir, dataset_path, val_batch=max(1, int(val_batch)), non_interactive=True)
