"""Ensure confidence recommendation JSON exists for analyze sessions."""

from __future__ import annotations

import os
from typing import Any

from smartrain.core.runtime.run_artifacts import preferred_run_model_path
from smartrain.core.training.confidence_recommendation import (
    read_recommendation_file,
    recommendation_file_path,
    recommendations_complete,
    write_not_available_recommendations,
)
from smartrain.core.workflow_adapters.training_runtime_api import resolve_dataset_path_for_resume
from smartrain.services.training.train_resume_pt_test_runner import resume_ultralytics_pt_test_runner
from smartrain.services.training.train_runtime_data_yaml_service import coerce_dataset_root


def ensure_confidence_recommendations_for_analyze_runs(
    run_dirs: list[str],
    *,
    run_data_yaml_map: dict[str, str] | None = None,
    workspace_root: str,
    val_batch: int = 1,
    val_imgsz: int = 640,
) -> None:
    run_data_yaml_map = run_data_yaml_map or {}
    for run_dir in run_dirs:
        rd = os.path.abspath(run_dir.rstrip(os.sep))
        label = os.path.basename(rd)
        test_path = recommendation_file_path(rd, "test")
        if recommendations_complete(read_recommendation_file(test_path)):
            continue
        weights = preferred_run_model_path(rd, ".pt")
        if not os.path.isfile(weights):
            try:
                write_not_available_recommendations(
                    model_dir=rd,
                    split="test",
                    reason="canonical_pt_missing",
                )
            except OSError as exc:
                print(f"[WARN] {label}: confidence skipped — PT model not found ({exc})")
            else:
                print(f"[WARN] {label}: confidence skipped — PT model not found.")
            continue
        # run_data_yaml_map stores paths to data.yaml files; the PT runner expects a dataset root.
        dataset_path = str(run_data_yaml_map.get(rd) or run_data_yaml_map.get(run_dir) or "").strip()
        if not dataset_path:
            dataset_path = resolve_dataset_path_for_resume(rd, workspace_root) or ""
        if dataset_path:
            try:
                dataset_path, _src_yaml = coerce_dataset_root(dataset_path)
            except ValueError:
                dataset_path = ""
        if not dataset_path:
            try:
                write_not_available_recommendations(
                    model_dir=rd,
                    split="test",
                    reason="dataset_unresolved",
                )
            except OSError as exc:
                print(f"[WARN] {label}: confidence skipped — dataset path unresolved ({exc})")
            else:
                print(f"[WARN] {label}: confidence skipped — dataset path unresolved.")
            continue
        print(f"[INFO] {label}: computing missing confidence recommendations (test/val)...")
        try:
            resume_ultralytics_pt_test_runner(
                rd,
                dataset_path,
                val_batch=max(1, int(val_batch)),
                val_imgsz=int(val_imgsz),
                non_interactive=True,
            )
        except Exception as exc:
            try:
                write_not_available_recommendations(
                    model_dir=rd,
                    split="test",
                    reason=f"confidence_compute_failed: {exc}",
                )
            except OSError:
                pass
            print(f"[WARN] {label}: confidence compute failed: {exc}")
