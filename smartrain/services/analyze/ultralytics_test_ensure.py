"""Ensure canonical Ultralytics PT test artifacts exist before analyze report collection."""

from __future__ import annotations

import os
from typing import Any, Callable

from smartrain.core.runtime.run_artifacts import preferred_run_model_path
from smartrain.core.runtime.workspace_paths import resolve_workspace_root
from smartrain.services.testing.model_test_service import (
    complete_missing_test_artifacts,
    has_complete_test_artifacts,
    persist_target_test_artifacts_state,
)
from smartrain.services.training.train_resume_pt_test_runner import resume_ultralytics_pt_test_runner


def _ensure_enabled_for_profile(profile: str, args: Any) -> bool:
    explicit = getattr(args, "ensure_ultralytics_test", None)
    if explicit is not None:
        return bool(explicit)
    return str(profile or "").strip().lower() == "full"


def ensure_ultralytics_test_for_runs(
    run_dirs: list[str],
    *,
    args: Any,
    profile: str,
    workspace_cli: str | None,
    run_data_yaml_map: dict[str, str],
    record_failure_cb: Callable[..., None],
) -> None:
    if not _ensure_enabled_for_profile(profile, args):
        return
    try:
        workspace_root = resolve_workspace_root(workspace_cli)
    except ValueError:
        workspace_root = os.getcwd()

    val_batch = int(getattr(args, "val_batch", 1) or 1)
    val_imgsz = int(getattr(args, "val_imgsz", 640) or 640)

    for run_dir in run_dirs:
        rd = os.path.abspath(run_dir)
        label = os.path.basename(rd.rstrip(os.sep))
        if has_complete_test_artifacts(rd, "pt"):
            canonical_pt = preferred_run_model_path(rd, ".pt")
            if os.path.isfile(canonical_pt):
                persist_target_test_artifacts_state(
                    rd,
                    format_name="pt",
                    target_path=canonical_pt,
                    backend="ultralytics",
                    status="ok",
                )
            continue
        print(f"[INFO] Ultralytics test incomplete for {label}; running PT test...")
        try:
            complete_missing_test_artifacts(
                rd,
                workspace_root=workspace_root,
                pt_test_runner=resume_ultralytics_pt_test_runner,
                pt_test_runner_kwargs={
                    "non_interactive": True,
                    "val_batch": val_batch,
                    "val_imgsz": val_imgsz,
                },
            )
            if not has_complete_test_artifacts(rd, "pt"):
                record_failure_cb(
                    stage="ultralytics_test_ensure",
                    status="missing",
                    reason_code="test_incomplete",
                    reason_detail="PT test finished but rich artifacts are still incomplete",
                    run_dir=rd,
                    format_name="pt",
                    split="test",
                )
        except Exception as exc:
            record_failure_cb(
                stage="ultralytics_test_ensure",
                status="failed",
                reason_code="test_run_failed",
                reason_detail=str(exc),
                run_dir=rd,
                format_name="pt",
                split="test",
            )
            print(f"[WARN] {label}: Ultralytics PT test failed: {exc}")
