"""Post-parse model test execution (interactive prompts + artifact pipeline)."""

from __future__ import annotations

import argparse
from typing import Any


def run_model_test_after_setup(
    *,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    request: Any,
    workspace_root: str,
    interactive: bool,
    root_dir: str,
    primary_path: str,
    target_kind: str,
    target_label: str | None,
    data_yaml: str,
    formats: list[str],
    onnx_provider_policy: str,
    requested_imgsz: int | None,
    requested_conf: float | None,
    requested_iou: float | None,
) -> None:
    from smartrain.workflows.testing import model_test_cli as mtc
    from smartrain.backends.train_test_registry import resolve_test_backend
    from smartrain.cli_support.cli_contracts import emit_replay
    from smartrain.workflows.testing.model_test_service import (
        SUPPORTED_TEST_FORMATS,
        has_complete_test_artifacts,
        persist_target_test_artifacts_state,
    )
    from smartrain.services.test_backend_dispatch import (
        run_internal_pt_uni_backend,
        run_non_pt_test_backend,
        run_pt_test_backend,
    )
    from smartrain.core.training.train_profile import task_to_metadata_task_type

    task_type = task_to_metadata_task_type(getattr(args, "task", None))

    def _backend_for(fmt: str) -> str:
        return resolve_test_backend(task_type=task_type, model_format=fmt).backend

    replay: str | None = None
    results: list[tuple[str, bool, str | None]] = []
    selected_artifacts: list[tuple[str, str]] = []
    if interactive and target_kind == "runs":
        candidates = mtc._discover_run_artifact_candidates(root_dir)
        enabled_formats = mtc._prompt_export_backends_interactive(root_dir, candidates)
        formats = [fmt for fmt in SUPPORTED_TEST_FORMATS if fmt in enabled_formats]
        args.formats = ",".join(formats)
        narrowed = {fmt: candidates[fmt] for fmt in formats if candidates.get(fmt)}
        selected_artifacts = mtc._prompt_artifact_selection_interactive(narrowed)
        selected_formats = {fmt for fmt, _ in selected_artifacts}
        if selected_formats:
            formats = [fmt for fmt in SUPPORTED_TEST_FORMATS if fmt in selected_formats]
            args.formats = ",".join(formats)
    mtc._print_test_plan(
        target_kind=target_kind,
        target_label=target_label,
        root_dir=root_dir,
        data_yaml=data_yaml,
        formats=formats,
        split_name="test",
    )
    replay = emit_replay(command_name="test", parser=parser, args=args, stage="before execution")
    predecisions = mtc._collect_interactive_rerun_decisions(
        interactive=interactive,
        force=bool(args.force),
        missing_only=bool(args.missing_only),
        root_dir=root_dir,
        target_kind=target_kind,
        primary_path=primary_path,
        formats=formats,
        selected_artifacts=selected_artifacts,
        data_yaml=data_yaml,
        requested_imgsz=requested_imgsz,
        requested_conf=requested_conf,
        requested_iou=requested_iou,
        deep_diagnostics=bool(args.deep_diagnostics),
    )

    if "pt" in formats:
        if target_kind == "runs" and (not args.missing_only or not has_complete_test_artifacts(root_dir, "pt")):
            print(f"  model[pt]: {primary_path}")
            should_rerun = predecisions.get(mtc._artifact_key("pt", primary_path))
            if should_rerun is None:
                should_rerun = mtc._should_rerun_existing_match(
                    interactive=interactive,
                    force=bool(args.force),
                    root_dir=root_dir,
                    format_name="pt",
                    target_path=primary_path,
                    dataset_yaml=data_yaml,
                    imgsz=requested_imgsz,
                    conf=requested_conf,
                    iou=requested_iou,
                    deep_diagnostics=bool(args.deep_diagnostics),
                )
            if not should_rerun:
                results.append(("pt", True, None))
            else:
                ok, err = run_pt_test_backend(
                    task_type=task_type,
                    target_kind="runs",
                    root_dir=root_dir,
                    primary_path=primary_path,
                    data_yaml=data_yaml,
                    workspace_root=workspace_root,
                    args=args,
                )
                results.append(("pt", ok, err))
        elif target_kind in {"models", "weights"} and (not args.missing_only or not has_complete_test_artifacts(root_dir, "pt")):
            print(f"  model[pt]: {primary_path}")
            should_rerun = predecisions.get(mtc._artifact_key("pt", primary_path))
            if should_rerun is None:
                should_rerun = mtc._should_rerun_existing_match(
                    interactive=interactive,
                    force=bool(args.force),
                    root_dir=root_dir,
                    format_name="pt",
                    target_path=primary_path,
                    dataset_yaml=data_yaml,
                    imgsz=requested_imgsz,
                    conf=requested_conf,
                    iou=requested_iou,
                    deep_diagnostics=bool(args.deep_diagnostics),
                )
            if not should_rerun:
                results.append(("pt", True, None))
            else:
                ok, err = run_pt_test_backend(
                    task_type=task_type,
                    target_kind=target_kind,
                    root_dir=root_dir,
                    primary_path=primary_path,
                    data_yaml=data_yaml,
                    workspace_root=workspace_root,
                    args=args,
                )
                results.append(("pt", ok, err))

    # Internal-only unified PT evaluation for PT vs PT-uni compare table.
    # Keep this path detection-only until dedicated cls/seg compare contract exists.
    if "pt" in formats and task_type == "detection":
        run_internal_pt_uni = bool(args.force) or (not args.missing_only) or (not has_complete_test_artifacts(root_dir, "pt_uni"))
        if run_internal_pt_uni:
            should_rerun_pt_uni = mtc._should_rerun_existing_match(
                interactive=False,
                force=bool(args.force),
                root_dir=root_dir,
                format_name="pt_uni",
                target_path=primary_path,
                dataset_yaml=data_yaml,
                imgsz=requested_imgsz,
                conf=requested_conf,
                iou=requested_iou,
                deep_diagnostics=bool(args.deep_diagnostics),
            )
            if should_rerun_pt_uni:
                print("[INFO] Generating internal PT-vs-PT-uni comparison artifacts.")
                _ok, err = run_internal_pt_uni_backend(
                    root_dir=root_dir,
                    primary_path=primary_path,
                    data_yaml=data_yaml,
                    args=args,
                    onnx_provider_policy=onnx_provider_policy,
                )
                if not _ok:
                    print(f"[WARN] Internal pt_uni compare artifacts failed: {err}")
    elif "pt" in formats:
        print(f"[INFO] Skipping internal pt_uni compare for task={task_type!r}; detection-only path.")

    queued: list[tuple[str, str]] = []
    if selected_artifacts:
        for fmt, path in selected_artifacts:
            # .pt is evaluated earlier via Ultralytics; native runner supports only onnx/engine/trt.
            if fmt in {"onnx", "engine", "trt"}:
                queued.append((fmt, path))
    else:
        for fmt in ("onnx", "engine", "trt"):
            if fmt not in formats:
                continue
            if args.missing_only and has_complete_test_artifacts(root_dir, fmt):
                continue
            try:
                artifact_path = mtc._resolve_existing_artifact(
                    root_dir=root_dir,
                    primary_path=primary_path,
                    format_name=fmt,
                    target_kind=target_kind,
                )
            except Exception as exc:
                backend = _backend_for(fmt)
                persist_target_test_artifacts_state(
                    root_dir,
                    format_name=fmt,
                    target_path=None,
                    dataset_yaml=data_yaml,
                    backend=backend,
                    status="failed",
                    error=str(exc),
                )
                results.append((fmt, False, str(exc)))
                continue
            queued.append((fmt, artifact_path))

    for fmt, artifact_path in queued:
        try:
            print(f"  model[{fmt}]: {artifact_path}")
            should_rerun = predecisions.get(mtc._artifact_key(fmt, artifact_path))
            if should_rerun is None:
                should_rerun = mtc._should_rerun_existing_match(
                    interactive=interactive,
                    force=bool(args.force),
                    root_dir=root_dir,
                    format_name=fmt,
                    target_path=artifact_path,
                    dataset_yaml=data_yaml,
                    imgsz=requested_imgsz,
                    conf=requested_conf,
                    iou=requested_iou,
                    deep_diagnostics=bool(args.deep_diagnostics),
                )
            if not should_rerun:
                results.append((fmt, True, None))
                continue
            ok, err = run_non_pt_test_backend(
                task_type=task_type,
                fmt=fmt,
                artifact_path=artifact_path,
                root_dir=root_dir,
                data_yaml=data_yaml,
                args=args,
                onnx_provider_policy=onnx_provider_policy,
            )
            results.append((fmt, ok, err))
        except Exception as exc:
            backend = _backend_for(fmt)
            persist_target_test_artifacts_state(
                root_dir,
                format_name=fmt,
                target_path=None,
                dataset_yaml=data_yaml,
                backend=backend,
                status="failed",
                error=str(exc),
            )
            results.append((fmt, False, str(exc)))

    if not results:
        print(f"[INFO] No test artifacts needed for target: {target_label or root_dir}")
    else:
        for fmt, ok, error in results:
            if ok:
                print(f"[OK] {fmt}: artifacts are ready in {root_dir}")
            else:
                print(f"[WARN] {fmt}: {error}")
    if replay:
        request.interactive_used = bool(interactive)
        emit_replay(command_name="test", parser=parser, args=args, stage="after execution")
