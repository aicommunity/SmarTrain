from __future__ import annotations

from typing import Any


def run_pt_test_backend(
    *,
    task_type: str,
    target_kind: str,
    root_dir: str,
    primary_path: str,
    data_yaml: str,
    workspace_root: str,
    args: Any,
) -> tuple[bool, str | None]:
    """
    Ultralytics PT test path for model test orchestration.

    Run targets use complete_missing_test_artifacts unless deep diagnostics or perf
    collection is requested; model/weight targets always use the isolated runner.
    """
    from smartrain import model_test_cli as mtc
    from smartrain.backends.train_test_registry import resolve_test_backend
    from smartrain.model_test_service import persist_target_test_artifacts_state

    def _backend_for(local_fmt: str) -> str:
        return resolve_test_backend(task_type=task_type, model_format=local_fmt).backend

    if target_kind == "runs":
        if bool(args.deep_diagnostics) or bool(args.perf):
            pt_result = mtc.run_ultralytics_backend(
                root_dir=root_dir,
                weights_path=primary_path,
                dataset_yaml_path=data_yaml,
                format_name="pt",
                imgsz=args.imgsz,
                val_conf=args.conf,
                val_iou=args.iou,
                val_batch=args.batch,
                deep_diagnostics=bool(args.deep_diagnostics),
                collect_performance=bool(args.perf),
                perf_warmup_images=int(max(0, args.perf_warmup_images)),
                runtime_device=args.device,
            )
            return pt_result.success, pt_result.error
        mtc.complete_missing_test_artifacts(
            root_dir,
            workspace_root=workspace_root,
            pt_test_runner=__import__("smartrain.model_training_module", fromlist=["test_yolo"]).test_yolo,
            pt_test_runner_kwargs={
                "val_imgsz": args.imgsz,
                "val_conf": args.conf,
                "val_iou": args.iou,
                "val_batch": args.batch,
            },
        )
        persist_target_test_artifacts_state(
            root_dir,
            format_name="pt",
            target_path=primary_path,
            dataset_yaml=data_yaml,
            backend=_backend_for("pt"),
            status="ok",
        )
        return True, None

    pt_result = mtc.run_ultralytics_backend(
        root_dir=root_dir,
        weights_path=primary_path,
        dataset_yaml_path=data_yaml,
        format_name="pt",
        imgsz=args.imgsz,
        val_conf=args.conf,
        val_iou=args.iou,
        val_batch=args.batch,
        deep_diagnostics=bool(args.deep_diagnostics),
        collect_performance=bool(args.perf),
        perf_warmup_images=int(max(0, args.perf_warmup_images)),
        runtime_device=args.device,
    )
    return pt_result.success, pt_result.error


def run_internal_pt_uni_backend(
    *,
    root_dir: str,
    primary_path: str,
    data_yaml: str,
    args: Any,
    onnx_provider_policy: str,
) -> tuple[bool, str | None]:
    """Internal PT-uni comparison artifacts (same weights as primary PT path)."""
    from smartrain import model_test_cli as mtc

    pt_uni_result = mtc.run_native_format_backend(
        root_dir=root_dir,
        weights_path=primary_path,
        dataset_yaml_path=data_yaml,
        format_name="pt_uni",
        imgsz=args.imgsz,
        val_conf=args.conf,
        val_iou=args.iou,
        val_batch=args.batch,
        deep_diagnostics=bool(args.deep_diagnostics),
        collect_performance=bool(args.perf),
        perf_warmup_images=int(max(0, args.perf_warmup_images)),
        onnx_provider_policy=onnx_provider_policy,
        runtime_device=args.device,
    )
    if not pt_uni_result.success:
        return False, pt_uni_result.error
    return True, None


def run_non_pt_test_backend(
    *,
    task_type: str,
    fmt: str,
    artifact_path: str,
    root_dir: str,
    data_yaml: str,
    args: Any,
    onnx_provider_policy: str,
) -> tuple[bool, str | None]:
    """
    Unified non-PT backend execution for model test orchestration.

    Returns (ok, error).
    """
    from smartrain import model_test_cli as mtc
    from smartrain.backends.train_test_registry import resolve_test_backend
    from smartrain.model_test_service import persist_target_test_artifacts_state

    def _backend_for(local_fmt: str) -> str:
        return resolve_test_backend(task_type=task_type, model_format=local_fmt).backend

    if fmt in {"engine", "trt"}:
        preflight_ok, preflight_reason = mtc._check_native_format_preflight(fmt)
        if not preflight_ok:
            persist_target_test_artifacts_state(
                root_dir,
                format_name=fmt,
                target_path=artifact_path,
                dataset_yaml=data_yaml,
                backend=_backend_for(fmt),
                status="failed",
                error=preflight_reason,
            )
            return False, preflight_reason
        ok, err = mtc._run_native_backend_isolated(
            root_dir=root_dir,
            weights_path=artifact_path,
            dataset_yaml_path=data_yaml,
            format_name=fmt,
            imgsz=args.imgsz,
            val_conf=args.conf,
            val_iou=args.iou,
            val_batch=args.batch,
            collect_performance=bool(args.perf),
            perf_warmup_images=int(max(0, args.perf_warmup_images)),
            runtime_device=args.device,
        )
        if not ok:
            persist_target_test_artifacts_state(
                root_dir,
                format_name=fmt,
                target_path=artifact_path,
                dataset_yaml=data_yaml,
                backend=_backend_for(fmt),
                status="failed",
                error=err,
            )
        return ok, err

    if fmt == "onnx":
        onnx_ok, onnx_reason = mtc._check_onnx_format_preflight(onnx_provider_policy)
        if not onnx_ok:
            persist_target_test_artifacts_state(
                root_dir,
                format_name=fmt,
                target_path=artifact_path,
                dataset_yaml=data_yaml,
                backend=_backend_for(fmt),
                status="failed",
                error=onnx_reason,
            )
            return False, onnx_reason
        if onnx_reason:
            print(f"[WARN] onnx preflight: {onnx_reason}")

    result = mtc.run_native_format_backend(
        root_dir=root_dir,
        weights_path=artifact_path,
        dataset_yaml_path=data_yaml,
        format_name=fmt,
        imgsz=args.imgsz,
        val_conf=args.conf,
        val_iou=args.iou,
        val_batch=args.batch,
        deep_diagnostics=bool(args.deep_diagnostics),
        collect_performance=bool(args.perf),
        perf_warmup_images=int(max(0, args.perf_warmup_images)),
        onnx_provider_policy=onnx_provider_policy if fmt == "onnx" else None,
        runtime_device=args.device,
    )
    return result.success, result.error

