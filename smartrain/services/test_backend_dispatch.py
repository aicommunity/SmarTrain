from __future__ import annotations

from typing import Any


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

