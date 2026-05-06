from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class TestBackendDispatchContext:
    task_type: str
    fmt: str
    target_kind: str
    root_dir: str
    primary_path: str
    artifact_path: str
    data_yaml: str
    workspace_root: str
    args: Any
    onnx_provider_policy: str


def _dispatch_pt(ctx: TestBackendDispatchContext) -> tuple[bool, str | None]:
    from smartrain.workflows.testing import model_test_cli as mtc
    from smartrain.backends.train_test_registry import resolve_test_backend
    from smartrain.workflows.testing.model_test_service import persist_target_test_artifacts_state

    def _backend_for(local_fmt: str) -> str:
        return resolve_test_backend(task_type=ctx.task_type, model_format=local_fmt).backend

    if ctx.target_kind == "runs":
        if bool(ctx.args.deep_diagnostics) or bool(ctx.args.perf):
            pt_result = mtc.run_ultralytics_backend(
                root_dir=ctx.root_dir,
                weights_path=ctx.primary_path,
                dataset_yaml_path=ctx.data_yaml,
                format_name="pt",
                imgsz=ctx.args.imgsz,
                val_conf=ctx.args.conf,
                val_iou=ctx.args.iou,
                val_batch=ctx.args.batch,
                deep_diagnostics=bool(ctx.args.deep_diagnostics),
                collect_performance=bool(ctx.args.perf),
                perf_warmup_images=int(max(0, ctx.args.perf_warmup_images)),
                runtime_device=ctx.args.device,
            )
            return pt_result.success, pt_result.error
        mtc.complete_missing_test_artifacts(
            ctx.root_dir,
            workspace_root=ctx.workspace_root,
            pt_test_runner=__import__("smartrain.workflows.training.model_training_module", fromlist=["test_yolo"]).test_yolo,
            pt_test_runner_kwargs={
                "val_imgsz": ctx.args.imgsz,
                "val_conf": ctx.args.conf,
                "val_iou": ctx.args.iou,
                "val_batch": ctx.args.batch,
            },
        )
        persist_target_test_artifacts_state(
            ctx.root_dir,
            format_name="pt",
            target_path=ctx.primary_path,
            dataset_yaml=ctx.data_yaml,
            backend=_backend_for("pt"),
            status="ok",
        )
        return True, None

    pt_result = mtc.run_ultralytics_backend(
        root_dir=ctx.root_dir,
        weights_path=ctx.primary_path,
        dataset_yaml_path=ctx.data_yaml,
        format_name="pt",
        imgsz=ctx.args.imgsz,
        val_conf=ctx.args.conf,
        val_iou=ctx.args.iou,
        val_batch=ctx.args.batch,
        deep_diagnostics=bool(ctx.args.deep_diagnostics),
        collect_performance=bool(ctx.args.perf),
        perf_warmup_images=int(max(0, ctx.args.perf_warmup_images)),
        runtime_device=ctx.args.device,
    )
    return pt_result.success, pt_result.error


def _dispatch_pt_uni(ctx: TestBackendDispatchContext) -> tuple[bool, str | None]:
    from smartrain.workflows.testing import model_test_cli as mtc

    pt_uni_result = mtc.run_native_format_backend(
        root_dir=ctx.root_dir,
        weights_path=ctx.primary_path,
        dataset_yaml_path=ctx.data_yaml,
        format_name="pt_uni",
        imgsz=ctx.args.imgsz,
        val_conf=ctx.args.conf,
        val_iou=ctx.args.iou,
        val_batch=ctx.args.batch,
        deep_diagnostics=bool(ctx.args.deep_diagnostics),
        collect_performance=bool(ctx.args.perf),
        perf_warmup_images=int(max(0, ctx.args.perf_warmup_images)),
        onnx_provider_policy=ctx.onnx_provider_policy,
        runtime_device=ctx.args.device,
    )
    if not pt_uni_result.success:
        return False, pt_uni_result.error
    return True, None


def _dispatch_non_pt(ctx: TestBackendDispatchContext) -> tuple[bool, str | None]:
    from smartrain.workflows.testing import model_test_cli as mtc
    from smartrain.backends.train_test_registry import resolve_test_backend
    from smartrain.workflows.testing.model_test_service import persist_target_test_artifacts_state

    def _backend_for(local_fmt: str) -> str:
        return resolve_test_backend(task_type=ctx.task_type, model_format=local_fmt).backend

    if ctx.fmt in {"engine", "trt"}:
        preflight_ok, preflight_reason = mtc._check_native_format_preflight(ctx.fmt)
        if not preflight_ok:
            persist_target_test_artifacts_state(
                ctx.root_dir,
                format_name=ctx.fmt,
                target_path=ctx.artifact_path,
                dataset_yaml=ctx.data_yaml,
                backend=_backend_for(ctx.fmt),
                status="failed",
                error=preflight_reason,
            )
            return False, preflight_reason
        ok, err = mtc._run_native_backend_isolated(
            root_dir=ctx.root_dir,
            weights_path=ctx.artifact_path,
            dataset_yaml_path=ctx.data_yaml,
            format_name=ctx.fmt,
            imgsz=ctx.args.imgsz,
            val_conf=ctx.args.conf,
            val_iou=ctx.args.iou,
            val_batch=ctx.args.batch,
            collect_performance=bool(ctx.args.perf),
            perf_warmup_images=int(max(0, ctx.args.perf_warmup_images)),
            runtime_device=ctx.args.device,
        )
        if not ok:
            persist_target_test_artifacts_state(
                ctx.root_dir,
                format_name=ctx.fmt,
                target_path=ctx.artifact_path,
                dataset_yaml=ctx.data_yaml,
                backend=_backend_for(ctx.fmt),
                status="failed",
                error=err,
            )
        return ok, err

    if ctx.fmt == "onnx":
        onnx_ok, onnx_reason = mtc._check_onnx_format_preflight(ctx.onnx_provider_policy)
        if not onnx_ok:
            persist_target_test_artifacts_state(
                ctx.root_dir,
                format_name=ctx.fmt,
                target_path=ctx.artifact_path,
                dataset_yaml=ctx.data_yaml,
                backend=_backend_for(ctx.fmt),
                status="failed",
                error=onnx_reason,
            )
            return False, onnx_reason
        if onnx_reason:
            print(f"[WARN] onnx preflight: {onnx_reason}")

    result = mtc.run_native_format_backend(
        root_dir=ctx.root_dir,
        weights_path=ctx.artifact_path,
        dataset_yaml_path=ctx.data_yaml,
        format_name=ctx.fmt,
        imgsz=ctx.args.imgsz,
        val_conf=ctx.args.conf,
        val_iou=ctx.args.iou,
        val_batch=ctx.args.batch,
        deep_diagnostics=bool(ctx.args.deep_diagnostics),
        collect_performance=bool(ctx.args.perf),
        perf_warmup_images=int(max(0, ctx.args.perf_warmup_images)),
        onnx_provider_policy=ctx.onnx_provider_policy if ctx.fmt == "onnx" else None,
        runtime_device=ctx.args.device,
    )
    return result.success, result.error


class TestBackendStrategy(Protocol):
    def run(self, ctx: TestBackendDispatchContext) -> tuple[bool, str | None]:
        ...


@dataclass(frozen=True)
class PtStrategy:
    def run(self, ctx: TestBackendDispatchContext) -> tuple[bool, str | None]:
        return _dispatch_pt(ctx)


@dataclass(frozen=True)
class PtUniStrategy:
    def run(self, ctx: TestBackendDispatchContext) -> tuple[bool, str | None]:
        return _dispatch_pt_uni(ctx)


@dataclass(frozen=True)
class NonPtNativeStrategy:
    def run(self, ctx: TestBackendDispatchContext) -> tuple[bool, str | None]:
        return _dispatch_non_pt(ctx)


_DISPATCH_STRATEGIES: dict[str, TestBackendStrategy | Any] = {
    "pt": PtStrategy(),
    "pt_uni": PtUniStrategy(),
    "onnx": NonPtNativeStrategy(),
    "engine": NonPtNativeStrategy(),
    "trt": NonPtNativeStrategy(),
}


def run_test_backend_via_registry(ctx: TestBackendDispatchContext) -> tuple[bool, str | None]:
    """
    Central strategy dispatcher keyed by model format.

    This keeps orchestration thin and makes adding new formats a local mapping
    change instead of ad-hoc branches across multiple functions.
    """
    fmt = str(ctx.fmt or "").strip().lower()
    handler = _DISPATCH_STRATEGIES.get(fmt)
    if handler is None:
        return False, f"Unsupported test backend format: {fmt!r}"
    # Keep backward-compatibility for tests/extensions that monkeypatch
    # registry entries with callables instead of strategy objects.
    if callable(handler):
        return handler(ctx)
    return handler.run(ctx)


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
    ctx = TestBackendDispatchContext(
        task_type=task_type,
        fmt="pt",
        target_kind=target_kind,
        root_dir=root_dir,
        primary_path=primary_path,
        artifact_path=primary_path,
        data_yaml=data_yaml,
        workspace_root=workspace_root,
        args=args,
        onnx_provider_policy="gpu_preferred",
    )
    return run_test_backend_via_registry(ctx)


def run_internal_pt_uni_backend(
    *,
    root_dir: str,
    primary_path: str,
    data_yaml: str,
    args: Any,
    onnx_provider_policy: str,
) -> tuple[bool, str | None]:
    ctx = TestBackendDispatchContext(
        task_type="detection",
        fmt="pt_uni",
        target_kind="runs",
        root_dir=root_dir,
        primary_path=primary_path,
        artifact_path=primary_path,
        data_yaml=data_yaml,
        workspace_root=root_dir,
        args=args,
        onnx_provider_policy=onnx_provider_policy,
    )
    return run_test_backend_via_registry(ctx)


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
    ctx = TestBackendDispatchContext(
        task_type=task_type,
        fmt=fmt,
        target_kind="runs",
        root_dir=root_dir,
        primary_path=artifact_path,
        artifact_path=artifact_path,
        data_yaml=data_yaml,
        workspace_root=root_dir,
        args=args,
        onnx_provider_policy=onnx_provider_policy,
    )
    return run_test_backend_via_registry(ctx)

