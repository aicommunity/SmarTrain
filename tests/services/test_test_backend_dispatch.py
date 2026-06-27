from __future__ import annotations

from argparse import Namespace

import smartrain.services.test_backend_dispatch as dispatch


def test_dispatch_pt_runs_always_calls_ultralytics_backend(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_run_ultralytics_backend(**kwargs):
        calls.append("ultra")
        from datetime import datetime

        from smartrain.workflows.testing.model_test_backends import BackendRunResult

        return BackendRunResult(
            format="pt",
            backend="ultralytics",
            success=True,
            test_start_time=datetime.now(),
            test_end_time=datetime.now(),
            inference={},
            target_path=kwargs.get("weights_path"),
        )

    from smartrain.core.workflow_adapters import testing_runtime_api as mtr

    monkeypatch.setattr(mtr, "run_ultralytics_backend", _fake_run_ultralytics_backend)

    def _boom(*_a, **_k):
        raise AssertionError("complete_missing_test_artifacts should not be used for runs PT")

    monkeypatch.setattr(mtr, "complete_missing_test_artifacts", _boom)

    ok, err = dispatch.run_pt_test_backend(
        task_type="detection",
        target_kind="runs",
        root_dir="/tmp/run",
        primary_path="/tmp/run/model.pt",
        data_yaml="/tmp/data.yaml",
        workspace_root="/tmp/ws",
        args=Namespace(
            imgsz=640,
            conf=0.25,
            iou=0.7,
            batch=1,
            deep_diagnostics=False,
            perf=False,
            perf_warmup_images=0,
            device="cpu",
        ),
    )
    assert ok is True
    assert err is None
    assert calls == ["ultra"]


def test_registry_returns_error_for_unknown_format() -> None:
    ctx = dispatch.TestBackendDispatchContext(
        task_type="detection",
        fmt="abc",
        target_kind="runs",
        root_dir="/tmp/run",
        primary_path="/tmp/run/model.pt",
        artifact_path="/tmp/run/model.pt",
        data_yaml="/tmp/data.yaml",
        workspace_root="/tmp",
        args=Namespace(imgsz=640, conf=0.25, iou=0.7, batch=1, deep_diagnostics=False, perf=False, perf_warmup_images=0, device="cpu"),
        onnx_provider_policy="gpu_preferred",
    )
    ok, err = dispatch.run_test_backend_via_registry(ctx)
    assert ok is False
    assert "Unsupported test backend format" in str(err)


def test_segmentation_native_onnx_skipped(monkeypatch) -> None:
    persisted: list[dict] = []

    def _fake_persist(root_dir, **kwargs):
        persisted.append(kwargs)

    from smartrain.core.workflow_adapters import testing_runtime_api as mtr

    monkeypatch.setattr(mtr, "persist_target_test_artifacts_state", _fake_persist)
    monkeypatch.setattr(mtr, "run_native_format_backend", lambda **_k: (_ for _ in ()).throw(AssertionError("should not run")))

    ok, err = dispatch.run_non_pt_test_backend(
        task_type="segmentation",
        fmt="onnx",
        artifact_path="/tmp/run/model.onnx",
        root_dir="/tmp/run",
        data_yaml="/tmp/data.yaml",
        args=Namespace(
            imgsz=640,
            conf=0.25,
            iou=0.7,
            batch=1,
            deep_diagnostics=False,
            perf=False,
            perf_warmup_images=0,
            device="cpu",
            force_native_seg_test=False,
        ),
        onnx_provider_policy="gpu_preferred",
    )
    assert ok is True
    assert err is None
    assert len(persisted) == 1
    assert persisted[0]["status"] == "skipped"
    assert "capability_gap" in str(persisted[0]["error"])


def test_segmentation_native_onnx_force_runs(monkeypatch) -> None:
    called = {"n": 0}

    def _fake_run(**_kwargs):
        called["n"] += 1
        from datetime import datetime

        from smartrain.workflows.testing.model_test_backends import BackendRunResult

        return BackendRunResult(
            format="onnx",
            backend="onnxruntime",
            success=True,
            test_start_time=datetime.now(),
            test_end_time=datetime.now(),
            inference={},
            target_path="/tmp/run/model.onnx",
        )

    from smartrain.core.workflow_adapters import testing_runtime_api as mtr

    monkeypatch.setattr(mtr, "run_native_format_backend", _fake_run)
    monkeypatch.setattr(mtr, "check_onnx_format_preflight", lambda _p: (True, None))

    ok, err = dispatch.run_non_pt_test_backend(
        task_type="segmentation",
        fmt="onnx",
        artifact_path="/tmp/run/model.onnx",
        root_dir="/tmp/run",
        data_yaml="/tmp/data.yaml",
        args=Namespace(
            imgsz=640,
            conf=0.25,
            iou=0.7,
            batch=1,
            deep_diagnostics=False,
            perf=False,
            perf_warmup_images=0,
            device="cpu",
            force_native_seg_test=True,
        ),
        onnx_provider_policy="gpu_preferred",
    )
    assert ok is True
    assert called["n"] == 1


def test_wrapper_pt_uni_uses_registry(monkeypatch) -> None:
    called = {"n": 0}

    def _fake(ctx: dispatch.TestBackendDispatchContext):
        called["n"] += 1
        assert ctx.fmt == "pt_uni"
        assert ctx.task_type == "segmentation"
        return True, None

    monkeypatch.setitem(dispatch._DISPATCH_STRATEGIES, "pt_uni", _fake)
    ok, err = dispatch.run_internal_pt_uni_backend(
        root_dir="/tmp/run",
        primary_path="/tmp/run/model.pt",
        data_yaml="/tmp/data.yaml",
        args=Namespace(imgsz=640, conf=0.25, iou=0.7, batch=1, deep_diagnostics=False, perf=False, perf_warmup_images=0, device="cpu"),
        onnx_provider_policy="gpu_preferred",
        task_type="segmentation",
    )
    assert ok is True
    assert err is None
    assert called["n"] == 1

