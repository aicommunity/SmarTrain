from __future__ import annotations

from argparse import Namespace

import smartrain.services.test_backend_dispatch as dispatch


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


def test_wrapper_pt_uni_uses_registry(monkeypatch) -> None:
    called = {"n": 0}

    def _fake(ctx: dispatch.TestBackendDispatchContext):
        called["n"] += 1
        assert ctx.fmt == "pt_uni"
        return True, None

    monkeypatch.setitem(dispatch._DISPATCH_STRATEGIES, "pt_uni", _fake)
    ok, err = dispatch.run_internal_pt_uni_backend(
        root_dir="/tmp/run",
        primary_path="/tmp/run/model.pt",
        data_yaml="/tmp/data.yaml",
        args=Namespace(imgsz=640, conf=0.25, iou=0.7, batch=1, deep_diagnostics=False, perf=False, perf_warmup_images=0, device="cpu"),
        onnx_provider_policy="gpu_preferred",
    )
    assert ok is True
    assert err is None
    assert called["n"] == 1

