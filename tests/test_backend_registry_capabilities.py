from __future__ import annotations

import pytest

from smartrain.backends.contracts import (
    BackendCapabilities,
    BackendExecutionResult,
    InferenceBackend as InferenceBackendProtocol,
    TrainBackend as TrainBackendProtocol,
)
from smartrain.backends.registry import CapabilityRegistry


def test_capability_registry_resolves_infer_backend() -> None:
    reg = CapabilityRegistry()
    reg.register(
        BackendCapabilities(
            backend="ultralytics",
            task_types=("detection",),
            model_formats=("pt", "onnx"),
            can_infer=True,
        )
    )
    caps = reg.resolve(task_type="detection", model_format="pt", require="infer")
    assert caps.backend == "ultralytics"


def test_capability_registry_raises_for_unsupported_combo() -> None:
    reg = CapabilityRegistry()
    reg.register(
        BackendCapabilities(
            backend="ultralytics",
            task_types=("detection",),
            model_formats=("pt",),
            can_infer=True,
        )
    )
    with pytest.raises(ValueError):
        reg.resolve(task_type="segmentation", model_format="pt", require="infer")


def test_capability_registry_rejects_unknown_require() -> None:
    reg = CapabilityRegistry()
    reg.register(
        BackendCapabilities(
            backend="ultralytics",
            task_types=("detection",),
            model_formats=("pt",),
            can_infer=True,
        )
    )
    with pytest.raises(ValueError):
        reg.resolve(task_type="detection", model_format="pt", require="export")


def test_capability_registry_resolve_backend_id_helper() -> None:
    reg = CapabilityRegistry()
    reg.register(
        BackendCapabilities(
            backend="onnxruntime",
            task_types=("detection",),
            model_formats=("onnx",),
            can_test=True,
        )
    )
    assert reg.resolve_backend_id(task_type="detection", model_format="onnx", require="test") == "onnxruntime"


def test_backend_protocol_shapes_accept_minimal_implementations() -> None:
    caps = BackendCapabilities(
        backend="stub",
        task_types=("detection",),
        model_formats=("pt",),
        can_train=True,
        can_test=True,
        can_infer=True,
    )

    class _Train:
        backend_id = "stub"
        capabilities = caps

        def train(self, *, request):
            return BackendExecutionResult(success=True, backend="stub", task_type="detection", model_format="pt")

    class _Test:
        backend_id = "stub"
        capabilities = caps

        def test(self, *, request):
            return BackendExecutionResult(success=True, backend="stub", task_type="detection", model_format="pt")

    class _Infer:
        backend_id = "stub"
        capabilities = caps

        def infer(self, *, request):
            return BackendExecutionResult(success=True, backend="stub", task_type="detection", model_format="pt")

    assert isinstance(_Train(), TrainBackendProtocol)
    assert isinstance(_Infer(), InferenceBackendProtocol)

