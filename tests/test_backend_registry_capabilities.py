from __future__ import annotations

import pytest

from smartrain.backends.contracts import BackendCapabilities
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

