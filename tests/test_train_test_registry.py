from __future__ import annotations

import pytest

from smartrain.backends.train_test_registry import (
    resolve_infer_backend,
    resolve_test_backend,
    resolve_train_backend,
)


def test_registry_resolves_train_backend_for_pt() -> None:
    caps = resolve_train_backend(task_type="detection", model_format="pt")
    assert caps.backend == "ultralytics"
    assert caps.can_train is True


def test_registry_resolves_test_and_infer_backends_by_format() -> None:
    test_caps = resolve_test_backend(task_type="detection", model_format="onnx")
    infer_caps = resolve_infer_backend(task_type="detection", model_format="onnx")
    assert test_caps.backend == "onnxruntime"
    assert infer_caps.backend == "onnxruntime"
    assert infer_caps.can_infer is True


def test_registry_raises_for_unsupported_train_format() -> None:
    with pytest.raises(ValueError):
        resolve_train_backend(task_type="detection", model_format="onnx")
