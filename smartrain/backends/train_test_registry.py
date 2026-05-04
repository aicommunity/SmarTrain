from __future__ import annotations

from smartrain.backends.contracts import BackendCapabilities
from smartrain.backends.registry import CapabilityRegistry
from smartrain.tasks.contracts import KNOWN_TASKS


_REGISTRY = CapabilityRegistry()
_REGISTRY.register(
    BackendCapabilities(
        backend="ultralytics",
        task_types=KNOWN_TASKS,
        model_formats=("pt",),
        can_train=True,
        can_test=True,
    )
)
_REGISTRY.register(
    BackendCapabilities(
        backend="onnxruntime",
        task_types=KNOWN_TASKS,
        model_formats=("onnx",),
        can_test=True,
    )
)
_REGISTRY.register(
    BackendCapabilities(
        backend="tensorrt",
        task_types=KNOWN_TASKS,
        model_formats=("engine", "trt"),
        can_test=True,
    )
)


def resolve_train_backend(*, task_type: str, model_format: str) -> BackendCapabilities:
    return _REGISTRY.resolve(task_type=task_type, model_format=model_format, require="train")


def resolve_test_backend(*, task_type: str, model_format: str) -> BackendCapabilities:
    return _REGISTRY.resolve(task_type=task_type, model_format=model_format, require="test")

