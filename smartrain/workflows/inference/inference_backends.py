"""Backward-compatible re-exports; implementations live under backends."""

from smartrain.backends.implementations.ultralytics.inference import (
    BackendPrediction,
    ExternalProviderBackend,
    InferenceBackend,
    InferenceBackendRegistry,
    UltralyticsBackend,
)

__all__ = [
    "BackendPrediction",
    "ExternalProviderBackend",
    "InferenceBackend",
    "InferenceBackendRegistry",
    "UltralyticsBackend",
]
