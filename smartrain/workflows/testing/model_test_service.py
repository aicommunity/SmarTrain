"""Workflow facade: model-test artifact helpers."""

from __future__ import annotations

from smartrain.services.testing import model_test_service as _impl


def __getattr__(name: str):
    return getattr(_impl, name)
