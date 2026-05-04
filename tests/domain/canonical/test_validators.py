from __future__ import annotations

from dataclasses import replace

import pytest

from smartrain.domain.canonical.errors import CanonicalCompatibilityError, CanonicalValidationError
from smartrain.domain.canonical.models import (
    CanonicalMetricsRef,
    CanonicalModelRef,
    CanonicalPayload,
    CanonicalPredictionRef,
)
from smartrain.domain.canonical.validators import validate_payload


def _base_payload() -> CanonicalPayload:
    return CanonicalPayload(
        schema_version="2.0.0",
        generated_at="2026-05-04T00:00:00Z",
        producer="tests",
        models=[
            CanonicalModelRef(
                model_id="m1",
                format="pt",
                weights_path="/tmp/model.pt",
                config_path=None,
                labels_path=None,
                provenance={"source": "unit"},
                task_type="detection",
                backend_type="ultralytics",
            )
        ],
        metrics=[
            CanonicalMetricsRef(
                namespace="detection.main",
                primary_metrics={},
                secondary_metrics={},
                raw_path="/tmp/metrics.json",
                producer="ultralytics",
                task_type="detection",
            )
        ],
        predictions=[
            CanonicalPredictionRef(
                task_type="detection",
                items_path="/tmp/preds.jsonl",
                schema_version="2.0.0",
                producer="ultralytics",
                count=1,
            )
        ],
    )


def test_validate_payload_ok() -> None:
    validate_payload(_base_payload())


def test_validate_payload_rejects_empty_weights_path() -> None:
    payload = _base_payload()
    payload.models[0] = replace(payload.models[0], weights_path="")
    with pytest.raises(CanonicalValidationError):
        validate_payload(payload)


def test_validate_payload_rejects_backend_format_mismatch() -> None:
    payload = _base_payload()
    payload.models[0] = replace(payload.models[0], backend_type="onnxruntime")
    with pytest.raises(CanonicalCompatibilityError):
        validate_payload(payload)


def test_validate_payload_rejects_metrics_namespace_mismatch() -> None:
    payload = _base_payload()
    payload.metrics[0] = replace(payload.metrics[0], namespace="segmentation.main")
    with pytest.raises(CanonicalValidationError):
        validate_payload(payload)

