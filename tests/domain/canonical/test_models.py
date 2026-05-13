from __future__ import annotations

from smartrain.domain.canonical.models import (
    CanonicalArtifactRef,
    CanonicalMetricsRef,
    CanonicalModelRef,
    CanonicalPayload,
    CanonicalPredictionRef,
    CanonicalRunRef,
)


def test_canonical_models_construct_minimal_payload() -> None:
    payload = CanonicalPayload(
        schema_version="2.0.0",
        generated_at="2026-05-04T00:00:00Z",
        producer="tests",
        artifacts=[
            CanonicalArtifactRef(
                id="a1",
                source_kind="run",
                created_at="2026-05-04T00:00:00Z",
                task_type="detection",
                backend_type="ultralytics",
            )
        ],
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
        runs=[
            CanonicalRunRef(
                run_id="r1",
                workspace="/tmp/ws",
                dataset_ref="ds",
                training_ref="tr",
            )
        ],
        metrics=[
            CanonicalMetricsRef(
                namespace="detection.main",
                primary_metrics={"map50": 0.5},
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
    assert payload.schema_version == "2.0.0"
    assert payload.models[0].model_id == "m1"

