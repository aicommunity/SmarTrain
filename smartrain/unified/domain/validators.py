from __future__ import annotations

import re

from smartrain.backends.train_test_registry import resolve_test_backend
from smartrain.unified.domain.errors import (
    UnifiedCompatibilityError,
    UnifiedErrorDetails,
    UnifiedValidationError,
)
from smartrain.unified.domain.models import UnifiedPayload
from smartrain.tasks.contracts import KNOWN_TASKS

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
SUPPORTED_SCHEMA_MAJOR = 2


def _require_non_empty(value: str | None, *, entity: str, field: str) -> None:
    if isinstance(value, str) and value.strip():
        return
    raise UnifiedValidationError(
        UnifiedErrorDetails(
            error_code="required_field_missing",
            entity=entity,
            field=field,
            hint="Provide non-empty value.",
        ),
        f"{entity}.{field} is required",
    )


def validate_schema_version(version: str) -> None:
    _require_non_empty(version, entity="UnifiedPayload", field="schema_version")
    if not _SEMVER_RE.match(version.strip()):
        raise UnifiedValidationError(
            UnifiedErrorDetails(
                error_code="invalid_schema_version",
                entity="UnifiedPayload",
                field="schema_version",
                hint="Use semantic version format, e.g. 2.0.0.",
            ),
            f"Invalid schema_version: {version!r}",
        )
    major = int(version.split(".", 1)[0])
    if major > SUPPORTED_SCHEMA_MAJOR:
        raise UnifiedCompatibilityError(
            UnifiedErrorDetails(
                error_code="unsupported_schema_major",
                entity="UnifiedPayload",
                field="schema_version",
                hint=f"Use schema major <= {SUPPORTED_SCHEMA_MAJOR}.",
            ),
            f"Unsupported schema major version: {major}",
        )


def validate_unified_payload(payload: UnifiedPayload) -> None:
    validate_schema_version(payload.schema_version)
    _require_non_empty(payload.producer, entity="UnifiedPayload", field="producer")
    _require_non_empty(payload.generated_at, entity="UnifiedPayload", field="generated_at")

    for idx, model in enumerate(payload.models):
        entity = f"UnifiedModelRef[{idx}]"
        _require_non_empty(model.model_id, entity=entity, field="model_id")
        _require_non_empty(model.weights_path, entity=entity, field="weights_path")
        if model.task_type not in KNOWN_TASKS:
            raise UnifiedValidationError(
                UnifiedErrorDetails(
                    error_code="unsupported_task_type",
                    entity=entity,
                    field="task_type",
                    hint=f"Use one of: {', '.join(KNOWN_TASKS)}.",
                ),
                f"Unsupported task_type: {model.task_type!r}",
            )
        if not isinstance(model.provenance, dict) or not model.provenance:
            raise UnifiedValidationError(
                UnifiedErrorDetails(
                    error_code="missing_provenance",
                    entity=entity,
                    field="provenance",
                    hint="Add at least one provenance key/value.",
                ),
                "Model provenance is required.",
            )
        try:
            resolved = resolve_test_backend(task_type=model.task_type, model_format=model.format).backend
        except Exception as exc:
            raise UnifiedCompatibilityError(
                UnifiedErrorDetails(
                    error_code="backend_format_incompatible",
                    entity=entity,
                    field="format",
                    hint="Check task_type/model_format capability matrix.",
                ),
                str(exc),
            ) from exc
        if resolved != model.backend_type:
            raise UnifiedCompatibilityError(
                UnifiedErrorDetails(
                    error_code="backend_type_mismatch",
                    entity=entity,
                    field="backend_type",
                    hint=f"Expected backend_type={resolved!r} for format={model.format!r}.",
                ),
                f"Incompatible backend_type={model.backend_type!r} for format={model.format!r}.",
            )

    for idx, metric in enumerate(payload.metrics):
        entity = f"UnifiedMetricsRef[{idx}]"
        _require_non_empty(metric.raw_path, entity=entity, field="raw_path")
        _require_non_empty(metric.producer, entity=entity, field="producer")
        ns = metric.namespace.strip().lower()
        task = metric.task_type
        if task not in KNOWN_TASKS:
            raise UnifiedValidationError(
                UnifiedErrorDetails(
                    error_code="unsupported_task_type",
                    entity=entity,
                    field="task_type",
                ),
                f"Unsupported task_type: {task!r}",
            )
        if not ns.startswith(task):
            raise UnifiedValidationError(
                UnifiedErrorDetails(
                    error_code="metrics_namespace_mismatch",
                    entity=entity,
                    field="namespace",
                    hint=f"Namespace should start with task prefix {task!r}.",
                ),
                f"Namespace {metric.namespace!r} does not match task_type={task!r}.",
            )

    for idx, pred in enumerate(payload.predictions):
        entity = f"UnifiedPredictionRef[{idx}]"
        _require_non_empty(pred.items_path, entity=entity, field="items_path")
        _require_non_empty(pred.producer, entity=entity, field="producer")
        validate_schema_version(pred.schema_version)

