from __future__ import annotations

from smartrain.backends.train_test_registry import resolve_test_backend
from smartrain.run_model_contract.domain.errors import UnifiedCompatibilityError, UnifiedErrorDetails
from smartrain.run_model_contract.domain.models import UnifiedPayload


def validate_unified_model_backends(payload: UnifiedPayload) -> None:
    """Check task_type/model_format/backend_type against capability registry (not domain-pure)."""
    for idx, model in enumerate(payload.models):
        entity = f"UnifiedModelRef[{idx}]"
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
