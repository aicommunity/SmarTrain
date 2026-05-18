from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExternalModelRef:
    provider_id: str | None
    model_ref: str
    raw_value: str
    is_external: bool


def parse_external_model_ref(value: Any) -> ExternalModelRef:
    """
    Parse provider-prefixed model references.

    Supported form: "<provider_id>:<model_ref>".
    Returns structured parse result.
    """
    raw = str(value or "").strip()
    if not raw or ":" not in raw:
        return ExternalModelRef(provider_id=None, model_ref=raw, raw_value=raw, is_external=False)
    provider, model_part = raw.split(":", 1)
    provider = provider.strip().lower()
    model_part = model_part.strip()
    if not provider or not model_part:
        return ExternalModelRef(provider_id=None, model_ref=raw, raw_value=raw, is_external=False)
    return ExternalModelRef(provider_id=provider, model_ref=model_part, raw_value=raw, is_external=True)


def validate_external_model_ref(
    ref: ExternalModelRef,
    *,
    known_provider_ids: set[str],
) -> ExternalModelRef:
    """Validate provider id for parsed external model reference."""
    if not ref.is_external or not ref.provider_id:
        return ref
    if ref.provider_id not in known_provider_ids:
        known = ", ".join(sorted(known_provider_ids))
        raise ValueError(f"Unknown external provider in model ref: {ref.provider_id!r}. Known providers: {known}")
    return ref
