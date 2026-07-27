"""Schema contracts for analysis artifacts."""

from __future__ import annotations

from typing import Any

FORMAT_COMPARE_SCHEMA_VERSION = "1.0.0"


def ensure_format_compare_index(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("format compare index: payload must be an object")
    normalized = dict(payload)
    schema_version = str(normalized.get("schema_version") or "").strip()
    if schema_version and schema_version != FORMAT_COMPARE_SCHEMA_VERSION:
        raise ValueError(
            "format compare index: unsupported schema_version "
            + f"{schema_version!r}, expected {FORMAT_COMPARE_SCHEMA_VERSION!r}"
        )
    has_artifact_refs = any(
        bool(str(normalized.get(key) or "").strip())
        for key in ("csv", "test_csv", "val_csv", "pt_uni_csv", "eval_csv")
    )
    if not has_artifact_refs:
        raise ValueError("format compare index: expected at least one artifact reference key")
    normalized["schema_version"] = FORMAT_COMPARE_SCHEMA_VERSION
    normalized["schema_type"] = "format_compare"
    return normalized
