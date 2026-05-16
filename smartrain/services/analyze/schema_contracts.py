from __future__ import annotations

from typing import Any, Literal

ANALYZE_SESSION_SCHEMA_VERSION = "1.0.0"
FORMAT_COMPARE_SCHEMA_VERSION = "1.0.0"


def _require_keys(payload: dict[str, Any], keys: tuple[str, ...], *, context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"{context}: missing required fields: {', '.join(missing)}")


def ensure_analyze_session_manifest(
    payload: dict[str, Any],
    *,
    session_type: Literal["analyze_all", "compare"],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("analyze manifest: payload must be an object")
    normalized = dict(payload)
    _require_keys(normalized, ("session_name", "artifacts"), context="analyze manifest")
    if not isinstance(normalized.get("artifacts"), list):
        raise ValueError("analyze manifest: 'artifacts' must be a list")
    if session_type == "analyze_all":
        _require_keys(
            normalized,
            ("profile", "baseline", "others", "tables", "images"),
            context="analyze manifest (analyze_all)",
        )
    if session_type == "compare":
        _require_keys(normalized, ("baseline", "others"), context="analyze manifest (compare)")
    schema_version = str(normalized.get("schema_version") or "").strip()
    if schema_version and schema_version != ANALYZE_SESSION_SCHEMA_VERSION:
        raise ValueError(
            "analyze manifest: unsupported schema_version "
            + f"{schema_version!r}, expected {ANALYZE_SESSION_SCHEMA_VERSION!r}"
        )
    normalized["schema_version"] = ANALYZE_SESSION_SCHEMA_VERSION
    normalized["schema_type"] = session_type
    return normalized


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
