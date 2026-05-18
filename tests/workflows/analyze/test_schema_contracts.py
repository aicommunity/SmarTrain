from __future__ import annotations

import pytest

from smartrain.services.analyze.schema_contracts import (
    ANALYZE_SESSION_SCHEMA_VERSION,
    FORMAT_COMPARE_SCHEMA_VERSION,
    ensure_analyze_session_manifest,
    ensure_format_compare_index,
)


def test_ensure_analyze_session_manifest_sets_schema_fields() -> None:
    payload = {
        "session_name": "s1",
        "profile": "full",
        "baseline": "/tmp/r1",
        "others": ["/tmp/r2"],
        "artifacts": [],
        "tables": [],
        "images": [],
    }
    out = ensure_analyze_session_manifest(payload, session_type="analyze_all")
    assert out["schema_version"] == ANALYZE_SESSION_SCHEMA_VERSION
    assert out["schema_type"] == "analyze_all"


def test_ensure_analyze_session_manifest_rejects_unknown_schema_version() -> None:
    payload = {
        "session_name": "s1",
        "profile": "full",
        "baseline": "/tmp/r1",
        "others": [],
        "artifacts": [],
        "tables": [],
        "images": [],
        "schema_version": "9.9.9",
    }
    with pytest.raises(ValueError, match="unsupported schema_version"):
        ensure_analyze_session_manifest(payload, session_type="analyze_all")


def test_ensure_format_compare_index_requires_artifacts() -> None:
    with pytest.raises(ValueError, match="artifact reference"):
        ensure_format_compare_index({})


def test_ensure_format_compare_index_sets_schema_fields() -> None:
    out = ensure_format_compare_index({"test_csv": "artifacts/format_compare/test.csv"})
    assert out["schema_version"] == FORMAT_COMPARE_SCHEMA_VERSION
    assert out["schema_type"] == "format_compare"
