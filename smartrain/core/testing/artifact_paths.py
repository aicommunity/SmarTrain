"""Run test artifact path resolution (canonical layout, no workflow imports)."""

from __future__ import annotations

import json
import os
from typing import Any

from smartrain.core.runtime.run_artifacts import run_test_backend_dir, run_test_format_dir, run_tests_dir

TEST_ARTIFACTS_MANIFEST = "test_artifacts_manifest.json"
PUBLIC_TEST_FORMATS = ("pt", "onnx", "engine", "trt")
INTERNAL_TEST_FORMATS = ("pt_uni",)
SUPPORTED_TEST_FORMATS = PUBLIC_TEST_FORMATS
ALL_TEST_FORMATS = PUBLIC_TEST_FORMATS + INTERNAL_TEST_FORMATS


def normalize_format_name(format_name: str | None) -> str:
    raw = str(format_name or "pt").strip().lower()
    if raw in {"", "best", "base"}:
        return "pt"
    if raw in {"tensorrt-engine"}:
        return "engine"
    if raw in {"tensorrt-trt"}:
        return "trt"
    if raw not in ALL_TEST_FORMATS:
        raise ValueError(f"Unsupported test format: {format_name}")
    return raw


def format_suffix(format_name: str | None) -> str:
    fmt = normalize_format_name(format_name)
    return "" if fmt == "pt" else f"_{fmt}"


def format_test_dir(root_dir: str, format_name: str | None = "pt") -> str:
    return _format_test_dir(root_dir, format_name, prefer_legacy_for_read=True)


def format_test_dir_for_write(root_dir: str, format_name: str | None = "pt") -> str:
    return _format_test_dir(root_dir, format_name, prefer_legacy_for_read=False)


def _format_test_dir(root_dir: str, format_name: str | None, *, prefer_legacy_for_read: bool) -> str:
    fmt = normalize_format_name(format_name)
    preferred = run_test_backend_dir(root_dir, "ultralytics") if fmt == "pt" else run_test_format_dir(root_dir, fmt)
    legacy = os.path.join(root_dir, f"test{format_suffix(fmt)}")
    if prefer_legacy_for_read and os.path.isdir(legacy) and not preferred.exists():
        return legacy
    return str(preferred)


def format_metrics_path(root_dir: str, format_name: str | None = "pt") -> str:
    return _format_metrics_path(root_dir, format_name, prefer_legacy_for_read=True)


def format_metrics_path_for_write(root_dir: str, format_name: str | None = "pt") -> str:
    return _format_metrics_path(root_dir, format_name, prefer_legacy_for_read=False)


def _format_metrics_path(root_dir: str, format_name: str | None, *, prefer_legacy_for_read: bool) -> str:
    fmt = normalize_format_name(format_name)
    preferred = run_tests_dir(root_dir) / f"test_metrics{format_suffix(fmt)}.csv"
    legacy = os.path.join(root_dir, f"test_metrics{format_suffix(fmt)}.csv")
    if prefer_legacy_for_read and os.path.isfile(legacy) and not preferred.is_file():
        return legacy
    return str(preferred)


def format_metrics_path_for_split(root_dir: str, split: str, format_name: str | None = "pt") -> str:
    return _format_metrics_path_for_split(root_dir, split, format_name, prefer_legacy_for_read=True)


def format_metrics_path_for_split_write(root_dir: str, split: str, format_name: str | None = "pt") -> str:
    return _format_metrics_path_for_split(root_dir, split, format_name, prefer_legacy_for_read=False)


def _format_metrics_path_for_split(
    root_dir: str,
    split: str,
    format_name: str | None,
    *,
    prefer_legacy_for_read: bool,
) -> str:
    split_name = str(split).strip().lower()
    if split_name == "test":
        return _format_metrics_path(root_dir, format_name, prefer_legacy_for_read=prefer_legacy_for_read)
    if split_name == "val":
        fmt = normalize_format_name(format_name)
        preferred = run_tests_dir(root_dir) / f"val_metrics{format_suffix(fmt)}.csv"
        legacy = os.path.join(root_dir, f"val_metrics{format_suffix(fmt)}.csv")
        if prefer_legacy_for_read and os.path.isfile(legacy) and not preferred.is_file():
            return legacy
        return str(preferred)
    raise ValueError(f"Unsupported split: {split}")


def format_recommendation_path(root_dir: str, split: str, format_name: str | None = "pt") -> str:
    return _format_recommendation_path(root_dir, split, format_name, prefer_legacy_for_read=True)


def format_recommendation_path_for_write(root_dir: str, split: str, format_name: str | None = "pt") -> str:
    return _format_recommendation_path(root_dir, split, format_name, prefer_legacy_for_read=False)


def _format_recommendation_path(
    root_dir: str,
    split: str,
    format_name: str | None,
    *,
    prefer_legacy_for_read: bool,
) -> str:
    split_name = str(split).strip().lower()
    if split_name not in {"test", "val"}:
        raise ValueError(f"Unsupported split: {split}")
    fmt = normalize_format_name(format_name)
    preferred = run_tests_dir(root_dir) / f"confidence_recommendations_{split_name}{format_suffix(fmt)}.json"
    legacy = os.path.join(root_dir, f"confidence_recommendations_{split_name}{format_suffix(fmt)}.json")
    if prefer_legacy_for_read and os.path.isfile(legacy) and not preferred.is_file():
        return legacy
    return str(preferred)


def test_artifacts_manifest_path(root_dir: str) -> str:
    return _test_artifacts_manifest_path(root_dir, prefer_legacy_for_read=True)


def artifacts_manifest_path_for_write(root_dir: str) -> str:
    return _test_artifacts_manifest_path(root_dir, prefer_legacy_for_read=False)


def _test_artifacts_manifest_path(root_dir: str, *, prefer_legacy_for_read: bool) -> str:
    preferred = run_tests_dir(root_dir) / TEST_ARTIFACTS_MANIFEST
    legacy = os.path.join(root_dir, TEST_ARTIFACTS_MANIFEST)
    if prefer_legacy_for_read and os.path.isfile(legacy) and not preferred.is_file():
        return legacy
    return str(preferred)


def load_test_artifacts_manifest(root_dir: str) -> dict[str, Any]:
    path = test_artifacts_manifest_path(root_dir)
    if not os.path.isfile(path):
        return {"formats": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {"formats": {}}
    except Exception as e:
        return {
            "formats": {},
            "_diagnostics": {
                "status": "manifest_load_failed",
                "reason_code": "manifest_read_error",
                "reason_detail": str(e),
            },
        }
