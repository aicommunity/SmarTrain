"""Discover per-format metrics CSV paths on disk for run/model roots."""

from __future__ import annotations

import os
from typing import Any

from smartrain.core.testing.artifact_paths import (
    INTERNAL_TEST_FORMATS,
    SUPPORTED_TEST_FORMATS,
    format_metrics_path,
    format_metrics_path_for_split,
    load_test_artifacts_manifest,
)

__all__ = [
    "METRIC_AGG_COLUMNS",
    "read_metrics_by_format_for_split",
    "read_metrics_by_format_for_split_artifacts",
    "read_test_metrics_by_format",
    "read_test_metrics_by_format_artifacts",
]

METRIC_AGG_COLUMNS = ("mAP50-95", "mAP50", "Box-F1", "Box-P", "Box-R")


def iter_test_formats(include_internal: bool = False) -> tuple[str, ...]:
    return SUPPORTED_TEST_FORMATS + INTERNAL_TEST_FORMATS if include_internal else SUPPORTED_TEST_FORMATS


_iter_test_formats = iter_test_formats


def _resolve_manifest_metrics_path(run_dir: str, rel_path: str) -> str | None:
    rel = str(rel_path or "").strip()
    if not rel:
        return None
    candidate = os.path.abspath(os.path.join(run_dir, rel))
    if os.path.isfile(candidate):
        return candidate
    basename = os.path.basename(rel)
    if basename:
        tests_candidate = os.path.join(run_dir, "tests", basename)
        if os.path.isfile(tests_candidate):
            return os.path.abspath(tests_candidate)
    return None


def latest_test_metrics_path(run_dir: str, format_name: str | None = "pt") -> str | None:
    if format_name not in (None, "", "pt"):
        p = format_metrics_path(run_dir, format_name)
        return p if os.path.isfile(p) else None
    canonical_pt_metrics = format_metrics_path(run_dir, "pt")
    if os.path.isfile(canonical_pt_metrics):
        return canonical_pt_metrics
    legacy_pt_metrics = os.path.join(run_dir, "test_metrics.csv")
    return legacy_pt_metrics if os.path.isfile(legacy_pt_metrics) else None


def read_test_metrics_by_format(run_dir: str, *, include_internal: bool = False) -> dict[str, str]:
    out: dict[str, str] = {}
    manifest = load_test_artifacts_manifest(run_dir)
    formats = manifest.get("formats")
    if isinstance(formats, dict):
        for fmt in _iter_test_formats(include_internal):
            entry = formats.get(fmt)
            if not isinstance(entry, dict):
                continue
            rel = entry.get("metrics_csv")
            selected: str | None = None
            if isinstance(rel, str) and rel.strip():
                p = _resolve_manifest_metrics_path(run_dir, rel)
                if p:
                    selected = p
            if selected is None:
                artifacts = entry.get("artifacts")
                if isinstance(artifacts, list):
                    for item in artifacts:
                        if not isinstance(item, dict):
                            continue
                        rel_item = item.get("metrics_csv")
                        if not isinstance(rel_item, str) or not rel_item.strip():
                            continue
                        p = _resolve_manifest_metrics_path(run_dir, rel_item)
                        if p:
                            selected = p
                            break
            if selected is not None:
                out[fmt] = selected
    for fmt in _iter_test_formats(include_internal):
        p = latest_test_metrics_path(run_dir, fmt)
        if p and os.path.isfile(p):
            out.setdefault(fmt, p)
    return out


def read_test_metrics_by_format_artifacts(
    run_dir: str,
    *,
    include_internal: bool = False,
) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    manifest = load_test_artifacts_manifest(run_dir)
    formats = manifest.get("formats")
    if not isinstance(formats, dict):
        return out
    for fmt in _iter_test_formats(include_internal):
        entry = formats.get(fmt)
        if not isinstance(entry, dict):
            continue
        records: list[dict[str, str]] = []
        artifacts = entry.get("artifacts")
        if isinstance(artifacts, list):
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                rel_metrics = item.get("metrics_csv")
                if not isinstance(rel_metrics, str) or not rel_metrics.strip():
                    continue
                metrics_path = _resolve_manifest_metrics_path(run_dir, rel_metrics)
                if not metrics_path:
                    continue
                rel_target = item.get("target_path")
                target_path = os.path.abspath(os.path.join(run_dir, rel_target)) if isinstance(rel_target, str) and rel_target else ""
                records.append({"metrics_path": metrics_path, "target_path": target_path})
        if not records:
            rel = entry.get("metrics_csv")
            if isinstance(rel, str) and rel.strip():
                metrics_path = _resolve_manifest_metrics_path(run_dir, rel)
                if metrics_path:
                    rel_target = entry.get("target_path")
                    target_path = (
                        os.path.abspath(os.path.join(run_dir, rel_target))
                        if isinstance(rel_target, str) and rel_target
                        else ""
                    )
                    records.append({"metrics_path": metrics_path, "target_path": target_path})
        if records:
            out[fmt] = records
    return out


def read_metrics_by_format_for_split(
    run_dir: str,
    split: str,
    *,
    include_internal: bool = False,
) -> dict[str, str]:
    split_name = str(split).strip().lower()
    if split_name == "test":
        return read_test_metrics_by_format(run_dir, include_internal=include_internal)
    out: dict[str, str] = {}
    for fmt in _iter_test_formats(include_internal):
        p = format_metrics_path_for_split(run_dir, split_name, fmt)
        if os.path.isfile(p):
            out[fmt] = p
    return out


def read_metrics_by_format_for_split_artifacts(
    run_dir: str,
    split: str,
    *,
    include_internal: bool = False,
) -> dict[str, list[dict[str, str]]]:
    split_name = str(split).strip().lower()
    if split_name == "test":
        return read_test_metrics_by_format_artifacts(run_dir, include_internal=include_internal)
    out: dict[str, list[dict[str, str]]] = {}
    for fmt in _iter_test_formats(include_internal):
        p = format_metrics_path_for_split(run_dir, split_name, fmt)
        if os.path.isfile(p):
            out[fmt] = [{"metrics_path": p, "target_path": ""}]
    return out
