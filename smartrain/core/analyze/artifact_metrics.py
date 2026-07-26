"""Test-artifact metric readers for analysis workflows."""

from __future__ import annotations

import json
import os
from typing import Any

from smartrain.core.analyze.run_metrics_discovery import iter_test_formats
from smartrain.core.testing.artifact_paths import format_test_dir, load_test_artifacts_manifest


def read_test_performance_by_format_artifacts(
    run_dir: str, *, include_internal: bool = False
) -> dict[str, list[dict[str, Any]]]:
    def guess_target(format_name: str, stem: str) -> str:
        extension = {"pt": ".pt", "pt_uni": ".pt", "onnx": ".onnx", "engine": ".engine", "trt": ".trt"}.get(format_name, "")
        candidate = os.path.abspath(os.path.join(run_dir, "models", f"{stem}{extension}"))
        return candidate if extension and os.path.isfile(candidate) else stem

    output: dict[str, list[dict[str, Any]]] = {}
    formats = load_test_artifacts_manifest(run_dir).get("formats")
    if not isinstance(formats, dict):
        return output
    for format_name in iter_test_formats(include_internal):
        entry = formats.get(format_name)
        if not isinstance(entry, dict):
            continue
        records: list[dict[str, Any]] = []
        for item in entry.get("artifacts", []) if isinstance(entry.get("artifacts"), list) else []:
            if isinstance(item, dict) and isinstance(item.get("performance"), dict):
                target = item.get("target_path")
                records.append({
                    "target_path": os.path.abspath(os.path.join(run_dir, target)) if isinstance(target, str) and target else "",
                    "performance": item["performance"],
                })
        test_dir = format_test_dir(run_dir, format_name)
        if os.path.isdir(test_dir):
            for name in sorted(os.listdir(test_dir)):
                if not name.startswith("perf_") or not name.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(test_dir, name), encoding="utf-8") as file:
                        payload = json.load(file)
                except Exception:
                    continue
                if not isinstance(payload, dict) or not payload:
                    continue
                target = guess_target(format_name, name[len("perf_") : -len(".json")])
                if not any(record.get("target_path") == target for record in records):
                    records.append({"target_path": target, "performance": payload})
        if records:
            output[format_name] = records
    return output


def read_test_system_profile_by_format_artifacts(
    run_dir: str, *, include_internal: bool = False
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    formats = load_test_artifacts_manifest(run_dir).get("formats")
    if not isinstance(formats, dict):
        return output
    for format_name in iter_test_formats(include_internal):
        entry = formats.get(format_name)
        if not isinstance(entry, dict):
            continue
        records: list[dict[str, Any]] = []
        for item in entry.get("artifacts", []) if isinstance(entry.get("artifacts"), list) else []:
            if isinstance(item, dict) and isinstance(item.get("test_system_profile"), dict):
                target = item.get("target_path")
                records.append({
                    "target_path": os.path.abspath(os.path.join(run_dir, target)) if isinstance(target, str) and target else "",
                    "test_system_profile": item["test_system_profile"],
                })
        if records:
            output[format_name] = records
    return output
