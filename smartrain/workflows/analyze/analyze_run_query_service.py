from __future__ import annotations

import argparse
import os
from typing import Any

from smartrain.core.runtime.run_discovery import find_run_directories
from smartrain.workflows.analyze.analyze_models import RunRecord


def matches_optional_bool(value: bool | None, expected: bool | None) -> bool:
    if expected is None:
        return True
    return value is expected


def build_run_record_canonical(
    run_dir: str,
    *,
    read_test_metrics_for_run_cb: Any,
) -> RunRecord:
    from smartrain.orchestrators.canonical_gateway import load_target

    payload = load_target(run_dir, source_kind="run")
    model_name: str | None = None
    dataset_name: str | None = None
    if payload.models:
        model_name = str(payload.models[0].model_id or "").strip() or None
    if payload.runs:
        dataset_name = str(payload.runs[0].dataset_ref or "").strip() or None
    metrics = read_test_metrics_for_run_cb(run_dir)
    return RunRecord(
        run_dir=run_dir,
        model=model_name,
        dataset_name=dataset_name,
        training_ok=None,
        testing_ok=None,
        training_duration_s=None,
        test_metrics=metrics,
        train_last_metrics={},
    )


def read_test_metrics_for_run(run_dir: str, *, format_name: str = "pt") -> dict[str, Any]:
    from smartrain.orchestrators.canonical_gateway import load_metrics

    metric_refs = load_metrics(run_dir, source_kind="run", format_name=format_name)
    if metric_refs:
        out = dict(metric_refs[0].primary_metrics or {})
        out.update(dict(metric_refs[0].secondary_metrics or {}))
        return out
    return {}


def flat_row_canonical(run_dir: str, *, build_run_record_cb: Any) -> dict[str, Any]:
    rec = build_run_record_cb(run_dir)
    return {
        "run_dir": run_dir,
        "run_name": os.path.basename(run_dir.rstrip(os.sep)),
        "model": rec.model,
        "dataset_name": rec.dataset_name,
    }


def filtered_run_records(
    args: argparse.Namespace,
    *,
    build_run_record_cb: Any,
) -> list[tuple[str, Any]]:
    runs = find_run_directories(args.models_root)
    recs: list[tuple[str, Any]] = []
    filter_dataset = getattr(args, "filter_dataset", None)
    filter_model = getattr(args, "filter_model", None)
    filter_training_ok = getattr(args, "filter_training_ok", None)
    filter_testing_ok = getattr(args, "filter_testing_ok", None)
    for run_dir in runs:
        try:
            rec = build_run_record_cb(run_dir)
        except Exception as e:
            print(f"[WARN] {run_dir}: failed to index run ({e})")
            continue
        if filter_dataset and rec.dataset_name != filter_dataset:
            continue
        if filter_model and rec.model != filter_model:
            continue
        if not matches_optional_bool(rec.training_ok, filter_training_ok):
            continue
        if not matches_optional_bool(rec.testing_ok, filter_testing_ok):
            continue
        recs.append((run_dir, rec))
    return recs
