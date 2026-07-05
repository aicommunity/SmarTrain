#!/usr/bin/env python3
"""
Training runs scan, summary CSV export, and metric comparisons (CSV + PNG).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from io import StringIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from smartrain.services.analyze.compare import (
    build_delta_rows,
    compute_composite_score,
    generate_compare_insights,
)
from smartrain.services.analyze.report_writer import write_analysis_report, write_manifest
from smartrain.core.runtime.run_artifacts import (
    preferred_run_model_path,
    materialize_preferred_run_model,
    run_test_backend_dir,
)
from smartrain.services.analyze.cache import (
    append_cache_entry,
    compute_fingerprint,
    data_yaml_hash,
    run_cache_root,
    weights_hash,
)
from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.services.analyze.prompts import prompt_choice, prompt_int, prompt_text
from smartrain.services.analyze.metrics_reader import (
    DEFAULT_MAP_COL,
    latest_test_metrics_path,
    pick_map_column,
    load_metadata,
    results_csv_path,
    read_test_metrics_by_format,
    read_test_performance_by_format_artifacts,
    read_test_system_profile_by_format_artifacts,
)
from smartrain.core.runtime.run_discovery import (
    discover_analysis_targets,
    find_run_directories,
    is_run_directory,
    resolve_models_scan_root,
)
from smartrain.core.runtime.ultralytics_ephemeral import best_effort_prune_workspace_runs_detect, ultralytics_sidecar_dir
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.core.training.confidence_recommendation import recommendation_file_path, read_recommendation_file
from smartrain.services.analyze.models import RunRecord
from smartrain.services.analyze.compare_session import (
    resolve_default_relative_output,
    resolve_session_artifacts_dir,
    resolve_session_name,
    resolve_session_root,
)
from smartrain.services.analyze.benchmark import (
    collect_split_images as _svc_collect_split_images,
    resolve_inference_csv_path as _svc_resolve_inference_csv_path,
    resolve_inference_plot_png as _svc_resolve_inference_plot_png,
    resolve_selected_run_dirs as _svc_resolve_selected_run_dirs,
    run_inference_benchmark as _svc_run_inference_benchmark,
    run_inference_plot as _svc_run_inference_plot,
)
from smartrain.core.inference.ultralytics_metrics_pr import (
    extract_pr_curve_from_ultralytics_metrics,
    extract_pr_curve_per_class_from_ultralytics_metrics,
)
from smartrain.services.analyze.pr_curves import (
    resolve_pr_output_png as _svc_resolve_pr_output_png,
    run_pr_curves as _svc_run_pr_curves,
)
from smartrain.services.analyze.test_metrics_plot import (
    resolve_test_metrics_plot_png as _svc_resolve_test_metrics_plot_png,
    run_test_metrics_plot as _svc_run_test_metrics_plot,
)
from smartrain.services.analyze.all_selection import (
    prepare_all_selection as _svc_prepare_all_selection,
)
from smartrain.services.analyze.all_data_yaml import (
    resolve_all_data_yaml_context as _svc_resolve_all_data_yaml_context,
)
from smartrain.services.analyze.all_baseline_artifacts import (
    run_all_baseline_artifacts as _svc_run_all_baseline_artifacts,
)
from smartrain.services.analyze.all_quality_stage import (
    run_all_quality_stage as _svc_run_all_quality_stage,
)
from smartrain.services.analyze.all_speed_stage import (
    run_all_speed_stage as _svc_run_all_speed_stage,
)
from smartrain.services.analyze.all_pr_stage import (
    run_all_pr_stage as _svc_run_all_pr_stage,
)
from smartrain.services.analyze.all_finalize import (
    finalize_all_session as _svc_finalize_all_session,
)
from smartrain.services.analyze.all_command import (
    run_all_command as _svc_run_all_command,
)

# Subparser for `analyze all` — used to rebuild replay via cli_replay.build_non_interactive_command.
_ANALYZE_ALL_SUBPARSER: argparse.ArgumentParser | None = None


def _finalize_all_session_with_replay(**kwargs: Any) -> None:
    kwargs["replay_parser"] = _ANALYZE_ALL_SUBPARSER
    _svc_finalize_all_session(**kwargs)
from smartrain.services.analyze.run_query import (
    build_run_record_unified as _svc_build_run_record_unified,
    filtered_run_records as _svc_filtered_run_records,
    flat_row_unified as _svc_flat_row_unified,
    matches_optional_bool as _svc_matches_optional_bool,
    read_test_metrics_for_run as _svc_read_test_metrics_for_run,
)
from smartrain.services.analyze.recompute_cache import (
    collect_missing_metrics_recompute_plan as _svc_collect_missing_metrics_recompute_plan,
    load_recompute_status as _svc_load_recompute_status,
    recompute_run_test_metrics as _svc_recompute_run_test_metrics,
    recompute_status_fingerprint as _svc_recompute_status_fingerprint,
    recompute_status_path as _svc_recompute_status_path,
    runs_with_missing_metrics as _svc_runs_with_missing_metrics,
    save_recompute_status as _svc_save_recompute_status,
)
from smartrain.services.analyze.system_profile_compare import (
    write_system_profile_compare_csv as _svc_write_system_profile_compare_csv,
    write_test_system_profile_compare_csv as _svc_write_test_system_profile_compare_csv,
)
from smartrain.services.analyze.ultralytics_test_artifacts import (
    collect_ultralytics_test_artifacts as _svc_collect_ultralytics_test_artifacts,
)
from smartrain.services.analyze.compare_finalize import (
    finalize_compare_analytics_session as _svc_finalize_compare_analytics_session,
    resolve_compare_artifact_path as _svc_resolve_compare_artifact_path,
    resolve_compare_png_path as _svc_resolve_compare_png_path,
)
from smartrain.services.analyze.data_yaml import collect_data_yaml_candidates_for_run
from smartrain.services.analyze.table import export_runs_table, run_scan_command
from smartrain.services.analyze.compare_workflow import run_compare_workflow
from smartrain.services.analyze.artifact_builders import (
    collect_confidence_recommendation_tables,
    write_speed_quality_artifacts,
)
from smartrain.services.analyze.format_compare import write_format_compare_artifacts
from smartrain.services.analyze.interactive import run_interactive_workflow
from smartrain.services.analyze.leaderboard import (
    build_leaderboard_records,
    write_leaderboard_csv,
)
from smartrain.services.analyze.commands.registry import AnalyzeCommandRegistry
from smartrain.tasks.metric_columns import metric_agg_columns

METRIC_AGG_COLUMNS = metric_agg_columns("detection")

_ANALYZE_COMMAND_REGISTRY = AnalyzeCommandRegistry()


def _unified_read_enabled() -> bool:
    return True


def _clear_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()


def _run_val_memory_safe(
    model: Any,
    *,
    data_yaml: str,
    split: str,
    val_batch: int,
    val_imgsz: int,
    val_half: bool,
    gpu_only: bool,
    ultra_project: str,
    ultra_name: str = "val-analyze",
) -> Any:
    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if cuda_available:
        attempts = [
            {"device": "0", "workers": 0, "batch": max(1, int(val_batch)), "imgsz": max(320, int(val_imgsz)), "half": bool(val_half)},
            {"device": "0", "workers": 0, "batch": 1, "imgsz": max(320, int(val_imgsz)), "half": True},
            {"device": "0", "workers": 0, "batch": 1, "imgsz": min(max(320, int(val_imgsz)), 640), "half": True},
            {"device": "0", "workers": 0, "batch": 1, "imgsz": 512, "half": True},
        ]
        last_exc: Exception | None = None
        for idx, extra in enumerate(attempts, start=1):
            _clear_gpu_memory()
            try:
                return model.val(
                    data=data_yaml,
                    split=split,
                    plots=False,
                    save=False,
                    verbose=False,
                    project=ultra_project,
                    name=ultra_name,
                    exist_ok=True,
                    **extra,
                )
            except Exception as e:
                last_exc = e
                if _is_workers_pickle_error(e):
                    continue
                if _is_cuda_oom_error(e):
                    print(f"[WARN] CUDA OOM in val() attempt {idx}, retrying with lower GPU memory profile ...")
                    continue
                raise
        if gpu_only and last_exc is not None:
            raise last_exc

    return model.val(
        data=data_yaml,
        split=split,
        plots=False,
        save=False,
        verbose=False,
        project=ultra_project,
        name=ultra_name,
        exist_ok=True,
        workers=0,
        device="cpu",
    )


def _resolve_run_val_profile(
    run_dir: str,
    *,
    default_batch: int = 1,
    default_imgsz: int = 640,
    default_half: bool = True,
) -> tuple[int, int, bool]:
    args_yaml = os.path.join(run_dir, "train-ultralytics", "args.yaml")
    if not os.path.isfile(args_yaml):
        args_yaml = os.path.join(run_dir, "train", "args.yaml")
    batch = int(default_batch)
    imgsz = int(default_imgsz)
    half = bool(default_half)
    if not os.path.isfile(args_yaml):
        return batch, imgsz, half
    try:
        with open(args_yaml, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        if isinstance(payload, dict):
            b = payload.get("batch")
            if b is not None:
                b_num = int(float(b))
                if b_num > 0:
                    batch = b_num
            im = payload.get("imgsz")
            if isinstance(im, (list, tuple)) and im:
                im = im[0]
            if im is not None:
                im_num = int(float(im))
                if im_num > 0:
                    imgsz = im_num
            h = payload.get("half")
            if isinstance(h, bool):
                half = h
    except Exception:
        pass
    return max(1, batch), max(320, imgsz), bool(half)


def _session_name(raw: str | None) -> str:
    return resolve_session_name(raw)


def _session_root(workspace_cli: str | None, analytics_session: str | None) -> str:
    return resolve_session_root(workspace_cli, analytics_session)


def _session_artifacts_dir(workspace_cli: str | None, analytics_session: str | None, category: str) -> str:
    return resolve_session_artifacts_dir(workspace_cli, analytics_session, category)


def _default_relative_output(
    workspace_cli: str | None,
    analytics_session: str | None,
    category: str,
    file_name: str,
    raw: str | None,
) -> str:
    return resolve_default_relative_output(workspace_cli, analytics_session, category, file_name, raw)


def _resolve_data_yaml_for_run(run_dir: str, workspace_cli: str | None) -> tuple[str | None, str | None]:
    candidates = _collect_data_yaml_candidates_for_run(run_dir, workspace_cli)
    if candidates:
        return candidates[0]
    return None, None


def _collect_data_yaml_candidates_for_run(run_dir: str, workspace_cli: str | None) -> list[tuple[str, str]]:
    return collect_data_yaml_candidates_for_run(
        run_dir,
        workspace_cli,
        unified_read_enabled=_unified_read_enabled(),
        dataset_name_resolver=lambda rd: _build_run_record_unified(rd).dataset_name or None,
    )


def _has_split_dir(data_yaml_path: str, split_name: str) -> bool:
    try:
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            return False
        split_rel = str(cfg.get(split_name, "")).strip()
        if not split_rel:
            return False
        base_dir = os.path.dirname(os.path.abspath(data_yaml_path))
        split_path = os.path.abspath(os.path.join(base_dir, split_rel))
        return os.path.isdir(split_path)
    except Exception:
        return False


def _auto_select_data_yaml(
    baseline: str,
    others: list[str],
    workspace_cli: str | None,
    preferred_split: str | None = None,
) -> str | None:
    candidates: list[str] = []
    source_by_path: dict[str, str] = {}
    for rd in [baseline] + others:
        for p, src in _collect_data_yaml_candidates_for_run(rd, workspace_cli):
            if p not in candidates:
                candidates.append(p)
            source_by_path[p] = src
    if not candidates:
        return None
    if preferred_split:
        viable = [p for p in candidates if _has_split_dir(p, preferred_split)]
        if viable:
            candidates = viable
    if len(candidates) == 1:
        src = source_by_path.get(candidates[0], "unknown")
        print(f"[INFO] Auto-detected data.yaml: {candidates[0]} (source: {src})")
        return candidates[0]
    print("[INFO] Multiple data.yaml candidates detected:")
    for idx, path in enumerate(candidates, start=1):
        src = source_by_path.get(path, "unknown")
        print(f"  {idx}. {path}  [source: {src}]")
    picked = _workflow_attr("prompt_choice")(
        "Select data.yaml", candidates, default=candidates[0], show_options=False
    )
    return picked


def _build_run_data_yaml_map(
    run_dirs: list[str],
    workspace_cli: str | None,
    *,
    preferred_split: str | None = None,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    run_to_yaml: dict[str, str] = {}
    run_to_source: dict[str, str] = {}
    unresolved: list[str] = []
    for rd in run_dirs:
        candidates = _collect_data_yaml_candidates_for_run(rd, workspace_cli)
        if preferred_split:
            viable = [(p, src) for p, src in candidates if _has_split_dir(p, preferred_split)]
            if viable:
                candidates = viable
        if not candidates:
            unresolved.append(rd)
            continue
        run_to_yaml[rd] = candidates[0][0]
        run_to_source[rd] = candidates[0][1]
    return run_to_yaml, run_to_source, unresolved


def _group_runs_by_data_yaml(
    run_dirs: list[str],
    run_data_yaml_map: dict[str, str],
) -> tuple[dict[str, list[str]], list[str]]:
    groups: dict[str, list[str]] = {}
    unresolved: list[str] = []
    for rd in run_dirs:
        dy = str(run_data_yaml_map.get(rd) or "").strip()
        if not dy:
            unresolved.append(rd)
            continue
        groups.setdefault(dy, []).append(rd)
    return groups, unresolved


def _recompute_run_test_metrics(
    run_dir: str,
    data_yaml: str,
    split: str,
    *,
    val_batch: int = 1,
    val_imgsz: int = 640,
    val_half: bool = True,
    gpu_only: bool = False,
) -> dict[str, Any]:
    return _svc_recompute_run_test_metrics(
        run_dir,
        data_yaml,
        split,
        val_batch=val_batch,
        val_imgsz=val_imgsz,
        val_half=val_half,
        gpu_only=gpu_only,
        preferred_run_model_path_cb=preferred_run_model_path,
        materialize_preferred_run_model_cb=materialize_preferred_run_model,
        clear_gpu_memory_cb=_clear_gpu_memory,
        resolve_run_val_profile_cb=_resolve_run_val_profile,
        ultralytics_sidecar_dir_cb=ultralytics_sidecar_dir,
        run_val_memory_safe_cb=_run_val_memory_safe,
    )


def _load_dataset_class_names(data_yaml: str) -> dict[int, str]:
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except Exception:
        return {}
    names = payload.get("names")
    if isinstance(names, dict):
        out: dict[int, str] = {}
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}




def cmd_scan(args: argparse.Namespace) -> None:
    targets = discover_analysis_targets(
        workspace_cli=getattr(args, "workspace", None),
        models_root_cli=getattr(args, "models_root_cli", None),
    )
    run_scan_command(
        runs=targets,
        flat_row_for_run=_flat_row_for_run,
    )


def cmd_export_table(args: argparse.Namespace) -> None:
    runs = discover_analysis_targets(
        workspace_cli=getattr(args, "workspace", None),
        models_root_cli=getattr(args, "models_root_cli", None),
    )
    out_path = _default_relative_output(
        args.workspace, args.analytics_session, "table", "runs_summary.csv", args.output
    )
    analytics_dir: str | None = None
    if args.analytics_session is not None:
        session_name = args.analytics_session.strip()
        if not session_name:
            print("[ERROR] --analytics-session cannot be empty.", file=sys.stderr)
            sys.exit(1)
        try:
            ws = resolve_workspace_root(args.workspace)
        except ValueError:
            print(
                f"[ERROR] --analytics-session requires --workspace or {WORKSPACE_ENV_VAR}.",
                file=sys.stderr,
            )
            sys.exit(1)
        layout = WorkspaceLayout(ws)
        os.makedirs(layout.analytics, exist_ok=True)
        analytics_dir = os.path.join(layout.analytics, session_name)
        os.makedirs(analytics_dir, exist_ok=True)
        out_path = os.path.join(analytics_dir, os.path.basename(args.output))
    rc = export_runs_table(
        runs=runs,
        out_path=out_path,
        latest_test_metrics_path=latest_test_metrics_path,
        results_csv_path=results_csv_path,
        pick_map_column=pick_map_column,
        flat_row_for_run=_flat_row_for_run,
    )
    if rc != 0:
        sys.exit(rc)
    if analytics_dir is not None:
        manifest = {
            "scan_root": args.models_root,
            "run_directories": runs,
            "summary_csv": out_path,
        }
        with open(os.path.join(analytics_dir, "session.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[OK] Session manifest: {os.path.join(analytics_dir, 'session.json')}")


def _write_system_profile_compare_csv(run_dirs: list[str], out_csv: str) -> str | None:
    return _svc_write_system_profile_compare_csv(
        run_dirs,
        out_csv,
        flat_row_for_run_cb=_flat_row_for_run,
    )


def _write_test_system_profile_compare_csv(run_dirs: list[str], out_csv: str) -> str | None:
    return _svc_write_test_system_profile_compare_csv(
        run_dirs,
        out_csv,
        flat_row_for_run_cb=_flat_row_for_run,
        read_test_system_profile_by_format_artifacts_cb=read_test_system_profile_by_format_artifacts,
    )


def _finalize_compare_analytics_session(
    args: argparse.Namespace,
    baseline: str,
    others: list[str],
    out_csv: str,
    out_png: str,
    bar_path: str | None,
    insights_path: str | None,
) -> None:
    _svc_finalize_compare_analytics_session(
        args=args,
        baseline=baseline,
        others=others,
        out_csv=out_csv,
        out_png=out_png,
        bar_path=bar_path,
        insights_path=insights_path,
        resolve_workspace_root_cb=resolve_workspace_root,
        workspace_layout_cls=WorkspaceLayout,
        workspace_env_var=WORKSPACE_ENV_VAR,
    )


def _resolve_compare_png_path(
    workspace_cli: str | None,
    analytics_session: str | None,
    out_png_cli: str,
) -> str:
    return _svc_resolve_compare_png_path(
        out_png_cli,
        out_csv="compare_delta.csv",
        resolve_workspace_root_cb=resolve_workspace_root,
        workspace_layout_cls=WorkspaceLayout,
        workspace_cli=workspace_cli,
        session_name=analytics_session,
    )


def _resolve_compare_artifact_path(
    workspace_cli: str | None,
    analytics_session: str | None,
    category: str,
    raw_path: str,
    default_file_name: str,
) -> str:
    _ = category
    return _svc_resolve_compare_artifact_path(
        _default_relative_output(
            workspace_cli, analytics_session, category, default_file_name, raw_path
        ),
        _session_root(workspace_cli, analytics_session),
    )


def cmd_compare(args: argparse.Namespace) -> None:
    if not getattr(args, "baseline", None) or not getattr(args, "others", None):
        if not sys.stdin.isatty():
            print(
                "[ERROR] compare requires --baseline/--others or interactive terminal mode.",
                file=sys.stderr,
            )
            sys.exit(2)
        forwarded = argparse.Namespace(
            output_dir=None,
            metric_column=args.metric_column,
            workspace=args.workspace,
            analytics_session=args.analytics_session,
            models_root=args.models_root,
            preset="quality",
            quality_metrics="mAP50-95,Box-F1",
            data_yaml=None,
            benchmark_split="test",
            benchmark_frames=100,
            benchmark_device="cpu",
            benchmark_half=False,
            speed_metric="avg_inference_ms_per_frame",
            recompute_missing_metrics=True,
            recompute_split="test",
            filter_dataset=None,
            filter_model=None,
            filter_training_ok=None,
            filter_testing_ok=None,
        )
        cmd_interactive(forwarded)
        return

    baseline = os.path.abspath(args.baseline)
    others = [os.path.abspath(p) for p in args.others]
    out_png = _resolve_compare_png_path(args.workspace, args.analytics_session, args.out_png)
    out_csv = _resolve_compare_artifact_path(
        args.workspace, args.analytics_session, "compare", args.out_csv, "compare_delta.csv"
    )
    out_insights = _resolve_compare_artifact_path(
        args.workspace, args.analytics_session, "compare", args.out_insights, "compare_insights.txt"
    )
    bar_path: str | None = None
    all_runs = [baseline] + others
    for p in all_runs:
        if not is_run_directory(p):
            print(f"[ERROR] Not a run (missing training_metadata.json): {p}", file=sys.stderr)
            sys.exit(1)

    base_metrics = _read_test_metrics_for_run(baseline)
    if not base_metrics:
        print("[WARN] Baseline has no test_metrics*.csv; deltas only from train/results.csv", file=sys.stderr)
    bar_path, _delta_rows = run_compare_workflow(
        baseline=baseline,
        others=others,
        out_csv=out_csv,
        out_insights=out_insights,
        out_png=out_png,
        metric_column=args.metric_column,
        read_test_metrics_for_run=_read_test_metrics_for_run,
        build_delta_rows=build_delta_rows,
        generate_compare_insights=generate_compare_insights,
        results_csv_path=results_csv_path,
        pick_map_column=pick_map_column,
        default_map_col=DEFAULT_MAP_COL,
    )

    _finalize_compare_analytics_session(
        args, baseline, others, out_csv, out_png, bar_path, out_insights
    )


def _matches_optional_bool(value: bool | None, expected: bool | None) -> bool:
    return _svc_matches_optional_bool(value, expected)


def _build_run_record_unified(run_dir: str) -> RunRecord:
    return _svc_build_run_record_unified(
        run_dir,
        read_test_metrics_for_run_cb=_read_test_metrics_for_run,
    )


def _read_test_metrics_for_run(
    run_dir: str,
    *,
    format_name: str = "pt",
    source_kind: str | None = None,
) -> dict[str, Any]:
    return _svc_read_test_metrics_for_run(run_dir, format_name=format_name, source_kind=source_kind)


def _flat_row_unified(run_dir: str) -> dict[str, Any]:
    return _svc_flat_row_unified(run_dir, build_run_record_cb=_build_run_record_unified)

def _flat_row_for_run(run_dir: str) -> dict[str, Any]:
    return _flat_row_unified(run_dir)


def _filtered_run_records(args: argparse.Namespace) -> list[tuple[str, Any]]:
    return _svc_filtered_run_records(args, build_run_record_cb=_build_run_record_unified)


def _workflow_attr(name: str):
    """Resolve via services dispatch module (patchable in tests)."""

    from smartrain.services.analyze import workflow_dispatch as _wd

    return getattr(_wd, name)


def _workflow_analyze_cmd(name: str):
    """Delegate cmd_* via workflows facade (patchable in tests)."""

    def _run(args: argparse.Namespace) -> None:
        return _workflow_attr(name)(args)

    return _run


def _workflow_callback(name: str):
    """Delegate arbitrary facade attr (prompts, _recompute_*, etc.)."""

    def _call(*args: Any, **kwargs: Any) -> Any:
        return _workflow_attr(name)(*args, **kwargs)

    return _call


def cmd_interactive(args: argparse.Namespace) -> None:
    indexed = _filtered_run_records(args)
    if not indexed:
        print("No runs found after filters.")
        return
    run_interactive_workflow(
        args=args,
        indexed=indexed,
        session_artifacts_dir=_session_artifacts_dir,
        cmd_compare=_workflow_analyze_cmd("cmd_compare"),
        cmd_test_metrics_plot=_workflow_analyze_cmd("cmd_test_metrics_plot"),
        cmd_inference_benchmark=_workflow_analyze_cmd("cmd_inference_benchmark"),
        cmd_inference_plot=_workflow_analyze_cmd("cmd_inference_plot"),
        cmd_pr_curves=_workflow_analyze_cmd("cmd_pr_curves"),
        runs_with_missing_metrics=_runs_with_missing_metrics,
        auto_select_data_yaml=_auto_select_data_yaml,
        prompt_choice=_workflow_attr("prompt_choice"),
    )


def cmd_leaderboard(args: argparse.Namespace) -> None:
    if not getattr(args, "quality_metric", None) and sys.stdin.isatty():
        args.quality_metric = prompt_text("Quality metric", default="mAP50-95").strip() or "mAP50-95"
    if not getattr(args, "speed_metric", None) and sys.stdin.isatty():
        args.speed_metric = prompt_text("Speed metric", default="avg_inference_fps").strip() or "avg_inference_fps"
    runs = discover_analysis_targets(
        workspace_cli=getattr(args, "workspace", None),
        models_root_cli=getattr(args, "models_root_cli", None),
    )
    selected_norm = {
        os.path.abspath(os.path.expanduser(str(p)))
        for p in (getattr(args, "selected_run_dirs", None) or [])
        if str(p).strip()
    }
    if selected_norm:
        runs = [r for r in runs if os.path.abspath(r) in selected_norm]
        print(f"[INFO] Leaderboard scope: {len(runs)} run(s) selected")
    records = build_leaderboard_records(
        runs=runs,
        speed_metric=str(args.speed_metric or ""),
        quality_metric=str(args.quality_metric or ""),
        weight_quality=float(args.weight_quality),
        weight_speed=float(args.weight_speed),
        weight_stability=float(args.weight_stability),
        load_run_record=_build_run_record_unified,
        read_test_performance_by_format_artifacts=read_test_performance_by_format_artifacts,
        compute_composite_score=compute_composite_score,
    )
    if not records:
        print("[ERROR] No runs for leaderboard.", file=sys.stderr)
        sys.exit(1)
    out_csv = _default_relative_output(
        args.workspace, args.analytics_session, "leaderboard", "leaderboard.csv", args.out_csv
    )
    rc = write_leaderboard_csv(records=records, out_csv=out_csv)
    if rc != 0:
        print("[ERROR] No runs with enough metrics for leaderboard.", file=sys.stderr)
        sys.exit(1)


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w.\-+]+", "_", value, flags=re.UNICODE).strip("._") or "class"


def _collect_confidence_recommendation_tables(run_dirs: list[str], out_dir: str) -> dict[str, str]:
    return collect_confidence_recommendation_tables(
        run_dirs=run_dirs,
        out_dir=out_dir,
        flat_row_for_run=_flat_row_for_run,
        recommendation_file_path=recommendation_file_path,
        read_recommendation_file=read_recommendation_file,
    )


def _write_speed_quality_artifacts(
    session_root: str,
    inference_csv: str,
    requested_runs: list[str],
    metric_sources_payload: dict[str, Any] | None,
    *,
    scatter_x: str,
    scatter_y: str,
    run_data_yaml_map: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    return write_speed_quality_artifacts(
        session_root=session_root,
        inference_csv=inference_csv,
        requested_runs=requested_runs,
        metric_sources_payload=metric_sources_payload,
        scatter_x=scatter_x,
        scatter_y=scatter_y,
        run_data_yaml_map=run_data_yaml_map,
        read_test_metrics_for_run=_read_test_metrics_for_run,
    )


def _runs_with_missing_metrics(
    run_dirs: list[str],
    requested_metrics: list[str],
    *,
    data_yaml: str | None = None,
    workspace: str | None = None,
    split: str = "test",
) -> list[str]:
    return _svc_runs_with_missing_metrics(
        run_dirs,
        requested_metrics,
        data_yaml=data_yaml,
        workspace=workspace,
        split=split,
        collect_missing_metrics_recompute_plan_cb=_collect_missing_metrics_recompute_plan,
    )


def _collect_missing_metrics_recompute_plan(
    run_dirs: list[str],
    requested_metrics: list[str],
    *,
    data_yaml: str | None = None,
    run_data_yaml_map: dict[str, str] | None = None,
    workspace: str | None = None,
    split: str = "test",
) -> dict[str, list[dict[str, Any]]]:
    return _svc_collect_missing_metrics_recompute_plan(
        run_dirs,
        requested_metrics,
        data_yaml=data_yaml,
        run_data_yaml_map=run_data_yaml_map,
        workspace=workspace,
        split=split,
        read_test_metrics_for_run_cb=_read_test_metrics_for_run,
        preferred_run_model_path_cb=preferred_run_model_path,
        resolve_data_yaml_for_run_cb=_resolve_data_yaml_for_run,
        load_recompute_status_cb=_load_recompute_status,
    )


def _recompute_status_path(run_dir: str, fingerprint: str) -> str:
    return _svc_recompute_status_path(run_dir, fingerprint, run_cache_root_cb=run_cache_root)


def _recompute_status_fingerprint(run_dir: str, data_yaml: str, split: str, requested_metrics: list[str]) -> str:
    return _svc_recompute_status_fingerprint(
        run_dir,
        data_yaml,
        split,
        requested_metrics,
        compute_fingerprint_cb=compute_fingerprint,
        data_yaml_hash_cb=data_yaml_hash,
        weights_hash_cb=weights_hash,
    )


def _load_recompute_status(
    run_dir: str,
    data_yaml: str,
    split: str,
    requested_metrics: list[str],
) -> dict[str, Any] | None:
    return _svc_load_recompute_status(
        run_dir,
        data_yaml,
        split,
        requested_metrics,
        recompute_status_fingerprint_cb=_recompute_status_fingerprint,
        recompute_status_path_cb=_recompute_status_path,
    )


def _save_recompute_status(
    run_dir: str,
    data_yaml: str,
    split: str,
    requested_metrics: list[str],
    *,
    resolved: list[str],
    unresolved: list[str],
    status: str,
) -> None:
    _svc_save_recompute_status(
        run_dir,
        data_yaml,
        split,
        requested_metrics,
        resolved=resolved,
        unresolved=unresolved,
        status=status,
        recompute_status_fingerprint_cb=_recompute_status_fingerprint,
        recompute_status_path_cb=_recompute_status_path,
        append_cache_entry_cb=append_cache_entry,
    )


def _build_abbreviations_for_report(run_dirs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    dataset_to_idx: dict[str, int] = {}
    dataset_counter = 1
    for idx, rd in enumerate(run_dirs, start=1):
        run_name = os.path.basename(rd.rstrip(os.sep))
        if len(run_name) > 22:
            out[run_name] = f"R{idx}"
        try:
            rec = _build_run_record_unified(rd)
            model = str((rec.model or "")).strip()
            dataset_name = str((rec.dataset_name or "")).strip()
        except Exception:
            model = ""
            dataset_name = ""
        if model and len(model) > 16:
            out[model] = f"M{idx}"
        if dataset_name:
            if dataset_name not in dataset_to_idx:
                dataset_to_idx[dataset_name] = dataset_counter
                dataset_counter += 1
            out.setdefault(dataset_name, f"D{dataset_to_idx[dataset_name]}")
    return out


def _collect_ultralytics_test_artifacts(
    session_root: str,
    run_dirs: list[str],
    abbreviations: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    return _svc_collect_ultralytics_test_artifacts(
        session_root,
        run_dirs,
        abbreviations,
        run_test_backend_dir_cb=run_test_backend_dir,
        build_run_record_unified_cb=_build_run_record_unified,
    )


def _write_format_compare_artifacts(session_root: str, run_dirs: list[str]) -> dict[str, str] | None:
    return write_format_compare_artifacts(session_root, run_dirs)


def _resolve_pr_output_png(
    workspace_cli: str | None,
    out_png_cli: str | None,
    runs_group_dir: str,
) -> str:
    return _svc_resolve_pr_output_png(
        workspace_cli,
        out_png_cli,
        runs_group_dir,
        resolve_workspace_root_cb=resolve_workspace_root,
        workspace_layout_cls=WorkspaceLayout,
    )


def _is_workers_pickle_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    markers = (
        "can't get local object",
        "pickle",
        "lambda",
        "ran out of input",
        "multiprocessing",
    )
    return any(m in msg for m in markers)


def _is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "cuda out of memory" in msg or ("out of memory" in msg and "cuda" in msg)


def _resolve_selected_run_dirs(
    runs_group_dir: str,
    selected_run_dirs: list[str] | tuple[str, ...] | None,
) -> list[str]:
    return _svc_resolve_selected_run_dirs(runs_group_dir, selected_run_dirs)


def cmd_pr_curves(args: argparse.Namespace) -> None:
    _svc_run_pr_curves(
        args,
        prompt_text_cb=prompt_text,
        resolve_selected_run_dirs_cb=_resolve_selected_run_dirs,
        load_dataset_class_names_cb=_load_dataset_class_names,
        preferred_run_model_path_cb=preferred_run_model_path,
        run_cache_root_cb=run_cache_root,
        compute_fingerprint_cb=compute_fingerprint,
        data_yaml_hash_cb=data_yaml_hash,
        weights_hash_cb=weights_hash,
        clear_gpu_memory_cb=_clear_gpu_memory,
        resolve_run_val_profile_cb=_resolve_run_val_profile,
        ultralytics_sidecar_dir_cb=ultralytics_sidecar_dir,
        run_val_memory_safe_cb=_run_val_memory_safe,
        extract_pr_curve_cb=extract_pr_curve_from_ultralytics_metrics,
        extract_pr_curve_per_class_cb=extract_pr_curve_per_class_from_ultralytics_metrics,
        append_cache_entry_cb=append_cache_entry,
        safe_name_cb=_safe_name,
        resolve_workspace_root_cb=resolve_workspace_root,
        workspace_layout_cls=WorkspaceLayout,
    )


def _collect_split_images(data_yaml_path: str, split_name: str, limit: int) -> list[str]:
    return _svc_collect_split_images(data_yaml_path, split_name, limit)


def _resolve_inference_csv_path(
    workspace_cli: str | None,
    out_csv_cli: str | None,
    runs_group_dir: str,
) -> str:
    return _svc_resolve_inference_csv_path(
        workspace_cli,
        out_csv_cli,
        runs_group_dir,
        resolve_workspace_root_cb=resolve_workspace_root,
        workspace_layout_cls=WorkspaceLayout,
    )


def cmd_inference_benchmark(args: argparse.Namespace) -> None:
    _svc_run_inference_benchmark(
        args,
        prompt_text_cb=prompt_text,
        resolve_workspace_root_cb=resolve_workspace_root,
        workspace_layout_cls=WorkspaceLayout,
        preferred_run_model_path_cb=preferred_run_model_path,
        run_cache_root_cb=run_cache_root,
        compute_fingerprint_cb=compute_fingerprint,
        data_yaml_hash_cb=data_yaml_hash,
        weights_hash_cb=weights_hash,
        append_cache_entry_cb=append_cache_entry,
        ultralytics_sidecar_dir_cb=ultralytics_sidecar_dir,
        clear_gpu_memory_cb=_clear_gpu_memory,
    )


def _resolve_inference_plot_png(
    workspace_cli: str | None,
    out_png_cli: str | None,
    csv_path: str,
) -> str:
    return _svc_resolve_inference_plot_png(
        workspace_cli,
        out_png_cli,
        csv_path,
        resolve_workspace_root_cb=resolve_workspace_root,
        workspace_layout_cls=WorkspaceLayout,
    )


def cmd_inference_plot(args: argparse.Namespace) -> None:
    _svc_run_inference_plot(
        args,
        prompt_text_cb=prompt_text,
        default_relative_output_cb=_default_relative_output,
        resolve_workspace_root_cb=resolve_workspace_root,
        workspace_layout_cls=WorkspaceLayout,
    )


def _resolve_test_metrics_plot_png(
    workspace_cli: str | None,
    out_dir_cli: str | None,
    runs_group_dir: str,
    metric: str,
) -> str:
    return _svc_resolve_test_metrics_plot_png(
        workspace_cli,
        out_dir_cli,
        runs_group_dir,
        metric,
        resolve_workspace_root_cb=resolve_workspace_root,
        workspace_layout_cls=WorkspaceLayout,
    )


def cmd_test_metrics_plot(args: argparse.Namespace) -> None:
    _svc_run_test_metrics_plot(
        args,
        prompt_text_cb=prompt_text,
        resolve_selected_run_dirs_cb=_resolve_selected_run_dirs,
        latest_test_metrics_path_cb=latest_test_metrics_path,
        load_recompute_status_cb=_load_recompute_status,
        save_recompute_status_cb=_workflow_callback("_save_recompute_status"),
        resolve_data_yaml_for_run_cb=_resolve_data_yaml_for_run,
        recompute_run_test_metrics_cb=_workflow_callback("_recompute_run_test_metrics"),
        compute_fingerprint_cb=compute_fingerprint,
        run_cache_root_cb=run_cache_root,
        data_yaml_hash_cb=data_yaml_hash,
        append_cache_entry_cb=append_cache_entry,
        resolve_workspace_root_cb=resolve_workspace_root,
        workspace_layout_cls=WorkspaceLayout,
    )


def cmd_all(args: argparse.Namespace) -> None:
    _svc_run_all_command(
        args=args,
        prepare_all_selection_cb=_svc_prepare_all_selection,
        resolve_all_data_yaml_context_cb=_svc_resolve_all_data_yaml_context,
        session_root_cb=_session_root,
        filtered_run_records_cb=_filtered_run_records,
        prompt_int_cb=_workflow_attr("prompt_int"),
        prompt_text_cb=_workflow_attr("prompt_text"),
        prompt_choice_cb=_workflow_attr("prompt_choice"),
        build_run_data_yaml_map_cb=_build_run_data_yaml_map,
        auto_select_data_yaml_cb=_auto_select_data_yaml,
        run_all_baseline_artifacts_cb=_svc_run_all_baseline_artifacts,
        run_all_quality_stage_cb=_svc_run_all_quality_stage,
        run_all_speed_stage_cb=_svc_run_all_speed_stage,
        run_all_pr_stage_cb=_svc_run_all_pr_stage,
        finalize_all_session_cb=_finalize_all_session_with_replay,
        default_map_col=DEFAULT_MAP_COL,
        cmd_compare_cb=_workflow_analyze_cmd("cmd_compare"),
        cmd_export_table_cb=_workflow_analyze_cmd("cmd_export_table"),
        write_system_profile_compare_csv_cb=_write_system_profile_compare_csv,
        write_test_system_profile_compare_csv_cb=_write_test_system_profile_compare_csv,
        cmd_leaderboard_cb=_workflow_analyze_cmd("cmd_leaderboard"),
        collect_missing_metrics_recompute_plan_cb=_collect_missing_metrics_recompute_plan,
        cmd_test_metrics_plot_cb=_workflow_analyze_cmd("cmd_test_metrics_plot"),
        group_runs_by_data_yaml_cb=_group_runs_by_data_yaml,
        cmd_inference_benchmark_cb=_workflow_analyze_cmd("cmd_inference_benchmark"),
        cmd_inference_plot_cb=_workflow_analyze_cmd("cmd_inference_plot"),
        write_speed_quality_artifacts_cb=_write_speed_quality_artifacts,
        cmd_pr_curves_cb=_workflow_analyze_cmd("cmd_pr_curves"),
        safe_name_cb=_safe_name,
        build_abbreviations_for_report_cb=_build_abbreviations_for_report,
        collect_ultralytics_test_artifacts_cb=_workflow_callback("_collect_ultralytics_test_artifacts"),
        write_format_compare_artifacts_cb=_write_format_compare_artifacts,
        collect_confidence_recommendation_tables_cb=_collect_confidence_recommendation_tables,
        write_manifest_cb=write_manifest,
        write_analysis_report_cb=write_analysis_report,
    )

def build_analyze_arg_parser() -> argparse.ArgumentParser:
    global _ANALYZE_ALL_SUBPARSER
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR}) used for default runs root",
    )
    common.add_argument(
        "--models-root",
        type=str,
        default=None,
        help="Explicit root directory to scan for run folders with training_metadata.json",
    )
    common.add_argument(
        "--analytics-session",
        type=str,
        default=None,
        help="Subdirectory in workspace/analytics for artifacts and session.json (export-table, compare, interactive)",
    )

    parser = CliArgumentParser(description="YOLO training runs analysis (Ultralytics)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", parents=[common], help="List detected runs")
    p_scan.set_defaults(func=cmd_scan)

    p_all = sub.add_parser("all", parents=[common], help="Interactive full analysis session with report outputs")
    p_all.add_argument("--no-pdf", action="store_true", help="Skip PDF export for analyze report")
    p_all.add_argument("--no-odt", action="store_true", help="Skip ODT export for analyze report")
    p_all.add_argument("--baseline", type=str, default=None, help="Baseline run directory (non-interactive mode)")
    p_all.add_argument("--others", type=str, nargs="+", default=None, help="Candidate run directories (non-interactive mode)")
    p_all.add_argument(
        "--profile",
        type=str,
        choices=["quality", "speed", "full"],
        default=None,
        help="Analyze profile (non-interactive mode)",
    )
    p_all.add_argument("--data-yaml", type=str, default=None, help="Path to data.yaml for speed/full profile")
    p_all.add_argument(
        "--recompute-missing-metrics",
        dest="recompute_missing_metrics_choice",
        choices=["yes", "no"],
        default=None,
        help="Enable/disable missing metrics recompute without interactive prompt",
    )
    p_all.add_argument(
        "--report-languages",
        type=str,
        default="ru,en",
        help="Comma-separated report languages, e.g. ru,en or en",
    )
    p_all.add_argument(
        "--strict-diagnostics",
        action="store_true",
        help="Fail analyze all if critical artifacts are missing",
    )
    p_all.add_argument(
        "--scatter-x",
        type=str,
        default="avg_inference_ms_per_frame",
        help="X-axis metric for speed/quality scatter from inference benchmark CSV",
    )
    p_all.add_argument(
        "--scatter-y",
        type=str,
        default="mAP50-95",
        help="Y-axis metric for speed/quality scatter from test metrics",
    )
    p_all.add_argument(
        "--ensure-ultralytics-test",
        dest="ensure_ultralytics_test",
        action="store_true",
        default=None,
        help="Run full PT Ultralytics test when canonical test artifacts are incomplete (default: on for profile full)",
    )
    p_all.add_argument(
        "--no-ensure-ultralytics-test",
        dest="ensure_ultralytics_test",
        action="store_false",
        help="Skip auto PT Ultralytics test before report collection",
    )
    p_all.add_argument("--val-batch", type=int, default=1, help="Validation batch size for GPU memory-safe val()")
    p_all.add_argument("--val-imgsz", type=int, default=640, help="Validation image size for GPU memory-safe val()")
    p_all.add_argument("--val-half", dest="val_half", action="store_true", default=True, help="Use FP16 for validation on GPU")
    p_all.add_argument("--no-val-half", dest="val_half", action="store_false", help="Disable FP16 for validation")
    p_all.add_argument("--gpu-only-val", dest="gpu_only_val", action="store_true", default=True, help="Do not fallback to CPU in val()")
    p_all.add_argument("--allow-cpu-fallback", dest="gpu_only_val", action="store_false", help="Allow CPU fallback when GPU val() fails")
    p_all.set_defaults(func=cmd_all)
    _ANALYZE_ALL_SUBPARSER = p_all

    p_exp = sub.add_parser("export-table", parents=[common], help="Export summary CSV for all runs")
    p_exp.add_argument("-o", "--output", type=str, default="runs_summary.csv")
    p_exp.set_defaults(func=cmd_export_table)

    p_cmp = sub.add_parser(
        "compare",
        parents=[common],
        help="Compare baseline run with others (CSV + plots)",
    )
    p_cmp.add_argument("--baseline", type=str, required=False, help="Baseline run directory")
    p_cmp.add_argument(
        "--others",
        type=str,
        nargs="+",
        required=False,
        help="One or more run directories to compare with baseline",
    )
    p_cmp.add_argument("-o", "--out-csv", type=str, default="compare_delta.csv")
    p_cmp.add_argument("--out-png", type=str, default="compare_curves.png")
    p_cmp.add_argument(
        "--out-insights",
        type=str,
        default="compare_insights.txt",
        help="Path to plain text auto-insights report based on delta metrics",
    )
    p_cmp.add_argument(
        "--metric-column",
        type=str,
        default=DEFAULT_MAP_COL,
        help="Metric column from train/results.csv used for line chart",
    )
    p_cmp.set_defaults(func=cmd_compare)

    p_lb = sub.add_parser(
        "leaderboard",
        parents=[common],
        help="Build leaderboard by composite score from test metrics",
    )
    p_lb.add_argument("-o", "--out-csv", type=str, default="leaderboard.csv")
    p_lb.add_argument("--quality-metric", type=str, default="mAP50-95")
    p_lb.add_argument("--speed-metric", type=str, default="avg_inference_fps")
    p_lb.add_argument("--weight-quality", type=float, default=0.6)
    p_lb.add_argument("--weight-speed", type=float, default=0.25)
    p_lb.add_argument("--weight-stability", type=float, default=0.15)
    p_lb.set_defaults(func=cmd_leaderboard)

    p_pr = sub.add_parser(
        "pr-curves",
        parents=[common],
        help="Run test val for all models in folder, save per-run pr.csv, and build combined PR plot",
    )
    p_pr.add_argument(
        "--runs-group-dir",
        type=str,
        required=False,
        help="Directory like runs/<dataset_name>/ containing model run folders",
    )
    p_pr.add_argument(
        "--data-yaml",
        type=str,
        required=False,
        help="Path to dataset data.yaml for split=test",
    )
    p_pr.add_argument(
        "--out-png",
        type=str,
        default=None,
        help="Output PNG path for combined PR plot (default: workspace/analytics/pr_curves/pr_all_classes_<dataset>.png)",
    )
    p_pr.add_argument("--pr-per-class", action="store_true", default=True, help="Export PR curves per class")
    p_pr.add_argument(
        "--reuse-run-cache",
        action="store_true",
        default=True,
        help="Reuse run-level cached PR artifacts when fingerprints match",
    )
    p_pr.add_argument("--val-batch", type=int, default=1, help="Validation batch size")
    p_pr.add_argument("--val-imgsz", type=int, default=640, help="Validation image size")
    p_pr.add_argument("--val-half", dest="val_half", action="store_true", default=True, help="Use FP16 on GPU for val()")
    p_pr.add_argument("--no-val-half", dest="val_half", action="store_false", help="Disable FP16 for val()")
    p_pr.add_argument("--gpu-only-val", dest="gpu_only_val", action="store_true", default=True, help="Do not fallback to CPU in val()")
    p_pr.add_argument("--allow-cpu-fallback", dest="gpu_only_val", action="store_false", help="Allow CPU fallback when GPU val() fails")
    p_pr.set_defaults(func=cmd_pr_curves)

    p_inf = sub.add_parser(
        "inference-benchmark",
        parents=[common],
        help="Benchmark inference speed for all models in runs/<dataset>/ (average over N frames)",
    )
    p_inf.add_argument(
        "--runs-group-dir",
        type=str,
        required=False,
        help="Directory like runs/<dataset_name>/ containing model run folders",
    )
    p_inf.add_argument(
        "--data-yaml",
        type=str,
        required=False,
        help="Path to dataset data.yaml",
    )
    p_inf.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split used for benchmark frames",
    )
    p_inf.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames used to compute average latency",
    )
    p_inf.add_argument(
        "--device",
        type=str,
        default="0",
        help="Inference device (e.g. cpu, 0, 0,1)",
    )
    p_inf.add_argument(
        "--half",
        action="store_true",
        help="Enable FP16 inference (mainly useful on GPU)",
    )
    p_inf.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Output CSV path (default: workspace/analytics/inference_tests/<dataset>.csv)",
    )
    p_inf.add_argument(
        "--reuse-run-cache",
        action="store_true",
        default=True,
        help="Reuse run-level cached inference benchmark artifacts when fingerprints match",
    )
    p_inf.set_defaults(func=cmd_inference_benchmark)

    p_inf_plot = sub.add_parser(
        "inference-plot",
        parents=[common],
        help="Build bar chart from inference benchmark CSV",
    )
    p_inf_plot.add_argument(
        "--csv",
        type=str,
        required=False,
        help="Path to CSV generated by analyze inference-benchmark",
    )
    p_inf_plot.add_argument(
        "--metric",
        type=str,
        default="avg_inference_ms_per_frame",
        help=(
            "Column to plot, e.g.: avg_inference_ms_per_frame, "
            "avg_total_ms_per_frame, avg_total_fps, avg_inference_fps."
        ),
    )
    p_inf_plot.add_argument(
        "--out-png",
        type=str,
        default=None,
        help="Output PNG path (default: analytics/inference_tests/<csv_name>_bars.png)",
    )
    p_inf_plot.set_defaults(func=cmd_inference_plot)

    p_tm_plot = sub.add_parser(
        "test-metrics-plot",
        parents=[common],
        help="Build charts from test_metrics.csv for models in runs/<dataset>/",
    )
    p_tm_plot.add_argument(
        "--runs-group-dir",
        type=str,
        required=False,
        help="Directory like runs/<dataset_name>/ containing model run folders",
    )
    p_tm_plot.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=False,
        help=(
            "One or more metrics from test_metrics.csv, e.g.: "
            "mAP50 mAP50-95 Box-P Box-R Box-F1"
        ),
    )
    p_tm_plot.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save PNG files (default: analytics/metrics_comparison)",
    )
    p_tm_plot.add_argument(
        "--recompute-missing-metrics",
        action="store_true",
        default=False,
        help="Recompute missing requested metrics from run model + detected data.yaml",
    )
    p_tm_plot.add_argument(
        "--recompute-split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split used for metric recomputation",
    )
    p_tm_plot.add_argument("--val-batch", type=int, default=1, help="Validation batch size for recompute")
    p_tm_plot.add_argument("--val-imgsz", type=int, default=640, help="Validation image size for recompute")
    p_tm_plot.add_argument("--val-half", dest="val_half", action="store_true", default=True, help="Use FP16 on GPU for recompute")
    p_tm_plot.add_argument("--no-val-half", dest="val_half", action="store_false", help="Disable FP16 for recompute")
    p_tm_plot.add_argument("--gpu-only-val", dest="gpu_only_val", action="store_true", default=True, help="Do not fallback to CPU in recompute")
    p_tm_plot.add_argument("--allow-cpu-fallback", dest="gpu_only_val", action="store_false", help="Allow CPU fallback when GPU recompute fails")
    p_tm_plot.set_defaults(func=cmd_test_metrics_plot)

    _register_builtin_commands(_ANALYZE_COMMAND_REGISTRY)
    return parser


def _register_builtin_commands(registry: AnalyzeCommandRegistry) -> None:
    """Register subcommand runners (parsers are built in build_analyze_arg_parser)."""
    for name, runner in (
        ("scan", cmd_scan),
        ("all", cmd_all),
        ("export-table", cmd_export_table),
        ("compare", cmd_compare),
        ("leaderboard", cmd_leaderboard),
        ("pr-curves", cmd_pr_curves),
        ("inference-benchmark", cmd_inference_benchmark),
        ("inference-plot", cmd_inference_plot),
        ("test-metrics-plot", cmd_test_metrics_plot),
    ):
        registry.register_callable(
            name,
            register_parser=lambda _sub, _common, _n=name: argparse.ArgumentParser(),  # unused
            run=runner,
        )


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_analyze_arg_parser()
    args = parser.parse_args(argv)
    args.models_root_cli = getattr(args, "models_root", None)
    args.models_root = resolve_models_scan_root(args.workspace, args.models_root_cli)
    ws_prune: str | None = None
    try:
        ws_prune = resolve_workspace_root(getattr(args, "workspace", None))
    except ValueError:
        pass
    try:
        func = getattr(args, "func", None)
        if callable(func):
            func(args)
        else:
            _ANALYZE_COMMAND_REGISTRY.dispatch(args)
    finally:
        if ws_prune:
            best_effort_prune_workspace_runs_detect(ws_prune)


if __name__ == "__main__":
    main()
