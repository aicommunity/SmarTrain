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
import shlex
import shutil
import sys
import time
from datetime import datetime
from glob import glob
from io import StringIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from smartrain.workflows.analyze.compare_service import (
    build_delta_rows,
    compute_composite_score,
    generate_compare_insights,
)
from smartrain.workflows.analyze.analyze_report import write_analysis_report, write_manifest
from smartrain.core.runtime.run_artifacts import (
    canonical_run_model_path,
    materialize_canonical_run_model,
    run_test_backend_dir,
)
from smartrain.workflows.analyze.analyze_cache import (
    append_cache_entry,
    compute_fingerprint,
    data_yaml_hash,
    run_cache_root,
    weights_hash,
)
from smartrain.cli_support.cli_argparse import CliArgumentParser
from smartrain.cli_support.cli_prompts import prompt_choice, prompt_int, prompt_text
from smartrain.workflows.analyze.metrics_reader import (
    DEFAULT_MAP_COL,
    latest_test_metrics_path,
    pick_map_column,
    load_metadata,
    results_csv_path,
    read_test_metrics_by_format,
    read_test_performance_by_format_artifacts,
    read_test_system_profile_by_format_artifacts,
)
from smartrain.core.runtime.run_discovery import find_run_directories, is_run_directory, resolve_models_scan_root
from smartrain.core.runtime.ultralytics_ephemeral import best_effort_prune_workspace_runs_detect, ultralytics_sidecar_dir
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.core.training.confidence_recommendation import recommendation_file_path, read_recommendation_file
from smartrain.workflows.analyze.analyze_models import RunRecord
from smartrain.workflows.analyze.analyze_compare_session_service import (
    resolve_default_relative_output,
    resolve_session_artifacts_dir,
    resolve_session_name,
    resolve_session_root,
)
from smartrain.workflows.analyze.analyze_benchmark_service import (
    collect_split_images as _svc_collect_split_images,
    resolve_inference_csv_path as _svc_resolve_inference_csv_path,
    resolve_inference_plot_png as _svc_resolve_inference_plot_png,
    resolve_selected_run_dirs as _svc_resolve_selected_run_dirs,
    run_inference_benchmark as _svc_run_inference_benchmark,
    run_inference_plot as _svc_run_inference_plot,
)
from smartrain.workflows.analyze.analyze_pr_curves_service import (
    resolve_pr_output_png as _svc_resolve_pr_output_png,
    run_pr_curves as _svc_run_pr_curves,
)
from smartrain.workflows.analyze.analyze_test_metrics_service import (
    resolve_test_metrics_plot_png as _svc_resolve_test_metrics_plot_png,
    run_test_metrics_plot as _svc_run_test_metrics_plot,
)
from smartrain.workflows.analyze.analyze_all_selection_service import (
    prepare_all_selection as _svc_prepare_all_selection,
)
from smartrain.workflows.analyze.analyze_all_data_yaml_service import (
    resolve_all_data_yaml_context as _svc_resolve_all_data_yaml_context,
)
from smartrain.workflows.analyze.analyze_all_baseline_artifacts_service import (
    run_all_baseline_artifacts as _svc_run_all_baseline_artifacts,
)
from smartrain.workflows.analyze.analyze_all_quality_stage_service import (
    run_all_quality_stage as _svc_run_all_quality_stage,
)
from smartrain.workflows.analyze.analyze_all_speed_stage_service import (
    run_all_speed_stage as _svc_run_all_speed_stage,
)
from smartrain.services.analyze_data_yaml import collect_data_yaml_candidates_for_run
from smartrain.services.analyze_table_service import export_runs_table, scan_runs
from smartrain.services.analyze_compare_service import run_compare_workflow
from smartrain.services.analyze_artifact_builders import (
    collect_confidence_recommendation_tables,
    write_speed_quality_artifacts,
)
from smartrain.services.analyze_format_compare_service import write_format_compare_artifacts
from smartrain.services.analyze_interactive_service import run_interactive_workflow
from smartrain.services.analyze_leaderboard_service import (
    build_leaderboard_records,
    write_leaderboard_csv,
)
from smartrain.canonical.policy import emit_legacy_read_deprecation_warnings

METRIC_AGG_COLUMNS = ("mAP50-95", "mAP50", "Box-F1", "Box-P", "Box-R")


def _canonical_read_enabled() -> bool:
    emit_legacy_read_deprecation_warnings()
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
        canonical_read_enabled=_canonical_read_enabled(),
        dataset_name_resolver=lambda rd: _build_run_record_canonical(rd).dataset_name or None,
        metadata_loader=load_metadata,
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
    picked = prompt_choice("Select data.yaml", candidates, default=candidates[0], show_options=False)
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
    from ultralytics import YOLO

    best_pt = canonical_run_model_path(run_dir, ".pt")
    if not os.path.isfile(best_pt):
        materialized = materialize_canonical_run_model(run_dir, ext=".pt", move=True, normalize_metadata=True)
        if materialized is not None:
            best_pt = str(materialized)
    if not os.path.isfile(best_pt):
        raise FileNotFoundError(f"run model not found: {best_pt}")
    model = YOLO(best_pt)
    _clear_gpu_memory()
    rb, ri, rh = _resolve_run_val_profile(
        run_dir,
        default_batch=val_batch,
        default_imgsz=val_imgsz,
        default_half=val_half,
    )
    ultra_proj = ultralytics_sidecar_dir(run_dir, ".ultralytics_scratch")
    result = _run_val_memory_safe(
        model,
        data_yaml=data_yaml,
        split=split,
        val_batch=rb,
        val_imgsz=ri,
        val_half=rh,
        gpu_only=gpu_only,
        ultra_project=ultra_proj,
        ultra_name="val-recompute",
    )
    _clear_gpu_memory()
    csv_text = result.to_csv()
    rdf = pd.read_csv(StringIO(csv_text))
    if len(rdf) == 0:
        return {}
    row = rdf.iloc[0].to_dict()
    out_csv = os.path.join(run_dir, "test_metrics_recomputed.csv")
    rdf.to_csv(out_csv, index=False, encoding="utf-8")
    return row


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
    runs = find_run_directories(args.models_root)
    scan_runs(runs=runs, flat_row_for_run=_flat_row_for_run)


def cmd_export_table(args: argparse.Namespace) -> None:
    runs = find_run_directories(args.models_root)
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
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        try:
            row = _flat_row_for_run(run_dir)
        except Exception:
            continue
        rows.append(
            {
                "run_dir": run_dir,
                "run_name": os.path.basename(run_dir.rstrip(os.sep)),
                "model": row.get("model"),
                "dataset_name": row.get("dataset_name"),
                "sys_cpu_model": row.get("sys_cpu_model"),
                "sys_cpu_arch": row.get("sys_cpu_arch"),
                "sys_cpu_logical_cores": row.get("sys_cpu_logical_cores"),
                "sys_cpu_physical_cores": row.get("sys_cpu_physical_cores"),
                "sys_ram_total_gb": row.get("sys_ram_total_gb"),
                "sys_gpu_cuda_available": row.get("sys_gpu_cuda_available"),
                "sys_gpu_count": row.get("sys_gpu_count"),
                "sys_gpu_total_vram_gb": row.get("sys_gpu_total_vram_gb"),
                "sys_gpu_0_name": row.get("sys_gpu_0_name"),
                "sys_gpu_0_vram_gb": row.get("sys_gpu_0_vram_gb"),
                "sys_disk_mount_point": row.get("sys_disk_mount_point"),
                "sys_disk_fs": row.get("sys_disk_fs"),
                "sys_disk_total_gb": row.get("sys_disk_total_gb"),
                "sys_disk_free_gb": row.get("sys_disk_free_gb"),
                "sys_os": row.get("sys_os"),
                "sys_os_release": row.get("sys_os_release"),
                "sys_python_version": row.get("sys_python_version"),
                "sys_hostname": row.get("sys_hostname"),
            }
        )
    if not rows:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    return out_csv


def _write_test_system_profile_compare_csv(run_dirs: list[str], out_csv: str) -> str | None:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        try:
            flat = _flat_row_for_run(run_dir)
        except Exception:
            flat = {}
        by_fmt = read_test_system_profile_by_format_artifacts(run_dir)
        run_name = os.path.basename(run_dir.rstrip(os.sep))
        for fmt, records in by_fmt.items():
            for rec in records:
                profile = rec.get("test_system_profile") if isinstance(rec, dict) else None
                if not isinstance(profile, dict):
                    continue
                runtime = profile.get("runtime") if isinstance(profile.get("runtime"), dict) else {}
                cpu = profile.get("cpu") if isinstance(profile.get("cpu"), dict) else {}
                ram = profile.get("ram") if isinstance(profile.get("ram"), dict) else {}
                gpu = profile.get("gpu") if isinstance(profile.get("gpu"), dict) else {}
                platform = profile.get("platform") if isinstance(profile.get("platform"), dict) else {}
                devices = gpu.get("devices") if isinstance(gpu.get("devices"), list) else []
                row = {
                    "run_dir": run_dir,
                    "run_name": run_name,
                    "model": flat.get("model"),
                    "dataset_name": flat.get("dataset_name"),
                    "format": fmt,
                    "target_path": rec.get("target_path"),
                    "test_backend": runtime.get("backend"),
                    "test_provider": runtime.get("provider") or runtime.get("backend"),
                    "test_device": runtime.get("device"),
                    "sys_cpu_model": cpu.get("model"),
                    "sys_cpu_arch": cpu.get("architecture"),
                    "sys_cpu_logical_cores": cpu.get("logical_cores"),
                    "sys_cpu_physical_cores": cpu.get("physical_cores"),
                    "sys_ram_total_gb": ram.get("total_gb"),
                    "sys_gpu_cuda_available": gpu.get("cuda_available"),
                    "sys_gpu_count": len(devices),
                    "sys_gpu_total_vram_gb": gpu.get("total_vram_gb"),
                    "sys_gpu_0_name": devices[0].get("name") if len(devices) >= 1 and isinstance(devices[0], dict) else None,
                    "sys_gpu_0_vram_gb": (
                        devices[0].get("total_vram_gb") if len(devices) >= 1 and isinstance(devices[0], dict) else None
                    ),
                    "sys_os": platform.get("os"),
                    "sys_os_release": platform.get("os_release"),
                    "sys_python_version": platform.get("python_version"),
                    "sys_hostname": platform.get("hostname"),
                }
                rows.append(row)
    if not rows:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    return out_csv


def _finalize_compare_analytics_session(
    args: argparse.Namespace,
    baseline: str,
    others: list[str],
    out_csv: str,
    out_png: str,
    bar_path: str | None,
    insights_path: str | None,
) -> None:
    session_name = (getattr(args, "analytics_session", None) or "").strip()
    if not session_name:
        return
    try:
        ws = resolve_workspace_root(args.workspace)
    except ValueError:
        print(
            f"[ERROR] --analytics-session requires --workspace or {WORKSPACE_ENV_VAR}.",
            file=sys.stderr,
        )
        sys.exit(1)
    layout = WorkspaceLayout(ws)
    dest_root = os.path.join(layout.analytics, session_name)
    os.makedirs(dest_root, exist_ok=True)
    artifacts: list[dict[str, str]] = []
    for role, p in (
        ("delta_csv", out_csv),
        ("curves_png", out_png),
        ("bars_png", bar_path),
        ("insights_txt", insights_path),
    ):
        if not p:
            continue
        ap = os.path.abspath(p)
        if os.path.isfile(ap):
            bn = os.path.basename(ap)
            shutil.copy2(ap, os.path.join(dest_root, bn))
            artifacts.append({"role": role, "file": bn})
    manifest: dict[str, Any] = {
        "kind": "compare",
        "baseline": baseline,
        "others": others,
        "scan_root_at_generation": getattr(args, "models_root", None),
        "artifacts": artifacts,
    }
    sj = os.path.join(dest_root, "session.json")
    with open(sj, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] Compare session manifest: {sj}")


def _resolve_compare_png_path(
    workspace_cli: str | None,
    analytics_session: str | None,
    out_png_cli: str,
) -> str:
    return _default_relative_output(
        workspace_cli, analytics_session, "compare", "compare_curves.png", out_png_cli
    )


def _resolve_compare_artifact_path(
    workspace_cli: str | None,
    analytics_session: str | None,
    category: str,
    raw_path: str,
    default_file_name: str,
) -> str:
    return _default_relative_output(
        workspace_cli, analytics_session, category, default_file_name, raw_path
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
    if expected is None:
        return True
    return value is expected


def _build_run_record_canonical(run_dir: str) -> RunRecord:
    from smartrain.orchestrators.canonical_gateway import load_target

    payload = load_target(run_dir, source_kind="run")
    model_name: str | None = None
    dataset_name: str | None = None
    if payload.models:
        model_name = str(payload.models[0].model_id or "").strip() or None
    if payload.runs:
        dataset_name = str(payload.runs[0].dataset_ref or "").strip() or None
    metrics = _read_test_metrics_for_run(run_dir)
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


def _read_test_metrics_for_run(run_dir: str, *, format_name: str = "pt") -> dict[str, Any]:
    from smartrain.orchestrators.canonical_gateway import load_metrics

    metric_refs = load_metrics(run_dir, source_kind="run", format_name=format_name)
    if metric_refs:
        out = dict(metric_refs[0].primary_metrics or {})
        out.update(dict(metric_refs[0].secondary_metrics or {}))
        return out
    return {}


def _flat_row_canonical(run_dir: str) -> dict[str, Any]:
    rec = _build_run_record_canonical(run_dir)
    return {
        "run_dir": run_dir,
        "run_name": os.path.basename(run_dir.rstrip(os.sep)),
        "model": rec.model,
        "dataset_name": rec.dataset_name,
    }

def _flat_row_for_run(run_dir: str) -> dict[str, Any]:
    return _flat_row_canonical(run_dir)


def _filtered_run_records(args: argparse.Namespace) -> list[tuple[str, Any]]:
    runs = find_run_directories(args.models_root)
    recs: list[tuple[str, Any]] = []
    filter_dataset = getattr(args, "filter_dataset", None)
    filter_model = getattr(args, "filter_model", None)
    filter_training_ok = getattr(args, "filter_training_ok", None)
    filter_testing_ok = getattr(args, "filter_testing_ok", None)
    for run_dir in runs:
        try:
            rec = _build_run_record_canonical(run_dir)
        except Exception as e:
            print(f"[WARN] {run_dir}: failed to index run ({e})")
            continue
        if filter_dataset and rec.dataset_name != filter_dataset:
            continue
        if filter_model and rec.model != filter_model:
            continue
        if not _matches_optional_bool(rec.training_ok, filter_training_ok):
            continue
        if not _matches_optional_bool(rec.testing_ok, filter_testing_ok):
            continue
        recs.append((run_dir, rec))
    return recs


def cmd_interactive(args: argparse.Namespace) -> None:
    indexed = _filtered_run_records(args)
    if not indexed:
        print("No runs found after filters.")
        return
    run_interactive_workflow(
        args=args,
        indexed=indexed,
        session_artifacts_dir=_session_artifacts_dir,
        cmd_compare=cmd_compare,
        cmd_test_metrics_plot=cmd_test_metrics_plot,
        cmd_inference_benchmark=cmd_inference_benchmark,
        cmd_inference_plot=cmd_inference_plot,
        cmd_pr_curves=cmd_pr_curves,
        runs_with_missing_metrics=_runs_with_missing_metrics,
        auto_select_data_yaml=_auto_select_data_yaml,
        prompt_choice=prompt_choice,
    )


def cmd_leaderboard(args: argparse.Namespace) -> None:
    if not getattr(args, "quality_metric", None) and sys.stdin.isatty():
        args.quality_metric = prompt_text("Quality metric", default="mAP50-95").strip() or "mAP50-95"
    if not getattr(args, "speed_metric", None) and sys.stdin.isatty():
        args.speed_metric = prompt_text("Speed metric", default="avg_inference_fps").strip() or "avg_inference_fps"
    runs = find_run_directories(args.models_root)
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
        load_run_record=_build_run_record_canonical,
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


def _extract_pr_curve_from_metrics(metrics_obj: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Try to extract all-classes PR curve from Ultralytics metrics object."""
    sources = [metrics_obj, getattr(metrics_obj, "box", None)]
    for src in sources:
        if src is None:
            continue
        curves = getattr(src, "curves_results", None)
        if not curves:
            continue
        for item in curves:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            x = np.asarray(item[0], dtype=float)
            y = np.asarray(item[1], dtype=float)
            x_label = str(item[2]) if len(item) > 2 else ""
            y_label = str(item[3]) if len(item) > 3 else ""
            title = str(item[4]) if len(item) > 4 else ""
            marker = f"{x_label} {y_label} {title}".lower()

            # Keep only PR curve data (Recall -> Precision).
            if "recall" not in marker or "precision" not in marker:
                continue

            if y.ndim >= 2:
                # Usually shape: (num_classes, points); average across classes.
                valid_rows = ~np.all(np.isnan(y), axis=1)
                if bool(np.any(valid_rows)):
                    y = np.nanmean(y[valid_rows], axis=0)
                else:
                    continue
            if x.ndim > 1:
                x = np.ravel(x)
            if y.ndim > 1:
                y = np.ravel(y)

            n = min(len(x), len(y))
            if n == 0:
                continue
            return x[:n], y[:n]
    return None


def _extract_pr_curve_per_class_from_metrics(metrics_obj: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Return recall grid and per-class precision curves if available."""
    sources = [metrics_obj, getattr(metrics_obj, "box", None)]
    for src in sources:
        if src is None:
            continue
        curves = getattr(src, "curves_results", None)
        if not curves:
            continue
        for item in curves:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            x = np.asarray(item[0], dtype=float)
            y = np.asarray(item[1], dtype=float)
            marker = " ".join(str(v) for v in item[2:5]).lower()
            if "recall" not in marker or "precision" not in marker:
                continue
            if y.ndim < 2:
                continue
            if x.ndim > 1:
                x = np.ravel(x)
            n_points = min(len(x), int(y.shape[-1]))
            if n_points <= 0:
                continue
            y2d = y[:, :n_points] if y.shape[1] >= n_points else y[:n_points, :].T
            if y2d.shape[1] != n_points:
                continue
            return x[:n_points], y2d
    return None


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
    plan = _collect_missing_metrics_recompute_plan(
        run_dirs,
        requested_metrics,
        data_yaml=data_yaml,
        run_data_yaml_map=None,
        workspace=workspace,
        split=split,
    )
    return [str(x.get("run_dir")) for x in plan.get("recompute", []) if x.get("run_dir")]


def _collect_missing_metrics_recompute_plan(
    run_dirs: list[str],
    requested_metrics: list[str],
    *,
    data_yaml: str | None = None,
    run_data_yaml_map: dict[str, str] | None = None,
    workspace: str | None = None,
    split: str = "test",
) -> dict[str, list[dict[str, Any]]]:
    recompute: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        row = _read_test_metrics_for_run(run_dir)
        recomputed_csv = os.path.join(run_dir, "test_metrics_recomputed.csv")
        if os.path.isfile(recomputed_csv):
            try:
                rdf = pd.read_csv(recomputed_csv)
                if len(rdf) > 0:
                    row.update(rdf.iloc[0].to_dict())
            except Exception:
                pass
        if not row:
            missing_metrics = list(requested_metrics)
        else:
            missing_metrics = [m for m in requested_metrics if pd.isna(pd.to_numeric(row.get(m), errors="coerce"))]
        if missing_metrics:
            # Keep pre-check aligned with actual recompute path in cmd_test_metrics_plot:
            # per-run resolved data.yaml has priority over shared session choice.
            resolved_yaml = str((run_data_yaml_map or {}).get(run_dir) or "").strip() or _resolve_data_yaml_for_run(run_dir, workspace)[0]
            if not resolved_yaml:
                resolved_yaml = data_yaml
            best_pt = canonical_run_model_path(run_dir, ".pt")
            if not resolved_yaml:
                print(
                    "[INFO] "
                    + os.path.basename(run_dir.rstrip(os.sep))
                    + ": skip recompute prompt (no resolved data.yaml for run)."
                )
                skipped.append(
                    {
                        "run_dir": run_dir,
                        "missing_metrics": missing_metrics,
                        "reason": "no_data_yaml",
                    }
                )
                continue
            if not os.path.isfile(best_pt):
                print(
                    "[INFO] "
                    + os.path.basename(run_dir.rstrip(os.sep))
                    + ": skip recompute prompt (run model not found)."
                )
                skipped.append(
                    {
                        "run_dir": run_dir,
                        "missing_metrics": missing_metrics,
                        "reason": "missing_best_pt",
                    }
                )
                continue
            status_yaml = resolved_yaml or os.path.join(run_dir, "_missing_data_yaml_")
            st = _load_recompute_status(run_dir, status_yaml, split, requested_metrics)
            if st and isinstance(st, dict):
                unresolved = set(st.get("unresolved_metrics") or [])
                if unresolved and set(missing_metrics).issubset(unresolved):
                    # Already attempted for this fingerprint; asking again is usually noise.
                    print(
                        "[INFO] "
                        + os.path.basename(run_dir.rstrip(os.sep))
                        + ": skip recompute prompt for known unresolved metrics "
                        + f"{sorted(missing_metrics)} (fingerprint match)."
                    )
                    skipped.append(
                        {
                            "run_dir": run_dir,
                            "missing_metrics": missing_metrics,
                            "reason": "known_unresolved",
                        }
                    )
                    continue
            recompute.append(
                {
                    "run_dir": run_dir,
                    "missing_metrics": missing_metrics,
                    "data_yaml": resolved_yaml,
                }
            )
    return {"recompute": recompute, "skipped": skipped}


def _recompute_status_path(run_dir: str, fingerprint: str) -> str:
    return os.path.join(run_cache_root(run_dir), "metrics", f"recompute_status_{fingerprint}.json")


def _recompute_status_fingerprint(run_dir: str, data_yaml: str, split: str, requested_metrics: list[str]) -> str:
    return compute_fingerprint(
        {
            "tool": "analyze-v2",
            "task": "metrics_recompute_status",
            "split": split,
            "data_yaml_hash": data_yaml_hash(data_yaml),
            "weights_hash": weights_hash(run_dir),
            "requested_metrics": sorted([m.strip() for m in requested_metrics if m.strip()]),
        }
    )


def _load_recompute_status(
    run_dir: str,
    data_yaml: str,
    split: str,
    requested_metrics: list[str],
) -> dict[str, Any] | None:
    fp = _recompute_status_fingerprint(run_dir, data_yaml, split, requested_metrics)
    p = _recompute_status_path(run_dir, fp)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


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
    fp = _recompute_status_fingerprint(run_dir, data_yaml, split, requested_metrics)
    p = _recompute_status_path(run_dir, fp)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    payload = {
        "status": status,
        "resolved_metrics": sorted(set(resolved)),
        "unresolved_metrics": sorted(set(unresolved)),
        "requested_metrics": sorted(set(requested_metrics)),
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    append_cache_entry(
        run_dir,
        {
            "artifact": "metrics.recompute_status",
            "fingerprint": fp,
            "path": os.path.relpath(p, run_dir),
            "status": status,
        },
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
            rec = _build_run_record_canonical(rd)
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
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, str]] = []
    out_root = os.path.join(session_root, "artifacts", "ultralytics-test")
    for rd in run_dirs:
        run_name = os.path.basename(rd.rstrip(os.sep))
        run_code = abbreviations.get(run_name, run_name)
        preferred_test_dir = str(run_test_backend_dir(rd, "ultralytics"))
        legacy_test_dir = os.path.join(rd, "test")
        test_dir = preferred_test_dir if os.path.isdir(preferred_test_dir) else legacy_test_dir
        run_info: dict[str, Any] = {}
        machine_info: dict[str, Any] = {}
        rec = _build_run_record_canonical(rd)
        run_info = {
            "model": rec.model,
            "dataset_name": rec.dataset_name,
            "epochs": None,
            "batch_size": None,
            "train_image_size": None,
            "val_imgsz": None,
        }
        machine_info = {}
        row: dict[str, Any] = {
            "run_dir": rd,
            "run_name": run_name,
            "run_code": run_code,
            "test_dir": test_dir,
            "exists": os.path.isdir(test_dir),
            "run_info": run_info,
            "machine_info": machine_info,
            "files": [],
            "csv": {},
            "images": [],
        }
        if not os.path.isdir(test_dir):
            rows.append(row)
            continue
        safe_code = re.sub(r"[^\w.\-+]+", "_", str(run_code), flags=re.UNICODE).strip("._") or "run"
        dst_dir = os.path.join(out_root, safe_code)
        os.makedirs(dst_dir, exist_ok=True)
        csv_names = ("pr.csv", "pr_per_class.csv")
        image_patterns = (
            "PR_curve.png",
            "BoxPR_curve.png",
            "F1_curve.png",
            "BoxF1_curve.png",
            "P_curve.png",
            "BoxP_curve.png",
            "R_curve.png",
            "BoxR_curve.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "val_batch0_pred.jpg",
            "val_batch0_labels.jpg",
        )
        for name in csv_names:
            src = os.path.join(test_dir, name)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(dst_dir, name)
            try:
                shutil.copy2(src, dst)
                rel = os.path.relpath(dst, session_root)
                row["csv"][name] = rel
                artifacts.append({"role": f"ultralytics_test_{name.replace('.', '_')}", "path": rel})
            except Exception:
                pass
        for name in image_patterns:
            src = os.path.join(test_dir, name)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(dst_dir, name)
            try:
                shutil.copy2(src, dst)
                rel = os.path.relpath(dst, session_root)
                row["images"].append(rel)
                artifacts.append({"role": "ultralytics_test_image", "path": rel})
            except Exception:
                pass
        row["files"] = sorted(os.listdir(test_dir))
        rows.append(row)
    return rows, artifacts


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
        canonical_run_model_path_cb=canonical_run_model_path,
        run_cache_root_cb=run_cache_root,
        compute_fingerprint_cb=compute_fingerprint,
        data_yaml_hash_cb=data_yaml_hash,
        weights_hash_cb=weights_hash,
        clear_gpu_memory_cb=_clear_gpu_memory,
        resolve_run_val_profile_cb=_resolve_run_val_profile,
        ultralytics_sidecar_dir_cb=ultralytics_sidecar_dir,
        run_val_memory_safe_cb=_run_val_memory_safe,
        extract_pr_curve_cb=_extract_pr_curve_from_metrics,
        extract_pr_curve_per_class_cb=_extract_pr_curve_per_class_from_metrics,
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
        canonical_run_model_path_cb=canonical_run_model_path,
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
        save_recompute_status_cb=_save_recompute_status,
        resolve_data_yaml_for_run_cb=_resolve_data_yaml_for_run,
        recompute_run_test_metrics_cb=_recompute_run_test_metrics,
        compute_fingerprint_cb=compute_fingerprint,
        run_cache_root_cb=run_cache_root,
        data_yaml_hash_cb=data_yaml_hash,
        append_cache_entry_cb=append_cache_entry,
        resolve_workspace_root_cb=resolve_workspace_root,
        workspace_layout_cls=WorkspaceLayout,
    )


def cmd_all(args: argparse.Namespace) -> None:
    baseline, others, profile, interactive_mode = _svc_prepare_all_selection(
        args,
        filtered_run_records_cb=_filtered_run_records,
        prompt_int_cb=prompt_int,
        prompt_text_cb=prompt_text,
        prompt_choice_cb=prompt_choice,
    )
    report_languages, data_yaml, selected_run_dirs, run_data_yaml_map, unresolved_data_yaml_runs = (
        _svc_resolve_all_data_yaml_context(
            args=args,
            baseline=baseline,
            others=others,
            profile=profile,
            interactive_mode=interactive_mode,
            build_run_data_yaml_map_cb=_build_run_data_yaml_map,
            auto_select_data_yaml_cb=_auto_select_data_yaml,
            prompt_choice_cb=prompt_choice,
            prompt_text_cb=prompt_text,
        )
    )
    session_root = _session_root(args.workspace, args.analytics_session)
    artifacts: list[dict[str, str]] = []
    cache_events: list[dict[str, Any]] = []
    artifact_failures: list[dict[str, Any]] = []

    def _record_failure(
        *,
        stage: str,
        status: str,
        reason_code: str,
        reason_detail: str = "",
        run_dir: str | None = None,
        format_name: str | None = None,
        split: str | None = None,
    ) -> None:
        artifact_failures.append(
            {
                "stage": stage,
                "status": status,
                "reason_code": reason_code,
                "reason_detail": reason_detail,
                "run_dir": run_dir or "",
                "format": format_name or "",
                "split": split or "",
            }
        )
    selected_labels = [os.path.basename(x.rstrip(os.sep)) for x in selected_run_dirs]
    print("[INFO] Selected compare runs:")
    for idx, (run_dir, label) in enumerate(zip(selected_run_dirs, selected_labels), start=1):
        role = "baseline" if idx == 1 else "other"
        print(f"[INFO]  - {role}: {label} ({run_dir})")

    baseline_artifacts, lb_csv = _svc_run_all_baseline_artifacts(
        baseline=baseline,
        others=others,
        selected_run_dirs=selected_run_dirs,
        session_root=session_root,
        workspace=args.workspace,
        analytics_session=args.analytics_session,
        models_root=args.models_root,
        default_map_col=DEFAULT_MAP_COL,
        cmd_compare_cb=cmd_compare,
        cmd_export_table_cb=cmd_export_table,
        write_system_profile_compare_csv_cb=_write_system_profile_compare_csv,
        write_test_system_profile_compare_csv_cb=_write_test_system_profile_compare_csv,
        cmd_leaderboard_cb=cmd_leaderboard,
    )
    artifacts.extend(baseline_artifacts)

    runs_group_dir = os.path.dirname(baseline)
    quality_artifacts, metric_sources_payload, recompute_missing_metrics = _svc_run_all_quality_stage(
        args=args,
        profile=profile,
        baseline=baseline,
        others=others,
        selected_run_dirs=selected_run_dirs,
        session_root=session_root,
        runs_group_dir=runs_group_dir,
        data_yaml=data_yaml,
        run_data_yaml_map=run_data_yaml_map,
        collect_missing_metrics_recompute_plan_cb=_collect_missing_metrics_recompute_plan,
        cmd_test_metrics_plot_cb=cmd_test_metrics_plot,
    )
    artifacts.extend(quality_artifacts)

    speed_artifacts, speed_cache_events = _svc_run_all_speed_stage(
        args=args,
        profile=profile,
        baseline=baseline,
        others=others,
        selected_run_dirs=selected_run_dirs,
        session_root=session_root,
        runs_group_dir=runs_group_dir,
        run_data_yaml_map=run_data_yaml_map,
        metric_sources_payload=metric_sources_payload,
        record_failure_cb=_record_failure,
        group_runs_by_data_yaml_cb=_group_runs_by_data_yaml,
        cmd_inference_benchmark_cb=cmd_inference_benchmark,
        cmd_inference_plot_cb=cmd_inference_plot,
        write_speed_quality_artifacts_cb=_write_speed_quality_artifacts,
    )
    artifacts.extend(speed_artifacts)
    cache_events.extend(speed_cache_events)

    if profile == "full":
        pr_groups, unresolved_for_pr = _group_runs_by_data_yaml(selected_run_dirs, run_data_yaml_map)
        if unresolved_for_pr:
            print("[WARN] PR stage: skipped runs without resolved data.yaml:")
            for rd in unresolved_for_pr:
                print(f"[WARN]  - {os.path.basename(rd.rstrip(os.sep))}")
                _record_failure(
                    stage="pr",
                    status="skipped",
                    reason_code="no_data_yaml",
                    reason_detail="run excluded from PR stage due to unresolved data.yaml",
                    run_dir=rd,
                    split="test",
                )
        pr_per_class_frames: list[pd.DataFrame] = []
        pr_png_written = False
        os.makedirs(os.path.join(session_root, "artifacts", "pr"), exist_ok=True)
        for g_idx, (group_yaml, group_runs) in enumerate(sorted(pr_groups.items()), start=1):
            group_pr_dir = os.path.join(session_root, "artifacts", "pr", f"group_{g_idx}")
            os.makedirs(group_pr_dir, exist_ok=True)
            pr_png = os.path.join(group_pr_dir, "pr_all_classes.png")
            pr_ns = argparse.Namespace(
                runs_group_dir=runs_group_dir,
                selected_run_dirs=group_runs,
                data_yaml=group_yaml,
                out_png=pr_png,
                workspace=args.workspace,
                models_root=args.models_root,
                analytics_session=args.analytics_session,
                pr_per_class=True,
                reuse_run_cache=True,
                cache_stats_out=os.path.join(session_root, "artifacts", "pr", f"cache_stats_group_{g_idx}.json"),
                val_batch=int(getattr(args, "val_batch", 1)),
                val_imgsz=int(getattr(args, "val_imgsz", 640)),
                val_half=bool(getattr(args, "val_half", True)),
                gpu_only_val=bool(getattr(args, "gpu_only_val", True)),
            )
            cmd_pr_curves(pr_ns)
            if os.path.isfile(pr_png):
                artifacts.append({"role": "pr_png", "path": os.path.relpath(pr_png, session_root)})
                pr_png_written = True
            part_csv = os.path.join(group_pr_dir, "per_class", "pr_per_class.csv")
            if os.path.isfile(part_csv):
                try:
                    pr_per_class_frames.append(pd.read_csv(part_csv))
                except Exception as e:
                    _record_failure(
                        stage="pr",
                        status="failed",
                        reason_code="per_class_csv_read_failed",
                        reason_detail=str(e),
                        split="test",
                    )
            else:
                for rd in group_runs:
                    _record_failure(
                        stage="pr",
                        status="missing",
                        reason_code="pr_per_class_missing",
                        reason_detail="group PR per-class CSV was not produced",
                        run_dir=rd,
                        split="test",
                    )
            pr_cache_stats = os.path.join(session_root, "artifacts", "pr", f"cache_stats_group_{g_idx}.json")
            if os.path.isfile(pr_cache_stats):
                try:
                    cache_events.extend(json.load(open(pr_cache_stats, "r", encoding="utf-8")).get("cache", []))
                except Exception as e:
                    _record_failure(
                        stage="pr",
                        status="failed",
                        reason_code="cache_stats_read_failed",
                        reason_detail=str(e),
                        split="test",
                    )
        if pr_per_class_frames:
            pr_per_class_csv = os.path.join(session_root, "artifacts", "pr", "per_class", "pr_per_class.csv")
            os.makedirs(os.path.dirname(pr_per_class_csv), exist_ok=True)
            merged = pd.concat(pr_per_class_frames, ignore_index=True)
            merged = merged.drop_duplicates()
            present_models = set(merged.get("model", pd.Series(dtype=str)).astype(str).tolist())
            for rd in selected_run_dirs:
                run_name = os.path.basename(rd.rstrip(os.sep))
                if run_name in present_models:
                    continue
                merged = pd.concat(
                    [
                        merged,
                        pd.DataFrame(
                            [
                                {
                                    "model": run_name,
                                    "class_name": "N/A",
                                    "ap": np.nan,
                                    "status": "missing",
                                    "reason_code": "pr_per_class_missing",
                                    "run_dir": rd,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            merged.to_csv(pr_per_class_csv, index=False, encoding="utf-8")
            artifacts.append({"role": "pr_per_class_csv", "path": os.path.relpath(pr_per_class_csv, session_root)})
            # Build unified per-class PNGs across all selected runs/groups.
            # This removes ambiguity when runs are split by data.yaml groups.
            combined_dir = os.path.join(session_root, "artifacts", "pr", "per_class_combined")
            os.makedirs(combined_dir, exist_ok=True)
            if {"class_id", "class_name", "model", "recall", "precision"}.issubset(merged.columns):
                expected_models = {
                    os.path.basename(str(r).rstrip(os.sep))
                    for r in selected_run_dirs
                }
                class_groups = merged.groupby(["class_id", "class_name"], dropna=False)
                for (class_id, class_name), cls_df in class_groups:
                    cdf = cls_df.copy()
                    cdf["recall_num"] = pd.to_numeric(cdf.get("recall"), errors="coerce")
                    cdf["precision_num"] = pd.to_numeric(cdf.get("precision"), errors="coerce")
                    cdf = cdf.dropna(subset=["recall_num", "precision_num"])
                    if len(cdf) == 0:
                        continue
                    plt.figure(figsize=(9, 6))
                    present_models: set[str] = set()
                    for model_name, mdf in cdf.groupby("model"):
                        mdf = mdf.sort_values("recall_num")
                        if len(mdf) == 0:
                            continue
                        present_models.add(str(model_name))
                        plt.plot(
                            mdf["recall_num"],
                            mdf["precision_num"],
                            linewidth=1.8,
                            label=str(model_name),
                        )
                    if not present_models:
                        plt.close()
                        continue
                    plt.title(f"PR per class (all runs): {class_name} (id={class_id})")
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.grid(True, linestyle="--", alpha=0.6)
                    plt.legend(fontsize=8)
                    plt.tight_layout()
                    class_id_int = int(class_id) if str(class_id).isdigit() else -1
                    out_png = os.path.join(
                        combined_dir,
                        f"pr_class_{class_id_int}_{_safe_name(str(class_name))}_all_runs.png",
                    )
                    plt.savefig(out_png, dpi=220)
                    plt.close()
                    artifacts.append({"role": "pr_per_class_png", "path": os.path.relpath(out_png, session_root)})
                    if len(present_models) < len(expected_models):
                        _record_failure(
                            stage="pr",
                            status="failed",
                            reason_code="png_incomplete_series",
                            reason_detail=(
                                f"pr_per_class_combined class={class_name} "
                                f"models={sorted(present_models)} expected={sorted(expected_models)}"
                            ),
                            split="test",
                        )
            try:
                expected = {
                    os.path.basename(str(r).rstrip(os.sep))
                    for r in selected_run_dirs
                }
                actual = set(merged.get("model", pd.Series(dtype=str)).astype(str).tolist())
                if len(actual) < len(expected):
                    _record_failure(
                        stage="pr",
                        status="failed",
                        reason_code="png_incomplete_series",
                        reason_detail=f"pr_per_class models={sorted(actual)} expected={sorted(expected)}",
                        split="test",
                    )
            except Exception as e:
                _record_failure(
                    stage="pr",
                    status="failed",
                    reason_code="pr_series_validation_failed",
                    reason_detail=str(e),
                    split="test",
                )
        if not pr_png_written:
            print("[WARN] PR stage completed without PR plot artifacts.")
            _record_failure(
                stage="pr",
                status="missing",
                reason_code="pr_plot_missing",
                reason_detail="no PR plot artifacts were produced",
                split="test",
            )
        for p in sorted(glob(os.path.join(session_root, "artifacts", "pr", "**", "per_class", "*.png"), recursive=True)):
            artifacts.append({"role": "pr_per_class_png", "path": os.path.relpath(p, session_root)})

    abbreviations = _build_abbreviations_for_report([baseline] + others)
    ultralytics_test_rows, ultralytics_test_artifacts = _collect_ultralytics_test_artifacts(
        session_root,
        [baseline] + others,
        abbreviations,
    )
    artifacts.extend(ultralytics_test_artifacts)
    format_compare = _write_format_compare_artifacts(session_root, [baseline] + others)
    if format_compare and format_compare.get("csv"):
        artifacts.append({"role": "format_compare_csv", "path": str(format_compare["csv"])})
        perf_rel = str(format_compare.get("perf_test_csv") or "")
        if perf_rel:
            perf_abs = os.path.join(session_root, perf_rel)
            if os.path.isfile(perf_abs):
                try:
                    perf_df = pd.read_csv(perf_abs)
                    if "performance_status" in perf_df.columns:
                        bad = perf_df[perf_df["performance_status"].astype(str).str.lower() != "ok"].copy()
                        for _, row in bad.iterrows():
                            _record_failure(
                                stage="format_performance",
                                status="missing",
                                reason_code=str(row.get("performance_reason") or "performance_not_collected"),
                                reason_detail="format performance row is incomplete",
                                run_dir=str(row.get("run_dir") or ""),
                                format_name=str(row.get("format") or ""),
                                split=str(row.get("split") or "test"),
                            )
                except Exception as e:
                    _record_failure(
                        stage="format_performance",
                        status="failed",
                        reason_code="format_perf_read_failed",
                        reason_detail=str(e),
                        split="test",
                    )
    conf_tables = _collect_confidence_recommendation_tables(
        [baseline] + others,
        os.path.join(session_root, "artifacts", "confidence"),
    )
    for objective in ("A", "B", "C"):
        p = conf_tables.get(objective)
        if p and os.path.isfile(p):
            artifacts.append(
                {
                    "role": f"confidence_recommendations_{objective.lower()}_csv",
                    "path": os.path.relpath(p, session_root),
                }
            )

    manifest = {
        "session_name": os.path.basename(session_root),
        "profile": profile,
        "baseline": baseline,
        "others": others,
        "artifacts": artifacts,
        "images": [a["path"] for a in artifacts if a["path"].endswith(".png")],
        "tables": [a["path"] for a in artifacts if a["path"].endswith(".csv")],
        "artifact_scope": {
            "single_run": ["test_metrics_recomputed.csv", "pr aggregate/per_class", "inference benchmark profile"],
            "cross_run": ["compare", "leaderboard", "speed_quality", "reports", "session manifest"],
        },
        "sections": [
            "executive_summary",
            "comparison_context",
            "quality_analysis",
            "speed_analysis",
            "format_compare",
            "per_class_analysis",
            "conclusion",
        ],
        "abbreviations": abbreviations,
        "run_data_yaml_map": run_data_yaml_map,
        "runs_with_unresolved_data_yaml": unresolved_data_yaml_runs,
    }
    if ultralytics_test_rows:
        manifest["ultralytics_test"] = ultralytics_test_rows
    if conf_tables:
        manifest["confidence_recommendations"] = {
            key: os.path.relpath(path, session_root) for key, path in conf_tables.items()
        }
    if metric_sources_payload is not None:
        manifest["metric_sources"] = metric_sources_payload
    else:
        _record_failure(
            stage="metrics",
            status="missing",
            reason_code="metric_sources_missing",
            reason_detail="metric_sources.json missing or unreadable",
            split="test",
        )
    if artifact_failures:
        manifest["artifact_failures"] = artifact_failures
        by_reason: dict[str, int] = {}
        for item in artifact_failures:
            reason = str(item.get("reason_code") or "unknown")
            by_reason[reason] = by_reason.get(reason, 0) + 1
        manifest["artifact_failures_summary"] = {
            "total": len(artifact_failures),
            "by_reason_code": by_reason,
        }
    if cache_events:
        manifest["cache"] = {
            "events": cache_events,
            "hits": sum(1 for e in cache_events if e.get("status") == "hit"),
            "misses": sum(1 for e in cache_events if e.get("status") == "miss"),
        }
    if profile == "full":
        manifest["pr_per_class"] = {
            "csv": "artifacts/pr/per_class/pr_per_class.csv",
            "dir": "artifacts/pr/per_class",
        }
    if profile in ("speed", "full"):
        manifest["speed_quality"] = {
            "csv": "artifacts/speed_quality/speed_quality.csv",
            "png": "artifacts/speed_quality/speed_vs_map.png",
            "scatter_x": str(getattr(args, "scatter_x", "avg_inference_ms_per_frame")),
            "scatter_y": str(getattr(args, "scatter_y", "mAP50-95")),
        }
    if format_compare:
        manifest["format_comparison"] = format_compare
        for key in ("test_csv", "val_csv", "pt_uni_csv", "eval_csv", "csv"):
            rel = str(format_compare.get(key) or "")
            if rel and rel not in manifest["tables"]:
                manifest["tables"].append(rel)
    manifest_path = os.path.join(session_root, "session.json")
    write_manifest(manifest_path, manifest)
    strict_diag = bool(getattr(args, "strict_diagnostics", False))
    if strict_diag:
        critical_missing = []
        if profile in ("quality", "full") and "metric_sources" not in manifest:
            critical_missing.append("metric_sources")
        if profile == "full":
            pr_meta = manifest.get("pr_per_class") if isinstance(manifest.get("pr_per_class"), dict) else {}
            pr_csv_rel = str((pr_meta or {}).get("csv") or "")
            if not pr_csv_rel:
                critical_missing.append("pr_per_class_csv")
            elif not os.path.isfile(os.path.join(session_root, pr_csv_rel)):
                critical_missing.append("pr_per_class_csv")
        if critical_missing:
            print(
                "[ERROR] Strict diagnostics failed: missing critical artifacts: "
                + ", ".join(critical_missing),
                file=sys.stderr,
            )
            sys.exit(1)
    report_files = write_analysis_report(
        session_root,
        manifest,
        no_pdf=bool(args.no_pdf),
        no_odt=bool(args.no_odt),
        languages=report_languages,
    )
    print(f"[OK] Analyze session: {session_root}")
    print(f"[OK] Manifest: {manifest_path}")
    for key, path in report_files.items():
        print(f"[OK] Report {key}: {path}")
    replay_parts = [
        "smartrain", "analyze", "all",
        "--baseline", baseline,
        "--profile", profile,
        "--report-languages", ",".join(report_languages),
        "--scatter-x", str(getattr(args, "scatter_x", "avg_inference_ms_per_frame")),
        "--scatter-y", str(getattr(args, "scatter_y", "mAP50-95")),
        "--val-batch", str(int(getattr(args, "val_batch", 1))),
        "--val-imgsz", str(int(getattr(args, "val_imgsz", 640))),
        "--recompute-missing-metrics", ("yes" if recompute_missing_metrics else "no"),
    ]
    workspace_val = getattr(args, "workspace", None)
    if workspace_val is not None and str(workspace_val).strip().lower() not in {"", "none"}:
        replay_parts.extend(["--workspace", str(workspace_val)])
    for item in others:
        replay_parts.extend(["--others", item])
    if data_yaml:
        replay_parts.extend(["--data-yaml", data_yaml])
    if bool(getattr(args, "no_pdf", False)):
        replay_parts.append("--no-pdf")
    if bool(getattr(args, "no_odt", False)):
        replay_parts.append("--no-odt")
    replay_parts.append("--val-half" if bool(getattr(args, "val_half", True)) else "--no-val-half")
    replay_parts.append("--gpu-only-val" if bool(getattr(args, "gpu_only_val", True)) else "--allow-cpu-fallback")
    print("[INFO] Command for non-interactive retry:")
    print(" ".join(shlex.quote(p) for p in replay_parts if p))

def build_analyze_arg_parser() -> argparse.ArgumentParser:
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
    p_all.add_argument("--val-batch", type=int, default=1, help="Validation batch size for GPU memory-safe val()")
    p_all.add_argument("--val-imgsz", type=int, default=640, help="Validation image size for GPU memory-safe val()")
    p_all.add_argument("--val-half", dest="val_half", action="store_true", default=True, help="Use FP16 for validation on GPU")
    p_all.add_argument("--no-val-half", dest="val_half", action="store_false", help="Disable FP16 for validation")
    p_all.add_argument("--gpu-only-val", dest="gpu_only_val", action="store_true", default=True, help="Do not fallback to CPU in val()")
    p_all.add_argument("--allow-cpu-fallback", dest="gpu_only_val", action="store_false", help="Allow CPU fallback when GPU val() fails")
    p_all.set_defaults(func=cmd_all)

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

    return parser


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_analyze_arg_parser()
    args = parser.parse_args(argv)
    args.models_root = resolve_models_scan_root(args.workspace, args.models_root)
    ws_prune: str | None = None
    try:
        ws_prune = resolve_workspace_root(getattr(args, "workspace", None))
    except ValueError:
        pass
    try:
        args.func(args)
    finally:
        if ws_prune:
            best_effort_prune_workspace_runs_detect(ws_prune)


if __name__ == "__main__":
    main()
