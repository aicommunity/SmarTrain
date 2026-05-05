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

from smartrain.compare_service import (
    build_delta_rows,
    compute_composite_score,
    generate_compare_insights,
)
from smartrain.analyze_report import write_analysis_report, write_manifest
from smartrain.run_artifacts import (
    canonical_run_model_path,
    materialize_canonical_run_model,
    resolve_run_model_with_legacy_fallback,
    run_test_backend_dir,
)
from smartrain.analyze_cache import (
    append_cache_entry,
    compute_fingerprint,
    data_yaml_hash,
    run_cache_root,
    weights_hash,
)
from smartrain.cli_argparse import CliArgumentParser
from smartrain.cli_prompts import prompt_choice, prompt_int, prompt_text
from smartrain.metrics_reader import (
    DEFAULT_MAP_COL,
    build_run_record,
    flatten_metadata,
    latest_test_metrics_path,
    load_metadata,
    pick_map_column,
    read_test_metrics_row,
    results_csv_path,
    read_test_metrics_by_format,
    read_metrics_by_format_for_split,
    read_metrics_by_format_for_split_artifacts,
    read_test_performance_by_format_artifacts,
    read_test_system_profile_by_format_artifacts,
)
from smartrain.model_test_service import load_test_artifacts_manifest
from smartrain.run_discovery import find_run_directories, is_run_directory, resolve_models_scan_root
from smartrain.ultralytics_ephemeral import best_effort_prune_workspace_runs_detect, ultralytics_sidecar_dir
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.confidence_recommendation import recommendation_file_path, read_recommendation_file
from smartrain.analyze_models import RunRecord
from smartrain.services.analyze_artifacts import (
    default_relative_output,
    session_artifacts_dir,
    session_name,
    session_root,
)
from smartrain.services.analyze_data_yaml import collect_data_yaml_candidates_for_run
from smartrain.services.analyze_table_service import export_runs_table, scan_runs

METRIC_AGG_COLUMNS = ("mAP50-95", "mAP50", "Box-F1", "Box-P", "Box-R")


def _canonical_read_enabled() -> bool:
    legacy_fallback_allowed = str(os.getenv("SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK", "")).strip() == "1"
    return not (legacy_fallback_allowed and str(os.getenv("SMARTTRAIN_CANONICAL_READ", "")).strip() == "0")


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
    return session_name(raw)


def _session_root(workspace_cli: str | None, analytics_session: str | None) -> str:
    return session_root(workspace_cli, analytics_session)


def _session_artifacts_dir(workspace_cli: str | None, analytics_session: str | None, category: str) -> str:
    return session_artifacts_dir(workspace_cli, analytics_session, category)


def _default_relative_output(
    workspace_cli: str | None,
    analytics_session: str | None,
    category: str,
    file_name: str,
    raw: str | None,
) -> str:
    return default_relative_output(workspace_cli, analytics_session, category, file_name, raw)


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

    other_rows = [_read_test_metrics_for_run(other) for other in others]
    delta_rows = build_delta_rows(baseline, base_metrics, others, other_rows)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    pd.DataFrame(delta_rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Test metrics comparison: {out_csv}")
    insight_lines = generate_compare_insights(baseline, others, delta_rows)
    os.makedirs(os.path.dirname(out_insights) or ".", exist_ok=True)
    with open(out_insights, "w", encoding="utf-8") as f:
        f.write("\n".join(insight_lines).rstrip() + "\n")
    print(f"[OK] Insights: {out_insights}")

    metric_col = args.metric_column
    plt.figure(figsize=(12, 7))
    plotted = False
    labels: list[str] = []
    for i, rd in enumerate(all_runs):
        rc = results_csv_path(rd)
        label = os.path.basename(rd.rstrip(os.sep))[:40]
        labels.append(label)
        if not rc:
            print(f"[WARN] Missing train/results.csv: {rd}")
            continue
        try:
            df = pd.read_csv(rc)
            df.columns = [str(c).strip() for c in df.columns]
            mcol = metric_col if metric_col in df.columns else pick_map_column(df)
            if mcol is None or "epoch" not in df.columns:
                print(f"[WARN] Missing epoch / mAP columns in {rc}")
                continue
            plt.plot(df["epoch"], df[mcol], label=label, linewidth=2)
            plotted = True
        except Exception as e:
            print(f"[WARN] {rc}: {e}")

    if plotted:
        plt.title("Metrics Comparison Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric_col or DEFAULT_MAP_COL)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Model", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[OK] Plot: {out_png}")
    else:
        plt.close()

    # Bar chart using last-epoch mAP from results.csv
    last_vals: list[float] = []
    last_labs: list[str] = []
    for rd, lab in zip(all_runs, labels):
        rc = results_csv_path(rd)
        if not rc:
            continue
        try:
            df = pd.read_csv(rc)
            df.columns = [str(c).strip() for c in df.columns]
            mcol = metric_col if metric_col in df.columns else pick_map_column(df)
            if mcol and len(df) > 0:
                v = df[mcol].iloc[-1]
                if pd.notna(v):
                    last_vals.append(float(v))
                    last_labs.append(lab)
        except Exception:
            pass
    if len(last_vals) >= 2:
        plt.figure(figsize=(10, 5))
        x = range(len(last_labs))
        plt.bar(x, last_vals, tick_label=last_labs)
        plt.ylabel(metric_col or DEFAULT_MAP_COL)
        plt.title("Last Epoch Comparison")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        bar_path = re.sub(r"\.png$", "_bars.png", out_png)
        plt.savefig(bar_path, dpi=200)
        plt.close()
        print(f"[OK] Bar chart: {bar_path}")

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
    if _canonical_read_enabled():
        from smartrain.orchestrators.canonical_gateway import load_metrics

        metric_refs = load_metrics(run_dir, source_kind="run", format_name=format_name)
        if metric_refs:
            out = dict(metric_refs[0].primary_metrics or {})
            out.update(dict(metric_refs[0].secondary_metrics or {}))
            return out
        return {}
    return read_test_metrics_row(run_dir, format_name) or {}


def _flat_row_canonical(run_dir: str) -> dict[str, Any]:
    rec = _build_run_record_canonical(run_dir)
    return {
        "run_dir": run_dir,
        "run_name": os.path.basename(run_dir.rstrip(os.sep)),
        "model": rec.model,
        "dataset_name": rec.dataset_name,
    }


def _flat_row_legacy(run_dir: str) -> dict[str, Any]:
    md = load_metadata(run_dir)
    return flatten_metadata(md, run_dir)


def _flat_row_for_run(run_dir: str) -> dict[str, Any]:
    if _canonical_read_enabled():
        return _flat_row_canonical(run_dir)
    return _flat_row_legacy(run_dir)


def _filtered_run_records(args: argparse.Namespace) -> list[tuple[str, Any]]:
    runs = find_run_directories(args.models_root)
    recs: list[tuple[str, Any]] = []
    use_canonical = _canonical_read_enabled()
    filter_dataset = getattr(args, "filter_dataset", None)
    filter_model = getattr(args, "filter_model", None)
    filter_training_ok = getattr(args, "filter_training_ok", None)
    filter_testing_ok = getattr(args, "filter_testing_ok", None)
    for run_dir in runs:
        try:
            rec = _build_run_record_canonical(run_dir) if use_canonical else build_run_record(run_dir)
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
    print(f"{'#':>4}  {'model':<14}  {'dataset':<24}  {'mAP50-95':>9}  {'Box-F1':>9}  {'run_dir'}")
    print("-" * 140)
    for i, (rd, rec) in enumerate(indexed, start=1):
        q = rec.test_metrics.get("mAP50-95")
        f1 = rec.test_metrics.get("Box-F1")
        q_str = f"{float(q):.4f}" if q is not None and pd.notna(q) else "-"
        f1_str = f"{float(f1):.4f}" if f1 is not None and pd.notna(f1) else "-"
        print(
            f"{i:4d}  {str(rec.model or '?')[:14]:<14}  {str(rec.dataset_name or '?')[:24]:<24}  "
            f"{q_str:>9}  {f1_str:>9}  {rd}"
        )
    try:
        bi = int(input("Baseline run number: ").strip())
        oi = input("Other run numbers (comma-separated): ").strip()
        idxs = [int(x.strip()) for x in oi.split(",") if x.strip()]
    except ValueError:
        print("Invalid input.")
        sys.exit(1)
    if bi < 1 or bi > len(indexed):
        sys.exit(1)
    baseline = indexed[bi - 1][0]
    others: list[str] = []
    for j in idxs:
        if 1 <= j <= len(indexed) and indexed[j - 1][0] != baseline:
            others.append(indexed[j - 1][0])
    if not others:
        print("No runs selected for comparison.")
        sys.exit(1)

    if args.output_dir:
        out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    else:
        out_dir = _session_artifacts_dir(args.workspace, args.analytics_session, "compare")
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(baseline.rstrip(os.sep))[:30]
    out_csv = os.path.join(out_dir, f"compare_{base_name}.csv")
    out_png = os.path.join(out_dir, f"compare_{base_name}.png")
    out_insights = os.path.join(out_dir, f"compare_{base_name}_insights.txt")
    ns = argparse.Namespace(
        baseline=baseline,
        others=others,
        out_csv=out_csv,
        out_png=out_png,
        out_insights=out_insights,
        metric_column=args.metric_column,
        workspace=args.workspace,
        analytics_session=args.analytics_session,
        models_root=args.models_root,
    )
    cmd_compare(ns)

    preset = args.preset
    selected_data_yaml: str | None = args.data_yaml
    if preset in ("quality", "full"):
        metric_list = [m.strip() for m in (args.quality_metrics or "").split(",") if m.strip()]
        if metric_list:
            runs_group_dir = os.path.dirname(baseline)
            recompute_missing_metrics = bool(getattr(args, "recompute_missing_metrics", False))
            if sys.stdin.isatty():
                missing_runs = _runs_with_missing_metrics(
                    [baseline] + others,
                    metric_list,
                    data_yaml=selected_data_yaml,
                    split=str(getattr(args, "recompute_split", "test") or "test"),
                )
                if missing_runs:
                    choice = prompt_choice(
                        "Found missing metrics. Recompute from run model + data.yaml now?",
                        ["yes", "no"],
                        default="yes",
                        show_options=False,
                    )
                    recompute_missing_metrics = choice == "yes"
            tm_ns = argparse.Namespace(
                runs_group_dir=runs_group_dir,
                metrics=metric_list,
                out_dir=out_dir,
                workspace=args.workspace,
                recompute_missing_metrics=recompute_missing_metrics,
                recompute_split=getattr(args, "recompute_split", "test"),
            )
            cmd_test_metrics_plot(tm_ns)

    if preset in ("speed", "full"):
        data_yaml = selected_data_yaml
        if not data_yaml:
            data_yaml = _auto_select_data_yaml(
                baseline,
                others,
                args.workspace,
                preferred_split=str(getattr(args, "benchmark_split", "test") or "test"),
            )
            selected_data_yaml = data_yaml
        if not data_yaml:
            print("[WARN] Preset speed/full selected but --data-yaml is missing; speed benchmark skipped.")
        else:
            runs_group_dir = os.path.dirname(baseline)
            b_ns = argparse.Namespace(
                runs_group_dir=runs_group_dir,
                data_yaml=data_yaml,
                split=args.benchmark_split,
                frames=args.benchmark_frames,
                device=args.benchmark_device,
                half=args.benchmark_half,
                out_csv=os.path.join(out_dir, f"inference_{os.path.basename(runs_group_dir)}.csv"),
                workspace=args.workspace,
            )
            cmd_inference_benchmark(b_ns)
            p_ns = argparse.Namespace(
                csv=b_ns.out_csv,
                metric=args.speed_metric,
                out_png=os.path.join(out_dir, f"inference_{os.path.basename(runs_group_dir)}_bars.png"),
                workspace=args.workspace,
            )
            cmd_inference_plot(p_ns)

    pr_data_yaml = selected_data_yaml
    if preset == "full" and pr_data_yaml:
        runs_group_dir = os.path.dirname(baseline)
        pr_ns = argparse.Namespace(
            runs_group_dir=runs_group_dir,
            data_yaml=pr_data_yaml,
            out_png=os.path.join(out_dir, f"pr_all_classes_{os.path.basename(runs_group_dir)}.png"),
            workspace=args.workspace,
        )
        cmd_pr_curves(pr_ns)


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
    records = []
    use_canonical = _canonical_read_enabled()

    def _resolve_speed_metric_from_performance(run_dir: str, metric_name: str) -> float | None:
        metric = str(metric_name or "").strip().lower()
        if not metric:
            return None
        perf_by_fmt = read_test_performance_by_format_artifacts(run_dir)
        candidates: list[float] = []
        for rows in perf_by_fmt.values():
            for row in rows:
                perf = row.get("performance") if isinstance(row, dict) else None
                if not isinstance(perf, dict):
                    continue
                value: Any = None
                if metric in {"avg_inference_fps", "throughput_img_s"}:
                    value = perf.get("throughput_img_s")
                elif metric in {"avg_inference_ms_per_frame", "latency_p50_ms"}:
                    latency_ms = perf.get("latency_ms")
                    if isinstance(latency_ms, dict):
                        steady = latency_ms.get("steady")
                        all_stats = latency_ms.get("all")
                        if isinstance(steady, dict) and steady.get("p50") is not None:
                            value = steady.get("p50")
                        elif isinstance(all_stats, dict):
                            value = all_stats.get("p50")
                elif metric == "latency_p95_ms":
                    latency_ms = perf.get("latency_ms")
                    if isinstance(latency_ms, dict):
                        steady = latency_ms.get("steady")
                        all_stats = latency_ms.get("all")
                        if isinstance(steady, dict) and steady.get("p95") is not None:
                            value = steady.get("p95")
                        elif isinstance(all_stats, dict):
                            value = all_stats.get("p95")
                try:
                    if value is None:
                        continue
                    fv = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fv):
                    candidates.append(fv)
        if not candidates:
            return None
        # Prefer best attainable speed for run-level leaderboard.
        if "fps" in metric or "throughput" in metric:
            return float(max(candidates))
        return float(min(candidates))

    for run_dir in runs:
        try:
            rec = _build_run_record_canonical(run_dir) if use_canonical else build_run_record(run_dir)
        except Exception as e:
            print(f"[WARN] {run_dir}: failed to load run ({e})")
            continue
        speed_value = rec.test_metrics.get(args.speed_metric)
        if speed_value is None or (isinstance(speed_value, float) and pd.isna(speed_value)):
            fallback_speed = _resolve_speed_metric_from_performance(run_dir, str(args.speed_metric or ""))
            if fallback_speed is not None:
                rec.test_metrics[args.speed_metric] = fallback_speed
        score = compute_composite_score(
            rec,
            weight_quality=args.weight_quality,
            weight_speed=args.weight_speed,
            weight_stability=args.weight_stability,
            quality_metric=args.quality_metric,
            speed_metric=args.speed_metric,
        )
        records.append(
            {
                "run_dir": rec.run_dir,
                "model": rec.model,
                "dataset_name": rec.dataset_name,
                "training_ok": rec.training_ok,
                "testing_ok": rec.testing_ok,
                "quality_metric": rec.test_metrics.get(args.quality_metric),
                "speed_metric": rec.test_metrics.get(args.speed_metric),
                "composite_score": score,
            }
        )
    if not records:
        print("[ERROR] No runs for leaderboard.", file=sys.stderr)
        sys.exit(1)
    df = pd.DataFrame(records)
    df = df.dropna(subset=["composite_score"]).sort_values("composite_score", ascending=False)
    if len(df) == 0:
        print("[ERROR] No runs with enough metrics for leaderboard.", file=sys.stderr)
        sys.exit(1)
    out_csv = _default_relative_output(
        args.workspace, args.analytics_session, "leaderboard", "leaderboard.csv", args.out_csv
    )
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Leaderboard CSV: {out_csv}")


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
    rows_by_objective: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    for run_dir in run_dirs:
        model_name: str | None = None
        dataset_name: str | None = None
        try:
            flat = _flat_row_for_run(run_dir)
            model_name = flat.get("model")
            dataset_name = flat.get("dataset_name")
        except Exception:
            model_name = None
            dataset_name = None
        model_name = model_name or os.path.basename(run_dir.rstrip(os.sep))
        dataset_name = dataset_name or os.path.basename(os.path.dirname(run_dir.rstrip(os.sep)))

        for split in ("val", "test"):
            payload = read_recommendation_file(recommendation_file_path(run_dir, split))
            if not isinstance(payload, dict):
                continue
            objectives = payload.get("objectives")
            if not isinstance(objectives, dict):
                continue
            for objective in ("A", "B", "C"):
                item = objectives.get(objective)
                if not isinstance(item, dict):
                    continue
                beta = item.get("beta")
                global_row = item.get("global")
                if isinstance(global_row, dict):
                    rows_by_objective[objective].append(
                        {
                            "run_dir": run_dir,
                            "run_name": os.path.basename(run_dir.rstrip(os.sep)),
                            "model": model_name,
                            "dataset": dataset_name,
                            "split": split,
                            "objective": objective,
                            "beta": beta,
                            "level": "global",
                            "class_id": -1,
                            "class_name": "all",
                            "recommended_conf": global_row.get("threshold"),
                            "target_metric": global_row.get("metric_value"),
                            "precision": global_row.get("precision"),
                            "recall": global_row.get("recall"),
                            "f1": global_row.get("f1"),
                            "support_instances": None,
                            "status": global_row.get("status") or payload.get("status"),
                            "reason": global_row.get("reason") or payload.get("reason"),
                        }
                    )
                per_class = item.get("per_class")
                if isinstance(per_class, list):
                    for row in per_class:
                        if not isinstance(row, dict):
                            continue
                        rows_by_objective[objective].append(
                            {
                                "run_dir": run_dir,
                                "run_name": os.path.basename(run_dir.rstrip(os.sep)),
                                "model": model_name,
                                "dataset": dataset_name,
                                "split": split,
                                "objective": objective,
                                "beta": beta,
                                "level": "class",
                                "class_id": row.get("class_id"),
                                "class_name": row.get("class_name"),
                                "recommended_conf": row.get("threshold"),
                                "target_metric": row.get("metric_value"),
                                "precision": row.get("precision"),
                                "recall": row.get("recall"),
                                "f1": row.get("f1"),
                                "support_instances": row.get("support_instances"),
                                "status": row.get("status") or payload.get("status"),
                                "reason": row.get("reason") or payload.get("reason"),
                            }
                        )

    out: dict[str, str] = {}
    os.makedirs(out_dir, exist_ok=True)
    sort_cols = ["run_name", "split", "level", "class_id"]
    for objective in ("A", "B", "C"):
        rows = rows_by_objective.get(objective) or []
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if set(sort_cols).issubset(df.columns):
            df = df.sort_values(sort_cols, ascending=[True, True, True, True])
        out_path = os.path.join(out_dir, f"confidence_recommendations_{objective}.csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        out[objective] = out_path
    return out


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
    if not os.path.isfile(inference_csv):
        return None
    df = pd.read_csv(inference_csv)
    if len(df) == 0:
        return None
    if "run_dir" not in df.columns:
        return None
    source_map: dict[str, dict[str, str]] = {}
    if isinstance(metric_sources_payload, dict):
        source_map = metric_sources_payload.get("sources") or {}
    rows: list[dict[str, Any]] = []
    run_data_yaml_map = run_data_yaml_map or {}
    df_with_name = df.copy()
    if "run_name" not in df_with_name.columns:
        if "run_dir" in df_with_name.columns:
            df_with_name["run_name"] = df_with_name["run_dir"].astype(str).map(
                lambda p: os.path.basename(str(p).rstrip(os.sep))
            )
        else:
            df_with_name["run_name"] = ""
    for run_dir in requested_runs:
        run_name = os.path.basename(run_dir.rstrip(os.sep))
        sub = df_with_name[(df_with_name["run_dir"] == run_dir) | (df_with_name["run_name"] == run_name)].copy()
        if len(sub) == 0:
            continue
        status_score = sub.get("benchmark_status", pd.Series(["ok"] * len(sub))).astype(str).map(
            lambda s: 0 if s == "ok" else 1
        )
        val_score = pd.to_numeric(sub.get(scatter_x), errors="coerce").isna().astype(int)
        sub = sub.assign(_status_score=status_score, _val_score=val_score).sort_values(
            ["_status_score", "_val_score"], ascending=[True, True]
        )
        rec = sub.iloc[0].to_dict()
        base_metrics = _read_test_metrics_for_run(run_dir)
        recomputed_csv = os.path.join(run_dir, "test_metrics_recomputed.csv")
        if os.path.isfile(recomputed_csv):
            try:
                rdf = pd.read_csv(recomputed_csv)
                if len(rdf) > 0:
                    base_metrics.update(rdf.iloc[0].to_dict())
            except Exception:
                pass
        quality = base_metrics.get(scatter_y)
        q_num = pd.to_numeric(quality, errors="coerce")
        s_num = pd.to_numeric(rec.get(scatter_x), errors="coerce")
        if pd.isna(q_num) or pd.isna(s_num):
            continue
        q_src = (source_map.get(run_dir) or {}).get(scatter_y, "original")
        rows.append(
            {
                "run_dir": run_dir,
                "model": rec.get("model") or os.path.basename(run_dir.rstrip(os.sep)),
                "scatter_x_metric": scatter_x,
                "scatter_y_metric": scatter_y,
                "scatter_x_value": float(s_num),
                "scatter_y_value": float(q_num),
                "quality_source": q_src,
                "dataset_yaml_used": run_data_yaml_map.get(run_dir, ""),
            }
        )
    if len(rows) < 2:
        return None
    out_dir = os.path.join(session_root, "artifacts", "speed_quality")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "speed_quality.csv")
    png_path = os.path.join(out_dir, "speed_vs_map.png")
    out_df = pd.DataFrame(rows).sort_values("scatter_x_value", ascending=True)
    out_df.to_csv(csv_path, index=False, encoding="utf-8")
    plt.figure(figsize=(9, 6))
    plt.scatter(out_df["scatter_x_value"], out_df["scatter_y_value"], alpha=0.9)
    for _, row in out_df.iterrows():
        plt.text(float(row["scatter_x_value"]), float(row["scatter_y_value"]), str(row["model"]), fontsize=8)
    plt.xlabel(scatter_x)
    plt.ylabel(scatter_y)
    plt.title("Speed vs Quality")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(png_path, dpi=220)
    plt.close()
    return {
        "csv": os.path.relpath(csv_path, session_root),
        "png": os.path.relpath(png_path, session_root),
    }


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
    use_canonical = _canonical_read_enabled()
    for idx, rd in enumerate(run_dirs, start=1):
        run_name = os.path.basename(rd.rstrip(os.sep))
        if len(run_name) > 22:
            out[run_name] = f"R{idx}"
        try:
            rec = _build_run_record_canonical(rd) if use_canonical else build_run_record(rd)
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
    use_canonical = _canonical_read_enabled()
    for rd in run_dirs:
        run_name = os.path.basename(rd.rstrip(os.sep))
        run_code = abbreviations.get(run_name, run_name)
        preferred_test_dir = str(run_test_backend_dir(rd, "ultralytics"))
        legacy_test_dir = os.path.join(rd, "test")
        test_dir = preferred_test_dir if os.path.isdir(preferred_test_dir) else legacy_test_dir
        run_info: dict[str, Any] = {}
        machine_info: dict[str, Any] = {}
        if use_canonical:
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
        else:
            flat: dict[str, Any] = {}
            try:
                flat = _flat_row_legacy(rd)
                run_info = {
                    "model": flat.get("model"),
                    "dataset_name": flat.get("dataset_name"),
                    "epochs": flat.get("epochs"),
                    "batch_size": flat.get("batch_size"),
                    "train_image_size": flat.get("train_image_size"),
                    "val_imgsz": flat.get("val_imgsz"),
                }
            except Exception:
                run_info = {}
            try:
                machine_info = {
                    "sys_cpu_model": flat.get("sys_cpu_model"),
                    "sys_cpu_logical_cores": flat.get("sys_cpu_logical_cores"),
                    "sys_ram_total_gb": flat.get("sys_ram_total_gb"),
                    "sys_gpu_0_name": flat.get("sys_gpu_0_name"),
                    "sys_gpu_0_vram_gb": flat.get("sys_gpu_0_vram_gb"),
                    "sys_os": flat.get("sys_os"),
                    "sys_os_release": flat.get("sys_os_release"),
                }
            except Exception:
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
    backend_fallback = {
        "pt": "ultralytics",
        "pt_uni": "unified_pt",
        "onnx": "onnxruntime",
        "engine": "tensorrt",
        "trt": "tensorrt",
    }
    ext_by_format = {
        "pt": ".pt",
        "pt_uni": ".pt",
        "onnx": ".onnx",
        "engine": ".engine",
        "trt": ".trt",
    }

    def _has_model_artifact(run_dir: str, fmt: str, entry: dict[str, Any]) -> bool:
        target = entry.get("target_path")
        if isinstance(target, str) and target.strip():
            candidate = target if os.path.isabs(target) else os.path.join(run_dir, target)
            if os.path.isfile(candidate):
                return True
        artifacts = entry.get("artifacts")
        if isinstance(artifacts, list):
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                target_item = item.get("target_path")
                if not isinstance(target_item, str) or not target_item.strip():
                    continue
                candidate = target_item if os.path.isabs(target_item) else os.path.join(run_dir, target_item)
                if os.path.isfile(candidate):
                    return True
        if fmt in {"pt", "pt_uni"}:
            return resolve_run_model_with_legacy_fallback(run_dir, ".pt") is not None
        ext = ext_by_format.get(fmt)
        if not ext:
            return False
        return any(os.path.isfile(p) for p in glob(os.path.join(run_dir, "**", f"*{ext}"), recursive=True))

    def _pick_target_path(entry: dict[str, Any]) -> str | None:
        target = entry.get("target_path")
        if isinstance(target, str) and target.strip():
            return target
        artifacts = entry.get("artifacts")
        if isinstance(artifacts, list):
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                t = item.get("target_path")
                if isinstance(t, str) and t.strip():
                    return t
        return None

    def _format_alias_prefix(fmt: str) -> str:
        return {
            "pt": "PT",
            "pt_uni": "PTUNI",
            "onnx": "ONNX",
            "engine": "ENGINE",
            "trt": "TRT",
        }.get(fmt, str(fmt).upper())

    def _iter_entry_variants(
        run_dir: str,
        fmt: str,
        entry: dict[str, Any],
        split_metrics: list[dict[str, str]],
        split_name: str,
    ) -> list[dict[str, Any]]:
        def _resolve_metrics_candidate(path_value: str | None) -> str | None:
            raw = str(path_value or "").strip()
            if not raw:
                return None
            candidate = os.path.abspath(os.path.join(run_dir, raw)) if not os.path.isabs(raw) else raw
            if os.path.isfile(candidate):
                return candidate
            # Legacy manifest values may omit "tests/" prefix after migration.
            base = os.path.basename(raw)
            if base:
                migrated = os.path.join(run_dir, "tests", base)
                if os.path.isfile(migrated):
                    return os.path.abspath(migrated)
            return candidate

        variants: list[dict[str, Any]] = []
        artifacts = entry.get("artifacts")
        if isinstance(artifacts, list):
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                target_rel = str(item.get("target_path") or "").strip()
                target_abs = (
                    os.path.abspath(os.path.join(run_dir, target_rel))
                    if target_rel and not os.path.isabs(target_rel)
                    else (target_rel or None)
                )
                metrics_rel = str(item.get("metrics_csv") or "").strip()
                metrics_abs = _resolve_metrics_candidate(metrics_rel)
                matched = None
                preferred_split_metrics = list(split_metrics)
                split_token = f"{split_name}_metrics"
                split_specific = [
                    rec
                    for rec in split_metrics
                    if split_token in os.path.basename(str(rec.get("metrics_path") or "")).lower()
                ]
                if split_specific:
                    preferred_split_metrics = split_specific
                if target_abs:
                    for rec in preferred_split_metrics:
                        if rec.get("target_path") == target_abs:
                            matched = rec
                            break
                if matched is None and metrics_abs:
                    for rec in preferred_split_metrics:
                        if rec.get("metrics_path") == metrics_abs:
                            matched = rec
                            break
                if matched is None and preferred_split_metrics and (not target_abs or fmt in {"pt", "pt_uni"}):
                    # Split-specific metrics should take precedence (e.g. val for pt_uni)
                    # even when artifact-level metrics_csv points to legacy test path.
                    matched = preferred_split_metrics[0]
                variants.append(
                    {
                        "target_path": target_rel or None,
                        "metrics_path": (matched or {}).get("metrics_path") or metrics_abs,
                        "status": item.get("status", entry.get("status")),
                        "error": item.get("error", entry.get("error")),
                        "backend": item.get("backend", entry.get("backend")),
                        "performance": item.get("performance"),
                    }
                )
        if not variants:
            fallback_metrics = split_metrics[0].get("metrics_path") if split_metrics else None
            variants.append(
                {
                    "target_path": _pick_target_path(entry),
                    "metrics_path": fallback_metrics,
                    "status": entry.get("status"),
                    "error": entry.get("error"),
                    "backend": entry.get("backend"),
                    "performance": entry.get("performance"),
                }
            )
        # Deduplicate by target+metrics and prefer variants with resolved metrics.
        deduped: dict[tuple[str, str], dict[str, Any]] = {}
        for item in variants:
            key = (
                str(item.get("target_path") or ""),
                str(item.get("metrics_path") or ""),
            )
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = item
                continue
            cur_has_metrics = bool(str(item.get("metrics_path") or "").strip())
            prev_has_metrics = bool(str(existing.get("metrics_path") or "").strip())
            if cur_has_metrics and not prev_has_metrics:
                deduped[key] = item
        variants = list(deduped.values())
        if len(variants) > 1:
            # Prefer entries with concrete target path when empty placeholders exist.
            with_target = [v for v in variants if str(v.get("target_path") or "").strip()]
            if with_target:
                variants = with_target
        if len(variants) > 1:
            # If at least one target exists on disk, drop non-existing targets.
            existing_target_variants = []
            for v in variants:
                t = str(v.get("target_path") or "").strip()
                if not t:
                    continue
                candidate = t if os.path.isabs(t) else os.path.join(run_dir, t)
                if os.path.isfile(candidate):
                    existing_target_variants.append(v)
            if existing_target_variants:
                variants = existing_target_variants
        # Stable order: metric-bearing entries first.
        variants.sort(key=lambda v: (0 if str(v.get("metrics_path") or "").strip() else 1, str(v.get("target_path") or "")))
        return variants

    def _read_eval_args(run_dir: str, fmt: str) -> dict[str, Any]:
        if fmt == "pt":
            args_yaml = os.path.join(run_dir, "tests", "test-ultralytics", "args.yaml")
            if not os.path.isfile(args_yaml):
                args_yaml = os.path.join(run_dir, "test-ultralytics", "args.yaml")
            if not os.path.isfile(args_yaml):
                args_yaml = os.path.join(run_dir, "test", "args.yaml")
        else:
            args_yaml = os.path.join(run_dir, "tests", f"test_{fmt}", "args.yaml")
            if not os.path.isfile(args_yaml):
                args_yaml = os.path.join(run_dir, f"test_{fmt}", "args.yaml")
        if not os.path.isfile(args_yaml):
            if fmt != "pt":
                return {}
            metadata_path = os.path.join(run_dir, "training_metadata.json")
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}
            inf = payload.get("inference") if isinstance(payload, dict) else {}
            if isinstance(inf, dict) and inf:
                return {
                    "imgsz": inf.get("imgsz"),
                    "conf": inf.get("conf", 0.001 if inf.get("conf") is None else inf.get("conf")),
                    "iou": inf.get("iou"),
                    "inference_source": "ultralytics_model_val",
                    "gt_source": "ultralytics_validator",
                    "nms_profile": "ultralytics_validator_multilabel",
                }
            # Newer runs may omit top-level "inference" but still have training_info.imgsz.
            ti = payload.get("training_info") if isinstance(payload, dict) else {}
            if isinstance(ti, dict):
                ut = ti.get("ultralytics_train") if isinstance(ti.get("ultralytics_train"), dict) else {}
                hp = ti.get("hyperparameters") if isinstance(ti.get("hyperparameters"), dict) else {}
                imgsz = ut.get("imgsz")
                if imgsz is None:
                    imgsz = hp.get("image_size")
                if imgsz is not None:
                    return {
                        "imgsz": imgsz,
                        "conf": 0.001,
                        "iou": 0.7,
                        "inference_source": "ultralytics_model_val",
                        "gt_source": "ultralytics_validator",
                        "nms_profile": "ultralytics_validator_multilabel",
                    }
            return {}
        try:
            with open(args_yaml, "r", encoding="utf-8") as f:
                payload = yaml.safe_load(f) or {}
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _read_metric_row(metrics_path: str | None) -> dict[str, Any]:
        if not metrics_path or not os.path.isfile(metrics_path):
            return {}
        try:
            mdf = pd.read_csv(metrics_path)
            if len(mdf) == 0:
                return {}
            mdf.columns = [str(c).strip() for c in mdf.columns]
            # Prefer explicit aggregate row for Ultralytics-like CSVs.
            if "Class" in mdf.columns:
                cls = mdf["Class"].astype(str).str.strip().str.lower()
                all_mask = cls.eq("all")
                if bool(all_mask.any()):
                    return dict(mdf.loc[all_mask].iloc[0].to_dict())
            # Some generated pt/pt_uni metrics CSVs are per-class only.
            # For compare tables we need run-level aggregate, so use macro mean.
            if "Class" in mdf.columns and len(mdf) > 1:
                out: dict[str, Any] = {}
                for col in METRIC_AGG_COLUMNS:
                    if col in mdf.columns:
                        out[col] = pd.to_numeric(mdf[col], errors="coerce").mean()
                if out:
                    out["Class"] = "all"
                    return out
            return dict(mdf.iloc[0].to_dict())
        except Exception:
            return {}

    def _metrics_path_matches_split(metrics_path: str | None, split_name: str) -> bool:
        if not metrics_path:
            return False
        base = os.path.basename(str(metrics_path)).lower()
        token = f"{split_name}_metrics"
        return token in base

    def _normalize_issue_reason(reason: str) -> tuple[str, str]:
        raw = str(reason or "").strip()
        if raw.startswith("[") and "]" in raw:
            maybe_code = raw[1 : raw.index("]")].strip().lower()
            detail = raw[raw.index("]") + 1 :].strip()
            if maybe_code:
                return maybe_code, detail or raw
        lower = raw.lower()
        if "timeout" in lower:
            return "timeout", raw
        if "out of memory" in lower or "bfc_arena" in lower or "cudamalloc" in lower:
            return "oom_gpu", raw
        if "terminated by signal" in lower:
            return "signal_terminated", raw
        if "runtime_exception" in lower or "onnxruntimeerror" in lower:
            return "runtime_exception", raw
        if "session init" in lower or "inferencesession" in lower:
            return "init_session_failed", raw
        if "missing" in lower:
            return "missing_artifact", raw
        return "unknown", raw

    def _is_invalid_zero_metrics(fmt: str, metric_row: dict[str, Any]) -> bool:
        if fmt not in {"engine", "trt"}:
            return False
        vals: list[float] = []
        for col in METRIC_AGG_COLUMNS:
            raw_v = metric_row.get(col)
            if raw_v is None or (isinstance(raw_v, float) and pd.isna(raw_v)):
                return False
            try:
                vals.append(float(raw_v))
            except (TypeError, ValueError):
                return False
        return bool(vals) and all(abs(v) <= 1e-12 for v in vals)

    def _perf_context_for_variant(run_dir: str, fmt: str, target_path: Any) -> dict[str, Any]:
        profile_map = read_test_system_profile_by_format_artifacts(run_dir)
        records = profile_map.get(fmt) if isinstance(profile_map, dict) else None
        if not isinstance(records, list) or not records:
            return {}
        target_abs = ""
        if isinstance(target_path, str) and target_path.strip():
            target_abs = os.path.abspath(os.path.join(run_dir, target_path))
        target_name = os.path.basename(target_abs) if target_abs else ""
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rec_target = str(rec.get("target_path") or "")
            rec_name = os.path.basename(rec_target) if rec_target else ""
            profile = rec.get("test_system_profile")
            if not isinstance(profile, dict) or not profile:
                continue
            if target_abs and rec_target and os.path.abspath(rec_target) == target_abs:
                return profile
            if target_name and rec_name and rec_name == target_name:
                return profile
        return {}

    def _extract_perf_details(
        perf: dict[str, Any], eval_args: dict[str, Any], profile: dict[str, Any]
    ) -> dict[str, Any]:
        lat_all = perf.get("latency_ms") if isinstance(perf.get("latency_ms"), dict) else {}
        all_stats = lat_all.get("all") if isinstance(lat_all.get("all"), dict) else {}
        steady_stats = lat_all.get("steady") if isinstance(lat_all.get("steady"), dict) else {}
        breakdown = perf.get("breakdown_ms") if isinstance(perf.get("breakdown_ms"), dict) else {}

        def _stage(*names: str) -> dict[str, Any]:
            for name in names:
                candidate = breakdown.get(name)
                if isinstance(candidate, dict):
                    return candidate
            return {}

        # Backends may serialize stage keys with different naming conventions.
        preprocess = _stage("preprocess", "preprocess_ms")
        inference = _stage("infer", "inference", "infer_ms")
        postprocess = _stage("postprocess", "decode_nms", "decode_nms_ms")
        total = _stage("total", "total_ms", "infer_total_only_ms")
        io_load = _stage("io_load_ms")
        diag_alloc = _stage("diagnostics_alloc_ms")
        diag_h2d = _stage("diagnostics_h2d_ms")
        diag_exec = _stage("diagnostics_execute_ms")
        diag_d2h = _stage("diagnostics_d2h_ms")
        diagnostics = (
            perf.get("diagnostics_overhead")
            if isinstance(perf.get("diagnostics_overhead"), dict)
            else {}
        )

        runtime = profile.get("runtime") if isinstance(profile.get("runtime"), dict) else {}
        device = (
            perf.get("eval_device")
            if perf.get("eval_device") is not None
            else (runtime.get("device") if runtime.get("device") is not None else eval_args.get("device"))
        )
        batch_raw = perf.get("eval_batch") if perf.get("eval_batch") is not None else eval_args.get("batch")
        try:
            batch_val = int(batch_raw) if batch_raw is not None else None
        except (TypeError, ValueError):
            batch_val = None
        if batch_val is None:
            batch_val = 1
        if device is None and batch_val is not None:
            # Keep device explicit for PT rows where runtime profile may not expose it.
            device = "0"

        infer_p50 = inference.get("p50") if inference.get("p50") is not None else inference.get("mean")
        infer_p95 = inference.get("p95") if inference.get("p95") is not None else inference.get("p90")
        try:
            infer_ms = float(inference.get("mean")) if inference.get("mean") is not None else None
        except (TypeError, ValueError):
            infer_ms = None
        pure_infer_throughput = (1000.0 / infer_ms) if (infer_ms is not None and infer_ms > 0) else None
        throughput_value = pure_infer_throughput if pure_infer_throughput is not None else perf.get("throughput_img_s")
        return {
            # For cross-format comparability prefer pure inference stage timing.
            "throughput_img_s": throughput_value,
            "latency_p50_ms": infer_p50 if infer_p50 is not None else steady_stats.get("p50", all_stats.get("p50")),
            "latency_p95_ms": infer_p95 if infer_p95 is not None else steady_stats.get("p95", all_stats.get("p95")),
            "perf_preprocess_ms_per_frame": preprocess.get("mean"),
            "perf_inference_ms_per_frame": inference.get("mean"),
            "perf_postprocess_ms_per_frame": postprocess.get("mean"),
            "perf_total_ms_per_frame": total.get("mean", steady_stats.get("mean", all_stats.get("mean"))),
            "perf_warmup_images": perf.get("warmup_images"),
            "perf_sample_count": perf.get("images_total"),
            "perf_batch": batch_val,
            "perf_device": device,
            # Non-comparable diagnostics: overhead excluded from primary KPI.
            "perf_io_load_ms_per_frame": io_load.get("mean"),
            "perf_diag_alloc_ms_per_frame": diag_alloc.get("mean"),
            "perf_diag_h2d_ms_per_frame": diag_h2d.get("mean"),
            "perf_diag_execute_ms_per_frame": diag_exec.get("mean"),
            "perf_diag_d2h_ms_per_frame": diag_d2h.get("mean"),
            "perf_diag_session_init_ms": diagnostics.get("session_init_ms"),
            "perf_diag_engine_init_ms": diagnostics.get("engine_init_ms"),
            "perf_diag_worker_wall_ms": diagnostics.get("worker_wall_ms"),
            "perf_diag_retries_count": diagnostics.get("retries_count"),
            "perf_diag_retry_sleep_ms": diagnostics.get("retry_sleep_ms"),
            "perf_diag_provider_switched_to_cpu": diagnostics.get("provider_switched_to_cpu"),
        }

    def _resolve_perf_and_reason(
        run_dir: str,
        fmt: str,
        target_path: Any,
        perf_candidate: Any,
        entry: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        perf = perf_candidate if isinstance(perf_candidate, dict) else {}
        if perf:
            return perf, "perf_present"
        perf_map = read_test_performance_by_format_artifacts(run_dir)
        records = perf_map.get(fmt) if isinstance(perf_map, dict) else None
        if not isinstance(records, list) or not records:
            artifacts = entry.get("artifacts") if isinstance(entry, dict) else None
            if isinstance(artifacts, list) and artifacts:
                return {}, "perf_not_collected_for_target"
            return {}, "perf_missing_manifest_entry"
        target_abs = ""
        if isinstance(target_path, str) and target_path.strip():
            target_abs = os.path.abspath(os.path.join(run_dir, target_path))
        target_name = os.path.basename(target_abs) if target_abs else ""
        target_stem = os.path.splitext(target_name)[0] if target_name else ""
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rec_perf = rec.get("performance")
            if not isinstance(rec_perf, dict) or not rec_perf:
                continue
            rec_target = str(rec.get("target_path") or "")
            rec_name = os.path.basename(rec_target) if rec_target else ""
            rec_stem = os.path.splitext(rec_name)[0] if rec_name else ""
            if target_abs and rec_target and os.path.abspath(rec_target) == target_abs:
                return rec_perf, "perf_present"
            if target_name and rec_name and rec_name == target_name:
                return rec_perf, "perf_present"
            if target_stem and rec_stem and rec_stem == target_stem:
                return rec_perf, "perf_present"
        if target_abs:
            return {}, "perf_target_mismatch_legacy_variant"
        return {}, "perf_not_collected_for_target"

    def _build_format_rows(
        split_name: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:

        rows: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        eval_rows: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        for run_dir in run_dirs:
            run_name = os.path.basename(run_dir.rstrip(os.sep))
            manifest = load_test_artifacts_manifest(run_dir)
            formats_meta = manifest.get("formats") if isinstance(manifest, dict) else {}
            metrics_paths = read_metrics_by_format_for_split(run_dir, split_name)
            metrics_artifacts = read_metrics_by_format_for_split_artifacts(run_dir, split_name)
            for fmt in ("pt", "onnx", "engine", "trt"):
                entry = formats_meta.get(fmt, {}) if isinstance(formats_meta, dict) else {}
                if not isinstance(entry, dict):
                    entry = {}
                fmt_metrics = list(metrics_artifacts.get(fmt) or [])
                if not fmt_metrics and metrics_paths.get(fmt):
                    fmt_metrics = [{"metrics_path": str(metrics_paths[fmt]), "target_path": ""}]
                variants = _iter_entry_variants(run_dir, fmt, entry, fmt_metrics, split_name)
                if len(variants) > 1:
                    with_metrics = [v for v in variants if str(v.get("metrics_path") or "").strip()]
                    if with_metrics:
                        variants = with_metrics
                if not variants and not _has_model_artifact(run_dir, fmt, entry):
                    continue
                eval_args = _read_eval_args(run_dir, fmt)
                has_any_metrics_variant = False
                for _v in variants:
                    _mp = str(_v.get("metrics_path") or "").strip()
                    if _mp and os.path.isfile(_mp):
                        has_any_metrics_variant = True
                        break
                for var in variants:
                    metrics_path = str(var.get("metrics_path") or "").strip() or None
                    metrics_exists = bool(metrics_path and os.path.isfile(metrics_path))
                    if split_name == "val" and metrics_exists and not _metrics_path_matches_split(metrics_path, split_name):
                        metrics_exists = False
                    if not metrics_exists and not _has_model_artifact(run_dir, fmt, entry):
                        continue
                    status_raw = str(var.get("status") or "").strip()
                    err_raw = str(var.get("error") or "").strip()
                    status_lower = status_raw.lower()
                    has_explicit_failure = bool(err_raw) or status_lower in {
                        "failed",
                        "error",
                        "timeout",
                        "terminated",
                        "unavailable",
                    }
                    if not metrics_exists:
                        if split_name == "val" and not has_explicit_failure:
                            continue
                        if split_name != "val" and not has_explicit_failure:
                            continue
                    metric_row = _read_metric_row(metrics_path)
                    invalid_zero_metrics = metrics_exists and _is_invalid_zero_metrics(fmt, metric_row)
                    row: dict[str, Any] = {
                        "run_dir": run_dir,
                        "run_name": run_name,
                        "split": split_name,
                        "format": fmt,
                        "backend_status": var.get("backend"),
                        "target_path": var.get("target_path"),
                        "metrics_source": os.path.relpath(metrics_path, run_dir) if metrics_exists and metrics_path else None,
                        "inference_source": eval_args.get("inference_source"),
                        "gt_source": eval_args.get("gt_source"),
                        "nms_profile": eval_args.get("nms_profile"),
                        "mAP50-95": None if invalid_zero_metrics else metric_row.get("mAP50-95"),
                        "mAP50": None if invalid_zero_metrics else metric_row.get("mAP50"),
                        "Box-F1": None if invalid_zero_metrics else metric_row.get("Box-F1"),
                        "Box-P": None if invalid_zero_metrics else metric_row.get("Box-P"),
                        "Box-R": None if invalid_zero_metrics else metric_row.get("Box-R"),
                    }
                    perf, perf_reason = _resolve_perf_and_reason(
                        run_dir, fmt, var.get("target_path"), var.get("performance"), entry
                    )
                    profile = _perf_context_for_variant(run_dir, fmt, var.get("target_path"))
                    row.update(_extract_perf_details(perf, eval_args, profile))
                    row["performance_status"] = "ok" if isinstance(perf, dict) and len(perf) > 0 else "performance_not_collected"
                    row["performance_reason"] = perf_reason
                    try:
                        thr = float(row["throughput_img_s"]) if row.get("throughput_img_s") is not None else None
                    except (TypeError, ValueError):
                        thr = None
                    try:
                        p50 = float(row["latency_p50_ms"]) if row.get("latency_p50_ms") is not None else None
                    except (TypeError, ValueError):
                        p50 = None
                    row["avg_inference_fps"] = thr
                    row["avg_inference_ms_per_frame"] = p50 if p50 is not None else ((1000.0 / thr) if thr and thr > 0 else None)
                    eval_rows.append(
                        {
                            "run_dir": run_dir,
                            "run_name": run_name,
                            "split": split_name,
                            "format": fmt,
                            "target_path": var.get("target_path"),
                            "eval_imgsz": eval_args.get("imgsz"),
                            "eval_conf": eval_args.get("conf"),
                            "eval_iou": eval_args.get("iou"),
                            "inference_source": eval_args.get("inference_source"),
                            "gt_source": eval_args.get("gt_source"),
                            "nms_profile": eval_args.get("nms_profile"),
                        }
                    )
                    if metrics_exists:
                        if not row.get("backend_status"):
                            row["backend_status"] = backend_fallback.get(fmt)
                        if invalid_zero_metrics:
                            issues.append(
                                {
                                    "run_name": run_name,
                                    "split": split_name,
                                    "format": fmt,
                                    "target_path": var.get("target_path"),
                                    "status": str(var.get("status") or "ok"),
                                    "reason": "metrics are all zeros; treated as invalid native evaluation output",
                                    "reason_code": "invalid_metrics",
                                }
                            )
                    else:
                        if status_raw or err_raw:
                            if has_any_metrics_variant:
                                # Do not report missing duplicates when at least one
                                # variant for the same run/split/format has metrics.
                                continue
                            reason_code, reason_detail = _normalize_issue_reason(err_raw or "metrics missing")
                            issues.append(
                                {
                                    "run_name": run_name,
                                    "split": split_name,
                                    "format": fmt,
                                    "target_path": var.get("target_path"),
                                    "status": status_raw or "unknown",
                                    "reason": reason_detail,
                                    "reason_code": reason_code,
                                }
                            )
                    rows.append(row)
                    sources.append(
                        {
                            "run_dir": run_dir,
                            "run_name": run_name,
                            "split": split_name,
                            "format": fmt,
                            "target_path": var.get("target_path"),
                            "metrics_source": row.get("metrics_source"),
                            "inference_source": row.get("inference_source"),
                            "gt_source": row.get("gt_source"),
                            "nms_profile": row.get("nms_profile"),
                        }
                    )
        return rows, sources, eval_rows, issues

    def _build_pt_uni_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        eval_rows: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        for split_name in ("test", "val"):
            for run_dir in run_dirs:
                run_name = os.path.basename(run_dir.rstrip(os.sep))
                manifest = load_test_artifacts_manifest(run_dir)
                formats_meta = manifest.get("formats") if isinstance(manifest, dict) else {}
                metrics_paths = read_metrics_by_format_for_split(run_dir, split_name, include_internal=True)
                metrics_artifacts = read_metrics_by_format_for_split_artifacts(run_dir, split_name, include_internal=True)
                for fmt in ("pt", "pt_uni"):
                    entry = formats_meta.get(fmt, {}) if isinstance(formats_meta, dict) else {}
                    if not isinstance(entry, dict):
                        entry = {}
                    fmt_metrics = list(metrics_artifacts.get(fmt) or [])
                    if not fmt_metrics and metrics_paths.get(fmt):
                        fmt_metrics = [{"metrics_path": str(metrics_paths[fmt]), "target_path": ""}]
                    variants = _iter_entry_variants(run_dir, fmt, entry, fmt_metrics, split_name)
                    if len(variants) > 1:
                        with_metrics = [v for v in variants if str(v.get("metrics_path") or "").strip()]
                        if with_metrics:
                            variants = with_metrics
                    eval_args = _read_eval_args(run_dir, fmt)
                    has_any_metrics_variant = False
                    for _v in variants:
                        _mp = str(_v.get("metrics_path") or "").strip()
                        if _mp and os.path.isfile(_mp):
                            has_any_metrics_variant = True
                            break
                    for var in variants:
                        metrics_path = str(var.get("metrics_path") or "").strip() or None
                        metrics_exists = bool(metrics_path and os.path.isfile(metrics_path))
                        if split_name == "val" and metrics_exists and not _metrics_path_matches_split(metrics_path, split_name):
                            metrics_exists = False
                        if not metrics_exists and not _has_model_artifact(run_dir, fmt, entry):
                            continue
                        status_raw = str(var.get("status") or "").strip()
                        err_raw = str(var.get("error") or "").strip()
                        status_lower = status_raw.lower()
                        has_explicit_failure = bool(err_raw) or status_lower in {
                            "failed",
                            "error",
                            "timeout",
                            "terminated",
                            "unavailable",
                        }
                        if not metrics_exists:
                            if split_name == "val" and not has_explicit_failure:
                                continue
                            if split_name != "val" and not has_explicit_failure:
                                continue
                        metric_row = _read_metric_row(metrics_path)
                        row: dict[str, Any] = {
                            "run_dir": run_dir,
                            "run_name": run_name,
                            "split": split_name,
                            "format": fmt,
                            "backend_status": var.get("backend"),
                            "target_path": var.get("target_path"),
                            "metrics_source": os.path.relpath(metrics_path, run_dir) if metrics_exists and metrics_path else None,
                            "inference_source": eval_args.get("inference_source"),
                            "gt_source": eval_args.get("gt_source"),
                            "nms_profile": eval_args.get("nms_profile"),
                            "mAP50-95": metric_row.get("mAP50-95"),
                            "mAP50": metric_row.get("mAP50"),
                            "Box-F1": metric_row.get("Box-F1"),
                            "Box-P": metric_row.get("Box-P"),
                            "Box-R": metric_row.get("Box-R"),
                        }
                        perf, perf_reason = _resolve_perf_and_reason(
                            run_dir, fmt, var.get("target_path"), var.get("performance"), entry
                        )
                        profile = _perf_context_for_variant(run_dir, fmt, var.get("target_path"))
                        row.update(_extract_perf_details(perf, eval_args, profile))
                        row["performance_status"] = "ok" if isinstance(perf, dict) and len(perf) > 0 else "performance_not_collected"
                        row["performance_reason"] = perf_reason
                        try:
                            thr = float(row["throughput_img_s"]) if row.get("throughput_img_s") is not None else None
                        except (TypeError, ValueError):
                            thr = None
                        try:
                            p50 = float(row["latency_p50_ms"]) if row.get("latency_p50_ms") is not None else None
                        except (TypeError, ValueError):
                            p50 = None
                        row["avg_inference_fps"] = thr
                        row["avg_inference_ms_per_frame"] = p50 if p50 is not None else ((1000.0 / thr) if thr and thr > 0 else None)
                        eval_rows.append(
                            {
                                "run_dir": run_dir,
                                "run_name": run_name,
                                "split": split_name,
                                "format": fmt,
                                "target_path": var.get("target_path"),
                                "eval_imgsz": eval_args.get("imgsz"),
                                "eval_conf": eval_args.get("conf"),
                                "eval_iou": eval_args.get("iou"),
                                "inference_source": eval_args.get("inference_source"),
                                "gt_source": eval_args.get("gt_source"),
                                "nms_profile": eval_args.get("nms_profile"),
                            }
                        )
                        if metrics_exists:
                            if not row.get("backend_status"):
                                row["backend_status"] = backend_fallback.get(fmt)
                        else:
                            if status_raw or err_raw:
                                if has_any_metrics_variant:
                                    continue
                                reason_code, reason_detail = _normalize_issue_reason(err_raw or "metrics missing")
                                issues.append(
                                    {
                                        "run_name": run_name,
                                        "split": split_name,
                                        "format": fmt,
                                        "target_path": var.get("target_path"),
                                        "status": status_raw or "unknown",
                                        "reason": reason_detail,
                                        "reason_code": reason_code,
                                    }
                                )
                        rows.append(row)
                        sources.append(
                            {
                                "run_dir": run_dir,
                                "run_name": run_name,
                                "split": split_name,
                                "format": fmt,
                                "target_path": var.get("target_path"),
                                "metrics_source": row.get("metrics_source"),
                                "inference_source": row.get("inference_source"),
                                "gt_source": row.get("gt_source"),
                                "nms_profile": row.get("nms_profile"),
                            }
                        )
        return rows, sources, eval_rows, issues

    test_rows, test_sources, test_eval_rows, test_issues = _build_format_rows("test")
    val_rows, val_sources, val_eval_rows, val_issues = _build_format_rows("val")
    pt_uni_rows, pt_uni_sources, pt_uni_eval_rows, pt_uni_issues = _build_pt_uni_rows()
    if not test_rows and not val_rows and not pt_uni_rows:
        return None
    out_dir = os.path.join(session_root, "artifacts", "format_compare")
    os.makedirs(out_dir, exist_ok=True)
    out: dict[str, str] = {}
    all_rows = test_rows + val_rows + pt_uni_rows
    alias_legend: list[dict[str, str]] = []
    alias_counters: dict[str, int] = {}
    for row in sorted(
        all_rows,
        key=lambda r: (str(r.get("format") or ""), str(r.get("run_name") or ""), str(r.get("target_path") or "")),
    ):
        fmt = str(row.get("format") or "")
        prefix = _format_alias_prefix(fmt)
        alias_counters[prefix] = int(alias_counters.get(prefix, 0)) + 1
        alias = f"{prefix}{alias_counters[prefix]}"
        row["alias"] = alias
        alias_legend.append(
            {
                "alias": alias,
                "format": fmt,
                "run_name": str(row.get("run_name") or ""),
                "target_path": str(row.get("target_path") or ""),
            }
        )
    if test_rows:
        out_csv = os.path.join(out_dir, "format_metrics_compare_test.csv")
        pd.DataFrame(test_rows).to_csv(out_csv, index=False, encoding="utf-8")
        out["test_csv"] = os.path.relpath(out_csv, session_root)
        perf_test = pd.DataFrame(test_rows)
        perf_csv = os.path.join(out_dir, "format_performance_compare_test.csv")
        perf_test.to_csv(perf_csv, index=False, encoding="utf-8")
        out["perf_test_csv"] = os.path.relpath(perf_csv, session_root)
    if val_rows:
        out_csv = os.path.join(out_dir, "format_metrics_compare_val.csv")
        pd.DataFrame(val_rows).to_csv(out_csv, index=False, encoding="utf-8")
        out["val_csv"] = os.path.relpath(out_csv, session_root)
    if pt_uni_rows:
        out_csv = os.path.join(out_dir, "format_metrics_compare_pt_uni.csv")
        pd.DataFrame(pt_uni_rows).to_csv(out_csv, index=False, encoding="utf-8")
        out["pt_uni_csv"] = os.path.relpath(out_csv, session_root)
    eval_rows = test_eval_rows + val_eval_rows + pt_uni_eval_rows
    alias_by_key = {
        (str(r.get("run_name") or ""), str(r.get("split") or ""), str(r.get("format") or ""), str(r.get("target_path") or "")): str(
            r.get("alias") or ""
        )
        for r in all_rows
    }
    for er in eval_rows:
        er["alias"] = alias_by_key.get(
            (
                str(er.get("run_name") or ""),
                str(er.get("split") or ""),
                str(er.get("format") or ""),
                str(er.get("target_path") or ""),
            ),
            "",
        )
    if eval_rows:
        eval_csv = os.path.join(out_dir, "format_eval_settings.csv")
        pd.DataFrame(eval_rows).drop_duplicates().to_csv(eval_csv, index=False, encoding="utf-8")
        out["eval_csv"] = os.path.relpath(eval_csv, session_root)
    if alias_legend:
        alias_csv = os.path.join(out_dir, "format_alias_legend.csv")
        pd.DataFrame(alias_legend).to_csv(alias_csv, index=False, encoding="utf-8")
        out["alias_legend_csv"] = os.path.relpath(alias_csv, session_root)
    issues = test_issues + val_issues + pt_uni_issues
    if issues:
        deduped_issues: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            key = (
                str(issue.get("run_name") or ""),
                str(issue.get("split") or ""),
                str(issue.get("format") or ""),
                str(issue.get("reason_code") or ""),
            )
            existing = deduped_issues.get(key)
            if existing is None:
                deduped_issues[key] = issue
                continue
            # Prefer richer records with concrete artifact path and non-failed status.
            cur_target = str(issue.get("target_path") or "").strip()
            prev_target = str(existing.get("target_path") or "").strip()
            cur_failed = str(issue.get("status") or "").strip().lower() in {"failed", "unavailable"}
            prev_failed = str(existing.get("status") or "").strip().lower() in {"failed", "unavailable"}
            if (cur_target and not prev_target) or (prev_failed and not cur_failed):
                deduped_issues[key] = issue
        issues = list(deduped_issues.values())
    if issues:
        issues_json = os.path.join(out_dir, "format_compare_issues.json")
        with open(issues_json, "w", encoding="utf-8") as f:
            json.dump(issues, f, ensure_ascii=False, indent=2)
        out["issues_json"] = os.path.relpath(issues_json, session_root)
    out_sources = os.path.join(out_dir, "format_metrics_sources.json")
    with open(out_sources, "w", encoding="utf-8") as f:
        json.dump(test_sources + val_sources + pt_uni_sources, f, ensure_ascii=False, indent=2)
    return out


def _resolve_pr_output_png(
    workspace_cli: str | None,
    out_png_cli: str | None,
    runs_group_dir: str,
) -> str:
    if out_png_cli:
        return os.path.abspath(os.path.expanduser(out_png_cli))
    try:
        ws = resolve_workspace_root(workspace_cli)
        analytics_dir = os.path.join(WorkspaceLayout(ws).analytics, "pr_curves")
    except ValueError:
        analytics_dir = os.path.join(
            os.path.dirname(os.path.abspath(runs_group_dir)),
            "analytics",
            "pr_curves",
        )
    os.makedirs(analytics_dir, exist_ok=True)
    ds_name = os.path.basename(os.path.normpath(runs_group_dir))
    return os.path.join(analytics_dir, f"pr_all_classes_{ds_name}.png")


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
    all_run_dirs = sorted(
        d for d in glob(os.path.join(runs_group_dir, "*"))
        if os.path.isdir(d)
    )
    if not all_run_dirs:
        return []
    selected_ordered: list[str] = []
    for p in (selected_run_dirs or []):
        ps = str(p).strip()
        if not ps:
            continue
        ap = os.path.abspath(os.path.expanduser(ps))
        if ap not in selected_ordered:
            selected_ordered.append(ap)
    if not selected_ordered:
        return all_run_dirs
    # When explicit run dirs are provided (e.g. mixed datasets), honor them
    # directly even if they live outside runs_group_dir.
    explicit_existing = [d for d in selected_ordered if os.path.isdir(d)]
    if explicit_existing:
        return explicit_existing
    selected_norm = set(selected_ordered)
    return [d for d in all_run_dirs if os.path.abspath(d) in selected_norm]


def cmd_pr_curves(args: argparse.Namespace) -> None:
    if (not getattr(args, "runs_group_dir", None) or not getattr(args, "data_yaml", None)) and sys.stdin.isatty():
        args.runs_group_dir = prompt_text("Runs group dir", default=str(args.models_root)).strip() or str(args.models_root)
        args.data_yaml = prompt_text("Path to data.yaml", default=str(getattr(args, "data_yaml", ""))).strip()
    if not getattr(args, "runs_group_dir", None) or not getattr(args, "data_yaml", None):
        print("[ERROR] Incomplete arguments: --runs-group-dir and --data-yaml are required.", file=sys.stderr)
        sys.exit(2)
    runs_group_dir = os.path.abspath(os.path.expanduser(args.runs_group_dir))
    if not os.path.isdir(runs_group_dir):
        print(f"[ERROR] Models directory not found: {runs_group_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.data_yaml:
        print("[ERROR] Please provide --data-yaml (path to data.yaml for split=test).", file=sys.stderr)
        sys.exit(1)
    data_yaml = os.path.abspath(os.path.expanduser(args.data_yaml))
    if not os.path.isfile(data_yaml):
        print(f"[ERROR] data.yaml not found: {data_yaml}", file=sys.stderr)
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[ERROR] Failed to import ultralytics: {e}", file=sys.stderr)
        sys.exit(1)

    run_dirs = _resolve_selected_run_dirs(
        runs_group_dir,
        getattr(args, "selected_run_dirs", None),
    )
    if not run_dirs:
        print(f"[ERROR] No run directories found for scope in: {runs_group_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] PR scope: {len(run_dirs)} run(s) selected")

    curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    per_class_rows: list[dict[str, Any]] = []
    per_class_curves: dict[int, list[tuple[str, np.ndarray, np.ndarray, str]]] = {}
    class_names = _load_dataset_class_names(data_yaml)
    per_class_enabled = bool(getattr(args, "pr_per_class", True))
    reuse_cache = bool(getattr(args, "reuse_run_cache", True))
    tool_version = "analyze-v2"
    cache_stats: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        label = os.path.basename(run_dir.rstrip(os.sep))
        best_pt = canonical_run_model_path(run_dir, ".pt")
        if not os.path.isfile(best_pt):
            print(f"[WARN] {label}: missing run model, skipping ({best_pt})")
            continue
        cache_root = run_cache_root(run_dir)
        fp = compute_fingerprint(
            {
                "tool": tool_version,
                "task": "pr_curves",
                "data_yaml_hash": data_yaml_hash(data_yaml),
                "split": "test",
                "weights_hash": weights_hash(run_dir),
                "per_class": per_class_enabled,
            }
        )
        cache_agg = os.path.join(cache_root, "pr", "aggregate", f"pr_{fp}.csv")
        cache_pc = os.path.join(cache_root, "pr", "per_class", f"pr_per_class_{fp}.csv")
        os.makedirs(os.path.dirname(cache_agg), exist_ok=True)
        os.makedirs(os.path.dirname(cache_pc), exist_ok=True)
        recall: np.ndarray | None = None
        precision: np.ndarray | None = None
        if reuse_cache and os.path.isfile(cache_agg):
            cdf = pd.read_csv(cache_agg)
            if {"recall", "precision"}.issubset(set(cdf.columns)):
                recall = cdf["recall"].to_numpy(dtype=float)
                precision = cdf["precision"].to_numpy(dtype=float)
                cache_stats.append({"run_dir": run_dir, "artifact": "pr.aggregate", "status": "hit"})
        per_class_df: pd.DataFrame | None = None
        if per_class_enabled and reuse_cache and os.path.isfile(cache_pc):
            per_class_df = pd.read_csv(cache_pc)
            if len(per_class_df) > 0:
                cache_stats.append({"run_dir": run_dir, "artifact": "pr.per_class", "status": "hit"})
            else:
                per_class_df = None
        if recall is None or precision is None or (per_class_enabled and per_class_df is None):
            print(f"[INFO] {label}: val(split=test) ...")
            _clear_gpu_memory()
            model = YOLO(best_pt)
            try:
                rb, ri, rh = _resolve_run_val_profile(
                    run_dir,
                    default_batch=int(getattr(args, "val_batch", 1)),
                    default_imgsz=int(getattr(args, "val_imgsz", 640)),
                    default_half=bool(getattr(args, "val_half", True)),
                )
                ultra_proj = ultralytics_sidecar_dir(run_dir, ".ultralytics_scratch")
                metrics = _run_val_memory_safe(
                    model,
                    data_yaml=data_yaml,
                    split="test",
                    val_batch=rb,
                    val_imgsz=ri,
                    val_half=rh,
                    gpu_only=bool(getattr(args, "gpu_only_val", True)),
                    ultra_project=ultra_proj,
                    ultra_name="val-pr-curves",
                )
            except Exception as e:
                print(f"[WARN] {label}: val() error: {e}")
                _clear_gpu_memory()
                continue
            pr = _extract_pr_curve_from_metrics(metrics)
            if pr is None:
                print(f"[WARN] {label}: PR curve not available in metrics object, skipping")
                continue
            recall, precision = pr
            pd.DataFrame({"recall": recall, "precision": precision}).to_csv(cache_agg, index=False, encoding="utf-8")
            append_cache_entry(
                run_dir,
                {"artifact": "pr.aggregate", "fingerprint": fp, "path": os.path.relpath(cache_agg, run_dir)},
            )
            cache_stats.append({"run_dir": run_dir, "artifact": "pr.aggregate", "status": "miss"})
            if per_class_enabled:
                pc = _extract_pr_curve_per_class_from_metrics(metrics)
                if pc is not None:
                    rx, y2d = pc
                    pc_rows: list[dict[str, Any]] = []
                    for class_id in range(y2d.shape[0]):
                        class_name = class_names.get(class_id, f"class_{class_id}")
                        ap = float(np.trapz(y2d[class_id], rx))
                        for i in range(len(rx)):
                            pc_rows.append(
                                {
                                    "run_dir": run_dir,
                                    "model": label,
                                    "class_id": class_id,
                                    "class_name": class_name,
                                    "recall": float(rx[i]),
                                    "precision": float(y2d[class_id][i]),
                                    "ap": ap,
                                }
                            )
                    per_class_df = pd.DataFrame(pc_rows)
                    per_class_df.to_csv(cache_pc, index=False, encoding="utf-8")
                    append_cache_entry(
                        run_dir,
                        {"artifact": "pr.per_class", "fingerprint": fp, "path": os.path.relpath(cache_pc, run_dir)},
                    )
                    cache_stats.append({"run_dir": run_dir, "artifact": "pr.per_class", "status": "miss"})
            _clear_gpu_memory()

        curves.append((label, recall, precision))
        pr_dir = os.path.join(run_dir, "tests", "test-ultralytics")
        if not os.path.isdir(pr_dir):
            pr_dir = os.path.join(run_dir, "test")
        os.makedirs(pr_dir, exist_ok=True)
        pr_csv = os.path.join(pr_dir, "pr.csv")
        pd.DataFrame({"recall": recall, "precision": precision}).to_csv(pr_csv, index=False, encoding="utf-8")
        if per_class_df is not None and len(per_class_df) > 0:
            pr_pc_csv = os.path.join(pr_dir, "pr_per_class.csv")
            per_class_df.to_csv(pr_pc_csv, index=False, encoding="utf-8")
            per_class_rows.extend(per_class_df.to_dict(orient="records"))
        print(f"[OK] {label}: saved {pr_csv}")

    if not curves:
        print("[ERROR] Failed to obtain any PR curves.", file=sys.stderr)
        sys.exit(1)

    out_png = _resolve_pr_output_png(args.workspace, args.out_png, runs_group_dir)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    plt.figure(figsize=(10, 7))
    for label, recall, precision in curves:
        plt.plot(recall, precision, linewidth=2, label=label)
    plt.title("PR curves (all classes, test split)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Model", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[OK] Combined PR plot: {out_png}")
    if per_class_enabled and per_class_rows:
        out_base_dir = os.path.join(os.path.dirname(out_png), "per_class")
        os.makedirs(out_base_dir, exist_ok=True)
        long_df = pd.DataFrame(per_class_rows)
        long_csv = os.path.join(out_base_dir, "pr_per_class.csv")
        long_df.to_csv(long_csv, index=False, encoding="utf-8")
        grouped = long_df.groupby(["class_id", "class_name"], dropna=False)
        for (class_id, class_name), cls_df in grouped:
            plt.figure(figsize=(9, 6))
            for model_name, mdf in cls_df.groupby("model"):
                mdf = mdf.sort_values("recall")
                plt.plot(mdf["recall"], mdf["precision"], linewidth=1.8, label=model_name)
            plt.title(f"PR per class: {class_name} (id={class_id})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(fontsize=8)
            plt.tight_layout()
            cls_png = os.path.join(out_base_dir, f"pr_class_{int(class_id)}_{_safe_name(str(class_name))}.png")
            plt.savefig(cls_png, dpi=220)
            plt.close()
            per_class_curves.setdefault(int(class_id), []).append((str(class_name), np.array([]), np.array([]), cls_png))
        print(f"[OK] Per-class PR artifacts: {out_base_dir}")
    stats_out = str(getattr(args, "cache_stats_out", "") or "").strip()
    if stats_out:
        os.makedirs(os.path.dirname(stats_out) or ".", exist_ok=True)
        with open(stats_out, "w", encoding="utf-8") as f:
            json.dump({"cache": cache_stats}, f, ensure_ascii=False, indent=2)


def _collect_split_images(data_yaml_path: str, split_name: str, limit: int) -> list[str]:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {data_yaml_path}")
    split_rel = data.get(split_name)
    if not split_rel or not isinstance(split_rel, str):
        raise ValueError(f"data.yaml has no path for split={split_name!r}")

    base_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    split_path = os.path.abspath(os.path.join(base_dir, split_rel))
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"Split directory not found: {split_path}")

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images = sorted(
        p for p in glob(os.path.join(split_path, "**", "*"), recursive=True)
        if os.path.isfile(p) and p.lower().endswith(exts)
    )
    return images[:limit]


def _resolve_inference_csv_path(
    workspace_cli: str | None,
    out_csv_cli: str | None,
    runs_group_dir: str,
) -> str:
    if out_csv_cli:
        return os.path.abspath(os.path.expanduser(out_csv_cli))
    try:
        ws = resolve_workspace_root(workspace_cli)
        base = os.path.join(WorkspaceLayout(ws).analytics, "inference_tests")
    except ValueError:
        base = os.path.join(os.path.dirname(os.path.abspath(runs_group_dir)), "analytics", "inference_tests")
    os.makedirs(base, exist_ok=True)
    group_name = os.path.basename(os.path.normpath(runs_group_dir))
    return os.path.join(base, f"{group_name}.csv")


def cmd_inference_benchmark(args: argparse.Namespace) -> None:
    if (not getattr(args, "runs_group_dir", None) or not getattr(args, "data_yaml", None)) and sys.stdin.isatty():
        args.runs_group_dir = prompt_text("Runs group dir", default=str(args.models_root)).strip() or str(args.models_root)
        args.data_yaml = prompt_text("Path to data.yaml", default=str(getattr(args, "data_yaml", ""))).strip()
    if not getattr(args, "runs_group_dir", None) or not getattr(args, "data_yaml", None):
        print("[ERROR] Incomplete arguments: --runs-group-dir and --data-yaml are required.", file=sys.stderr)
        sys.exit(2)
    runs_group_dir = os.path.abspath(os.path.expanduser(args.runs_group_dir))
    if not os.path.isdir(runs_group_dir):
        print(f"[ERROR] Models directory not found: {runs_group_dir}", file=sys.stderr)
        sys.exit(1)
    data_yaml = os.path.abspath(os.path.expanduser(args.data_yaml))
    if not os.path.isfile(data_yaml):
        print(f"[ERROR] data.yaml not found: {data_yaml}", file=sys.stderr)
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[ERROR] Failed to import ultralytics: {e}", file=sys.stderr)
        sys.exit(1)

    requested_device = str(args.device).strip() if args.device is not None else "cpu"
    effective_device = requested_device or "cpu"
    effective_half = bool(args.half)
    if effective_device.lower() != "cpu":
        try:
            import torch

            if not torch.cuda.is_available():
                print(
                    f"[WARN] CUDA is unavailable (torch.cuda.is_available()=False). "
                    f"Switching device from {effective_device!r} to 'cpu'."
                )
                effective_device = "cpu"
        except Exception as e:
            print(f"[WARN] Could not validate CUDA via torch ({e}); using CPU.")
            effective_device = "cpu"
    if effective_device.lower() == "cpu" and effective_half:
        print("[WARN] --half is not used on CPU; disabling half.")
        effective_half = False

    try:
        images = _collect_split_images(data_yaml, args.split, args.frames)
    except Exception as e:
        print(f"[ERROR] Failed to load test frames: {e}", file=sys.stderr)
        sys.exit(1)
    if not images:
        print("[ERROR] No images found for inference.", file=sys.stderr)
        sys.exit(1)

    run_dirs = _resolve_selected_run_dirs(
        runs_group_dir,
        getattr(args, "selected_run_dirs", None),
    )
    if not run_dirs:
        print(f"[ERROR] No run directories found for scope in: {runs_group_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Benchmark scope: {len(run_dirs)} run(s) selected")

    rows: list[dict[str, Any]] = []
    cache_stats: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        model_name = os.path.basename(run_dir.rstrip(os.sep))
        best_pt = canonical_run_model_path(run_dir, ".pt")
        if not os.path.isfile(best_pt):
            print(f"[WARN] {model_name}: missing run model, skipping")
            continue
        cache_root = run_cache_root(run_dir)
        fp = compute_fingerprint(
            {
                "tool": "analyze-v2",
                "task": "inference_benchmark",
                "split": args.split,
                "frames": int(args.frames),
                "device": effective_device,
                "half": bool(effective_half),
                "data_yaml_hash": data_yaml_hash(data_yaml),
                "weights_hash": weights_hash(run_dir),
            }
        )
        cache_csv = os.path.join(cache_root, "inference", f"bench_{fp}.csv")
        os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
        if bool(getattr(args, "reuse_run_cache", True)) and os.path.isfile(cache_csv):
            cdf = pd.read_csv(cache_csv)
            if len(cdf) > 0:
                row = cdf.iloc[0].to_dict()
                rows.append(row)
                cache_stats.append({"run_dir": run_dir, "artifact": "inference.benchmark", "status": "hit"})
                print(f"[INFO] {model_name}: benchmark cache hit")
                continue
        print(f"[INFO] {model_name}: benchmarking on {len(images)} frames ...")
        try:
            _clear_gpu_memory()
            model = YOLO(best_pt)
            pred_proj = ultralytics_sidecar_dir(run_dir, ".ultralytics_predict_scratch")
            pred_kw = dict(
                verbose=False,
                device=effective_device,
                half=effective_half,
                save=False,
                project=pred_proj,
                name="infer-bench",
                exist_ok=True,
            )
            # Warm-up to reduce first-iteration skew.
            model.predict(source=images[0], **pred_kw)
            timings_ms: list[float] = []
            prep_ms: list[float] = []
            infer_ms: list[float] = []
            post_ms: list[float] = []
            for img_path in tqdm(
                images,
                desc=f"{model_name} frames",
                unit="img",
                leave=False,
                disable=len(images) < 3,
            ):
                t0 = time.perf_counter()
                results = model.predict(source=img_path, **pred_kw)
                t1 = time.perf_counter()
                timings_ms.append((t1 - t0) * 1000.0)
                if results:
                    speed = getattr(results[0], "speed", None)
                    if isinstance(speed, dict):
                        p = speed.get("preprocess")
                        i = speed.get("inference")
                        po = speed.get("postprocess")
                        if p is not None:
                            prep_ms.append(float(p))
                        if i is not None:
                            infer_ms.append(float(i))
                        if po is not None:
                            post_ms.append(float(po))
            avg_ms = float(np.mean(timings_ms))
            avg_prep = float(np.mean(prep_ms)) if prep_ms else None
            avg_infer = float(np.mean(infer_ms)) if infer_ms else None
            avg_post = float(np.mean(post_ms)) if post_ms else None
            rows.append(
                {
                    "model": model_name,
                    "run_dir": run_dir,
                    "weights": best_pt,
                    "frames_count": len(images),
                    "device": effective_device,
                    "half": effective_half,
                    "avg_total_ms_per_frame": avg_ms,
                    "avg_preprocess_ms_per_frame": avg_prep,
                    "avg_inference_ms_per_frame": avg_infer,
                    "avg_postprocess_ms_per_frame": avg_post,
                    "avg_total_fps": (1000.0 / avg_ms) if avg_ms > 0 else None,
                    "avg_inference_fps": (1000.0 / avg_infer) if avg_infer and avg_infer > 0 else None,
                }
            )
            pd.DataFrame([rows[-1]]).to_csv(cache_csv, index=False, encoding="utf-8")
            append_cache_entry(
                run_dir,
                {"artifact": "inference.benchmark", "fingerprint": fp, "path": os.path.relpath(cache_csv, run_dir)},
            )
            cache_stats.append({"run_dir": run_dir, "artifact": "inference.benchmark", "status": "miss"})
            if avg_infer is not None:
                print(
                    f"[OK] {model_name}: total={avg_ms:.2f} ms/frame, "
                    f"infer={avg_infer:.2f} ms/frame"
                )
            else:
                print(f"[OK] {model_name}: total={avg_ms:.2f} ms/frame")
            _clear_gpu_memory()
        except Exception as e:
            print(f"[WARN] {model_name}: benchmark error: {e}")
            _clear_gpu_memory()

    if not rows:
        print("[ERROR] No benchmark results produced.", file=sys.stderr)
        sys.exit(1)

    out_csv = _resolve_inference_csv_path(args.workspace, args.out_csv, runs_group_dir)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    sort_col = "avg_inference_ms_per_frame" if any(
        r.get("avg_inference_ms_per_frame") is not None for r in rows
    ) else "avg_total_ms_per_frame"
    pd.DataFrame(rows).sort_values(sort_col).to_csv(
        out_csv, index=False, encoding="utf-8"
    )
    print(f"[OK] Results CSV: {out_csv}")
    stats_out = str(getattr(args, "cache_stats_out", "") or "").strip()
    if stats_out:
        os.makedirs(os.path.dirname(stats_out) or ".", exist_ok=True)
        with open(stats_out, "w", encoding="utf-8") as f:
            json.dump({"cache": cache_stats}, f, ensure_ascii=False, indent=2)


def _resolve_inference_plot_png(
    workspace_cli: str | None,
    out_png_cli: str | None,
    csv_path: str,
) -> str:
    if out_png_cli:
        return os.path.abspath(os.path.expanduser(out_png_cli))
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    try:
        ws = resolve_workspace_root(workspace_cli)
        base = os.path.join(WorkspaceLayout(ws).analytics, "inference_tests")
    except ValueError:
        base = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{csv_name}_bars.png")


def cmd_inference_plot(args: argparse.Namespace) -> None:
    if not getattr(args, "csv", None) and sys.stdin.isatty():
        default_csv = _default_relative_output(
            args.workspace, args.analytics_session, "inference", "benchmark.csv", None
        )
        args.csv = prompt_text("Path to benchmark CSV", default=default_csv).strip()
    if not getattr(args, "csv", None):
        print("[ERROR] Incomplete arguments: --csv is required.", file=sys.stderr)
        sys.exit(2)
    csv_path = os.path.abspath(os.path.expanduser(args.csv))
    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print(f"[ERROR] CSV is empty: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if "model" not in df.columns:
        print("[ERROR] CSV has no 'model' column.", file=sys.stderr)
        sys.exit(1)
    metric = args.metric
    if metric not in df.columns:
        print(
            f"[ERROR] CSV has no column {metric!r}. "
            f"Available: {', '.join(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    plot_df = df[["model", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    if len(plot_df) == 0:
        print(f"[ERROR] No numeric values in column {metric!r}.", file=sys.stderr)
        sys.exit(1)

    # For ms lower is better, for fps higher is better.
    ascending = "fps" not in metric.lower()
    plot_df = plot_df.sort_values(metric, ascending=ascending)

    out_png = _resolve_inference_plot_png(args.workspace, args.out_png, csv_path)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    plt.figure(figsize=(10, 6))
    x = range(len(plot_df))
    vals = plot_df[metric].tolist()
    bars = plt.bar(x, vals, tick_label=plot_df["model"].tolist())
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(metric)
    plt.title(f"Inference benchmark")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Numeric labels above bars.
    ymax = max(vals) if vals else 0.0
    y_pad = ymax * 0.015 if ymax > 0 else 0.01
    for bar, v in zip(bars, vals):
        x_text = bar.get_x() + bar.get_width() / 2.0
        y_text = bar.get_height()
        plt.text(
            x_text,
            y_text + y_pad,
            f"{float(v):.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[OK] Bar chart: {out_png}")


def _resolve_test_metrics_plot_png(
    workspace_cli: str | None,
    out_dir_cli: str | None,
    runs_group_dir: str,
    metric: str,
) -> str:
    if out_dir_cli:
        base = os.path.abspath(os.path.expanduser(out_dir_cli))
    else:
        try:
            ws = resolve_workspace_root(workspace_cli)
            base = os.path.join(WorkspaceLayout(ws).analytics, "metrics_comparison")
        except ValueError:
            base = os.path.join(
                os.path.dirname(os.path.abspath(runs_group_dir)),
                "analytics",
                "metrics_comparison",
            )
    os.makedirs(base, exist_ok=True)
    ds_name = os.path.basename(os.path.normpath(runs_group_dir))
    safe_metric = re.sub(r"[^\w.\-+]+", "_", metric, flags=re.UNICODE).strip("._")
    if not safe_metric:
        safe_metric = "metric"
    return os.path.join(base, f"test_metrics_{ds_name}_{safe_metric}.png")


def cmd_test_metrics_plot(args: argparse.Namespace) -> None:
    if (not getattr(args, "runs_group_dir", None) or not getattr(args, "metrics", None)) and sys.stdin.isatty():
        args.runs_group_dir = prompt_text("Runs group dir", default=str(args.models_root)).strip() or str(args.models_root)
        raw_metrics = prompt_text("Metrics (comma separated)", default="mAP50-95,Box-F1").strip()
        args.metrics = [m.strip() for m in raw_metrics.split(",") if m.strip()]
    if not getattr(args, "runs_group_dir", None) or not getattr(args, "metrics", None):
        print("[ERROR] Incomplete arguments: --runs-group-dir and --metrics are required.", file=sys.stderr)
        sys.exit(2)
    runs_group_dir = os.path.abspath(os.path.expanduser(args.runs_group_dir))
    if not os.path.isdir(runs_group_dir):
        print(f"[ERROR] Models directory not found: {runs_group_dir}", file=sys.stderr)
        sys.exit(1)

    run_dirs = _resolve_selected_run_dirs(
        runs_group_dir,
        getattr(args, "selected_run_dirs", None),
    )
    if not run_dirs:
        print(f"[ERROR] No run directories found for scope in: {runs_group_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Test-metrics scope: {len(run_dirs)} run(s) selected")

    recompute_enabled = bool(getattr(args, "recompute_missing_metrics", False))
    run_data_yaml_map = getattr(args, "run_data_yaml_map", None)
    if not isinstance(run_data_yaml_map, dict):
        run_data_yaml_map = {}
    rows: list[dict[str, Any]] = []
    run_rows: dict[str, dict[str, Any]] = {}
    metric_sources: dict[str, dict[str, str]] = {}
    recompute_status_by_run: dict[str, str] = {}
    for run_dir in run_dirs:
        model_name = os.path.basename(run_dir.rstrip(os.sep))
        tm = latest_test_metrics_path(run_dir)
        row = {"model": model_name, "run_dir": run_dir}
        if tm:
            try:
                df = pd.read_csv(tm)
                df.columns = [str(c).strip() for c in df.columns]
                if len(df) > 0:
                    row.update(df.iloc[0].to_dict())
                else:
                    print(f"[WARN] {model_name}: empty CSV {tm}")
            except Exception as e:
                print(f"[WARN] {model_name}: failed to read {tm}: {e}")
        elif not recompute_enabled:
            print(f"[WARN] {model_name}: missing test_metrics*.csv")
        rows.append(row)
        run_rows[run_dir] = row
        metric_sources[run_dir] = {}

    if not rows:
        print("[ERROR] No test_metrics data available for plotting.", file=sys.stderr)
        sys.exit(1)

    requested_metrics = [m.strip() for m in args.metrics if m.strip()]
    if not requested_metrics:
        print("[ERROR] --metrics list is empty.", file=sys.stderr)
        sys.exit(1)

    if recompute_enabled:
        split = str(getattr(args, "recompute_split", "test") or "test")
        for run_dir in run_dirs:
            row = run_rows.get(run_dir)
            if row is None:
                continue
            missing_for_run = [
                m for m in requested_metrics
                if m not in row or pd.isna(pd.to_numeric(row.get(m), errors="coerce"))
            ]
            if not missing_for_run:
                for m in requested_metrics:
                    metric_sources.setdefault(run_dir, {})[m] = "original"
                continue
            data_yaml = str(run_data_yaml_map.get(run_dir) or "").strip() or _resolve_data_yaml_for_run(run_dir, args.workspace)[0]
            if not data_yaml:
                print(f"[WARN] {os.path.basename(run_dir)}: no data.yaml detected, cannot recompute {missing_for_run}")
                for m in missing_for_run:
                    metric_sources.setdefault(run_dir, {})[m] = "missing"
                recompute_status_by_run[run_dir] = "skipped_no_data_yaml"
                _save_recompute_status(
                    run_dir,
                    data_yaml=os.path.join(run_dir, "_missing_data_yaml_"),
                    split=split,
                    requested_metrics=requested_metrics,
                    resolved=[],
                    unresolved=missing_for_run,
                    status="missing_data_yaml",
                )
                continue
            cached_status = _load_recompute_status(run_dir, data_yaml, split, requested_metrics)
            if cached_status and isinstance(cached_status, dict):
                unresolved_prev = set(cached_status.get("unresolved_metrics") or [])
                if unresolved_prev and set(missing_for_run).issubset(unresolved_prev):
                    for m in missing_for_run:
                        metric_sources.setdefault(run_dir, {})[m] = "missing"
                    recompute_status_by_run[run_dir] = "skipped_known_unresolved"
                    continue
            try:
                recomputed_csv = os.path.join(run_dir, "test_metrics_recomputed.csv")
                if os.path.isfile(recomputed_csv):
                    rdf = pd.read_csv(recomputed_csv)
                    recomputed = rdf.iloc[0].to_dict() if len(rdf) > 0 else {}
                    fp_metrics = compute_fingerprint(
                        {
                            "tool": "analyze-v2",
                            "task": "metrics_recompute",
                            "split": split,
                            "data_yaml_hash": data_yaml_hash(data_yaml),
                            "weights_hash": weights_hash(run_dir),
                        }
                    )
                    cache_metrics_csv = os.path.join(run_cache_root(run_dir), "metrics", f"recomputed_{fp_metrics}.csv")
                    os.makedirs(os.path.dirname(cache_metrics_csv), exist_ok=True)
                    if len(rdf) > 0 and not os.path.isfile(cache_metrics_csv):
                        rdf.to_csv(cache_metrics_csv, index=False, encoding="utf-8")
                    append_cache_entry(
                        run_dir,
                        {
                            "artifact": "metrics.recomputed",
                            "fingerprint": fp_metrics,
                            "path": os.path.relpath(cache_metrics_csv, run_dir),
                            "status": "hit",
                        },
                    )
                else:
                    recomputed = _recompute_run_test_metrics(
                        run_dir,
                        data_yaml,
                        split,
                        val_batch=int(getattr(args, "val_batch", 1)),
                        val_imgsz=int(getattr(args, "val_imgsz", 640)),
                        val_half=bool(getattr(args, "val_half", True)),
                        gpu_only=bool(getattr(args, "gpu_only_val", True)),
                    )
                    fp_metrics = compute_fingerprint(
                        {
                            "tool": "analyze-v2",
                            "task": "metrics_recompute",
                            "split": split,
                            "data_yaml_hash": data_yaml_hash(data_yaml),
                            "weights_hash": weights_hash(run_dir),
                        }
                    )
                    cache_metrics_csv = os.path.join(run_cache_root(run_dir), "metrics", f"recomputed_{fp_metrics}.csv")
                    os.makedirs(os.path.dirname(cache_metrics_csv), exist_ok=True)
                    if recomputed:
                        pd.DataFrame([recomputed]).to_csv(cache_metrics_csv, index=False, encoding="utf-8")
                    append_cache_entry(
                        run_dir,
                        {
                            "artifact": "metrics.recomputed",
                            "fingerprint": fp_metrics,
                            "path": os.path.relpath(cache_metrics_csv, run_dir),
                            "status": "miss",
                        },
                    )
            except Exception as e:
                print(f"[WARN] {os.path.basename(run_dir)}: recompute failed: {e}")
                for m in missing_for_run:
                    metric_sources.setdefault(run_dir, {})[m] = "missing"
                recompute_status_by_run[run_dir] = "error"
                _save_recompute_status(
                    run_dir,
                    data_yaml=data_yaml,
                    split=split,
                    requested_metrics=requested_metrics,
                    resolved=[],
                    unresolved=missing_for_run,
                    status="error",
                )
                continue
            if not recomputed:
                print(f"[WARN] {os.path.basename(run_dir)}: recompute produced no metrics")
                for m in missing_for_run:
                    metric_sources.setdefault(run_dir, {})[m] = "missing"
                recompute_status_by_run[run_dir] = "no_metrics"
                _save_recompute_status(
                    run_dir,
                    data_yaml=data_yaml,
                    split=split,
                    requested_metrics=requested_metrics,
                    resolved=[],
                    unresolved=missing_for_run,
                    status="no_metrics",
                )
                continue
            resolved_now: list[str] = []
            unresolved_now: list[str] = []
            for m in missing_for_run:
                if m in recomputed and pd.notna(pd.to_numeric(recomputed.get(m), errors="coerce")):
                    row[m] = recomputed[m]
                    metric_sources.setdefault(run_dir, {})[m] = "recomputed"
                    resolved_now.append(m)
                else:
                    metric_sources.setdefault(run_dir, {})[m] = "missing"
                    unresolved_now.append(m)
            for m in requested_metrics:
                metric_sources.setdefault(run_dir, {}).setdefault(m, "original")
            row["metrics_source"] = "recomputed"
            recompute_status_by_run[run_dir] = "recomputed"
            print(f"[INFO] {os.path.basename(run_dir)}: recomputed missing metrics from {data_yaml}")
            _save_recompute_status(
                run_dir,
                data_yaml=data_yaml,
                split=split,
                requested_metrics=requested_metrics,
                resolved=resolved_now,
                unresolved=unresolved_now,
                status="ok" if not unresolved_now else "partial",
            )
        all_df = pd.DataFrame(list(run_rows.values()))
    else:
        for run_dir, row in run_rows.items():
            for m in requested_metrics:
                val = row.get(m)
                metric_sources.setdefault(run_dir, {})[m] = (
                    "original" if pd.notna(pd.to_numeric(val, errors="coerce")) else "missing"
                )

    all_df = pd.DataFrame(list(run_rows.values()))
    for metric in requested_metrics:
        if metric not in all_df.columns:
            all_df[metric] = np.nan
        plot_df = all_df[["model", metric]].copy()
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
        plot_df = plot_df.dropna(subset=[metric])
        if len(plot_df) == 0:
            print(f"[WARN] Metric {metric!r}: no numeric values, skipping")
            continue
        if len(plot_df) < 2:
            print(f"[WARN] Metric {metric!r}: only one run with numeric value, skipping comparison chart")
            continue

        plot_df = plot_df.sort_values(metric, ascending=False)
        vals = plot_df[metric].tolist()
        out_png = _resolve_test_metrics_plot_png(
            args.workspace, args.out_dir, runs_group_dir, metric
        )

        plt.figure(figsize=(10, 6))
        x = range(len(plot_df))
        bars = plt.bar(x, vals, tick_label=plot_df["model"].tolist())
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(metric)
        plt.title(f"Test Metrics Comparison")
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)

        ymax = max(vals) if vals else 0.0
        y_pad = ymax * 0.015 if ymax > 0 else 0.01
        for bar, v in zip(bars, vals):
            x_text = bar.get_x() + bar.get_width() / 2.0
            y_text = bar.get_height()
            plt.text(
                x_text,
                y_text + y_pad,
                f"{float(v):.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(out_png, dpi=220)
        plt.close()
        print(f"[OK] Test-metrics chart ({metric}): {out_png}")

    sources_out = str(getattr(args, "metric_sources_out", "") or "").strip()
    if sources_out:
        out_path = os.path.abspath(os.path.expanduser(sources_out))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        selected_scope = [os.path.abspath(d) for d in run_dirs]
        recomputed_runs = sorted(
            [
                run_dir
                for run_dir, by_metric in metric_sources.items()
                if any(str(v) == "recomputed" for v in (by_metric or {}).values())
            ]
        )
        payload = {
            "requested_metrics": requested_metrics,
            "scope": {
                "mode": "selected_runs" if bool(getattr(args, "selected_run_dirs", None)) else "runs_group",
                "runs_group_dir": runs_group_dir,
                "selected_run_dirs": selected_scope,
            },
            "recomputed_runs": recomputed_runs,
            "sources": metric_sources,
            "recompute_status_by_run": recompute_status_by_run,
            "run_data_yaml_map": {k: v for k, v in run_data_yaml_map.items() if isinstance(v, str) and v.strip()},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] Metric sources: {out_path}")


def cmd_all(args: argparse.Namespace) -> None:
    interactive_mode = sys.stdin.isatty()
    baseline = str(getattr(args, "baseline", "") or "").strip()
    others = [str(x).strip() for x in (getattr(args, "others", []) or []) if str(x).strip()]
    profile = str(getattr(args, "profile", "") or "").strip().lower()
    if not baseline or profile not in {"quality", "speed", "full"}:
        if not interactive_mode:
            print(
                "[ERROR] Non-interactive `smartrain analyze all` requires --baseline and --profile.",
                file=sys.stderr,
            )
            sys.exit(2)
        indexed = _filtered_run_records(args)
        if len(indexed) < 2:
            print("[ERROR] Need at least two runs for full analysis.")
            sys.exit(1)
        runs_root = os.path.abspath(str(args.models_root))

        def _display_run_dir(path: str) -> str:
            ap = os.path.abspath(path)
            try:
                rel = os.path.relpath(ap, runs_root)
                if not rel.startswith(".."):
                    return rel
            except Exception:
                pass
            return ap

        print(f"{'#':>4}  {'model':<14}  {'dataset':<24}  {'run_dir (relative to runs root)'}")
        print("-" * 120)
        for i, (rd, rec) in enumerate(indexed, start=1):
            print(
                f"{i:4d}  {str(rec.model or '?')[:14]:<14}  "
                + f"{str(rec.dataset_name or '?')[:24]:<24}  {_display_run_dir(rd)}"
            )
        baseline_idx = prompt_int("Baseline run number", default=1)
        others_raw = prompt_text("Other run numbers (comma-separated)", default="").strip()
        try:
            others_idx = [int(x.strip()) for x in others_raw.split(",") if x.strip()]
        except ValueError:
            print("[ERROR] Invalid run numbers.")
            sys.exit(1)
        if baseline_idx < 1 or baseline_idx > len(indexed):
            print("[ERROR] Baseline index out of range.")
            sys.exit(1)
        baseline = indexed[baseline_idx - 1][0]
        others = [indexed[i - 1][0] for i in others_idx if 1 <= i <= len(indexed) and indexed[i - 1][0] != baseline]
        profile = prompt_choice("Profile", ["quality", "speed", "full"], default="full")
    report_languages_raw = str(getattr(args, "report_languages", "ru,en") or "ru,en")
    report_languages = [x.strip() for x in report_languages_raw.split(",") if x.strip()]
    if not report_languages:
        report_languages = ["ru", "en"]
    data_yaml = str(getattr(args, "data_yaml", "") or "").strip()
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
    selected_run_dirs = [baseline] + others
    run_data_yaml_map, run_data_yaml_source, unresolved_data_yaml_runs = _build_run_data_yaml_map(
        selected_run_dirs,
        args.workspace,
        preferred_split="test" if profile in ("speed", "full") else None,
    )
    unique_data_yaml = sorted(set(run_data_yaml_map.values()))
    if profile in ("speed", "full"):
        if data_yaml:
            for rd in selected_run_dirs:
                run_data_yaml_map.setdefault(rd, data_yaml)
            unique_data_yaml = sorted(set(run_data_yaml_map.values()))
        elif interactive_mode and len(unique_data_yaml) > 1:
            print("[INFO] Multiple datasets detected across selected runs.")
            for rd in selected_run_dirs:
                dy = run_data_yaml_map.get(rd)
                src = run_data_yaml_source.get(rd, "unknown")
                print(f"[INFO]  - {os.path.basename(rd.rstrip(os.sep))}: {dy or 'UNRESOLVED'} (source: {src})")
            mode = prompt_choice(
                "Data.yaml mode",
                ["auto_per_run", "single_shared"],
                default="auto_per_run",
                show_options=False,
            )
            if mode == "single_shared":
                auto_yaml = _auto_select_data_yaml(baseline, others, args.workspace, preferred_split="test")
                if auto_yaml:
                    data_yaml = auto_yaml
                    for rd in selected_run_dirs:
                        run_data_yaml_map[rd] = data_yaml
                    unique_data_yaml = [data_yaml]
        elif not data_yaml and len(unique_data_yaml) == 1:
            data_yaml = unique_data_yaml[0]
        elif not data_yaml and interactive_mode and not run_data_yaml_map:
            data_yaml = prompt_text("Path to data.yaml (required for speed/full)", default="").strip()
            if data_yaml:
                for rd in selected_run_dirs:
                    run_data_yaml_map[rd] = data_yaml
                unique_data_yaml = [data_yaml]
        if not data_yaml and not run_data_yaml_map and not interactive_mode:
            print(
                "[ERROR] No data.yaml resolved for selected runs; use --data-yaml or ensure metadata/runtime yaml is present.",
                file=sys.stderr,
            )
            sys.exit(2)
    selected_labels = [os.path.basename(x.rstrip(os.sep)) for x in selected_run_dirs]
    print("[INFO] Selected compare runs:")
    for idx, (run_dir, label) in enumerate(zip(selected_run_dirs, selected_labels), start=1):
        role = "baseline" if idx == 1 else "other"
        print(f"[INFO]  - {role}: {label} ({run_dir})")

    if others:
        compare_csv = os.path.join(session_root, "artifacts", "compare", "compare_delta.csv")
        compare_png = os.path.join(session_root, "artifacts", "compare", "compare_curves.png")
        compare_insights = os.path.join(session_root, "artifacts", "compare", "compare_insights.txt")
        cmp_ns = argparse.Namespace(
            baseline=baseline,
            others=others,
            out_csv=compare_csv,
            out_png=compare_png,
            out_insights=compare_insights,
            metric_column=DEFAULT_MAP_COL,
            workspace=args.workspace,
            analytics_session=args.analytics_session,
            models_root=args.models_root,
        )
        cmd_compare(cmp_ns)
        artifacts.extend(
            [
                {"role": "compare_csv", "path": os.path.relpath(compare_csv, session_root)},
                {"role": "compare_png", "path": os.path.relpath(compare_png, session_root)},
                {"role": "compare_insights", "path": os.path.relpath(compare_insights, session_root)},
            ]
        )
    else:
        print("[INFO] No candidate runs selected: compare artifacts are skipped (single-run report mode).")

    exp_csv = os.path.join(session_root, "artifacts", "table", "runs_summary.csv")
    exp_ns = argparse.Namespace(
        output=exp_csv,
        workspace=args.workspace,
        models_root=args.models_root,
        analytics_session=None,
    )
    cmd_export_table(exp_ns)
    artifacts.append({"role": "summary_csv", "path": os.path.relpath(exp_csv, session_root)})
    sys_profile_csv = os.path.join(session_root, "artifacts", "table", "system_profile_compare.csv")
    written_sys_profile = _write_system_profile_compare_csv([baseline] + others, sys_profile_csv)
    if written_sys_profile:
        artifacts.append(
            {"role": "system_profile_compare_csv", "path": os.path.relpath(sys_profile_csv, session_root)}
        )
    test_sys_profile_csv = os.path.join(session_root, "artifacts", "table", "test_system_profile_compare.csv")
    written_test_sys_profile = _write_test_system_profile_compare_csv([baseline] + others, test_sys_profile_csv)
    if written_test_sys_profile:
        artifacts.append(
            {"role": "test_system_profile_compare_csv", "path": os.path.relpath(test_sys_profile_csv, session_root)}
        )

    lb_csv = os.path.join(session_root, "artifacts", "leaderboard", "leaderboard.csv")
    lb_ns = argparse.Namespace(
        out_csv=lb_csv,
        selected_run_dirs=selected_run_dirs,
        quality_metric="mAP50-95",
        speed_metric="avg_inference_fps",
        weight_quality=0.6,
        weight_speed=0.25,
        weight_stability=0.15,
        workspace=args.workspace,
        models_root=args.models_root,
        analytics_session=args.analytics_session,
    )
    cmd_leaderboard(lb_ns)
    artifacts.append({"role": "leaderboard_csv", "path": os.path.relpath(lb_csv, session_root)})

    runs_group_dir = os.path.dirname(baseline)
    metric_sources_payload: dict[str, Any] | None = None
    recompute_missing_metrics = True
    if profile in ("quality", "full"):
        metric_sources_json = os.path.join(session_root, "artifacts", "metrics", "metric_sources.json")
        recompute_plan = _collect_missing_metrics_recompute_plan(
            [baseline] + others,
            ["mAP50-95", "Box-F1"],
            data_yaml=(data_yaml or None),
            run_data_yaml_map=run_data_yaml_map,
            workspace=args.workspace,
            split="test",
        )
        missing_runs = [str(x.get("run_dir")) for x in recompute_plan.get("recompute", []) if x.get("run_dir")]
        skipped_runs = recompute_plan.get("skipped", [])
        if missing_runs or skipped_runs:
            print("[INFO] Missing metrics recompute plan:")
            if missing_runs:
                print("[INFO]  Recompute candidates:")
                for item in recompute_plan.get("recompute", []):
                    rd = str(item.get("run_dir") or "")
                    mm = list(item.get("missing_metrics") or [])
                    dy = str(item.get("data_yaml") or "")
                    print(
                        "[INFO]   - "
                        + os.path.basename(rd.rstrip(os.sep))
                        + f": missing={mm}, data_yaml={dy}"
                    )
            if skipped_runs:
                print("[INFO]  Skipped candidates:")
                for item in skipped_runs:
                    rd = str(item.get("run_dir") or "")
                    mm = list(item.get("missing_metrics") or [])
                    reason = str(item.get("reason") or "unknown")
                    print(
                        "[INFO]   - "
                        + os.path.basename(rd.rstrip(os.sep))
                        + f": missing={mm}, reason={reason}"
                    )
        if missing_runs:
            recompute_choice = str(getattr(args, "recompute_missing_metrics_choice", "") or "").strip().lower()
            if recompute_choice in {"yes", "no"}:
                recompute_missing_metrics = recompute_choice == "yes"
                print(
                    "[INFO] Missing metrics recompute is "
                    + ("enabled" if recompute_missing_metrics else "disabled")
                    + f" by --recompute-missing-metrics={recompute_choice}."
                )
            else:
                recompute_missing_metrics = True
                print(
                    "[INFO] Found missing metrics in "
                    + f"{len(missing_runs)} run(s): auto-recompute is enabled."
                )
        tm_ns = argparse.Namespace(
            runs_group_dir=runs_group_dir,
            selected_run_dirs=selected_run_dirs,
            metrics=["mAP50-95", "Box-F1"],
            out_dir=os.path.join(session_root, "artifacts", "metrics"),
            workspace=args.workspace,
            models_root=args.models_root,
            analytics_session=args.analytics_session,
            recompute_missing_metrics=recompute_missing_metrics,
            recompute_split="test",
            metric_sources_out=metric_sources_json,
            val_batch=int(getattr(args, "val_batch", 1)),
            val_imgsz=int(getattr(args, "val_imgsz", 640)),
            val_half=bool(getattr(args, "val_half", True)),
            gpu_only_val=bool(getattr(args, "gpu_only_val", True)),
            run_data_yaml_map=run_data_yaml_map,
        )
        cmd_test_metrics_plot(tm_ns)
        artifacts.append({"role": "metrics_dir", "path": os.path.relpath(tm_ns.out_dir, session_root)})
        artifacts.append({"role": "metric_sources", "path": os.path.relpath(metric_sources_json, session_root)})
        if os.path.isfile(metric_sources_json):
            try:
                with open(metric_sources_json, "r", encoding="utf-8") as f:
                    metric_sources_payload = json.load(f)
            except Exception:
                metric_sources_payload = None

    if profile in ("speed", "full"):
        run_groups, unresolved_for_speed = _group_runs_by_data_yaml(selected_run_dirs, run_data_yaml_map)
        if unresolved_for_speed:
            print("[WARN] Speed stage: skipped runs without resolved data.yaml:")
            for rd in unresolved_for_speed:
                print(f"[WARN]  - {os.path.basename(rd.rstrip(os.sep))}")
                _record_failure(
                    stage="speed",
                    status="skipped",
                    reason_code="no_data_yaml",
                    reason_detail="run excluded from speed stage due to unresolved data.yaml",
                    run_dir=rd,
                    split="test",
                )
        if not run_groups:
            print("[WARN] Speed stage skipped: no runs with resolved data.yaml.")
        inf_csv = os.path.join(session_root, "artifacts", "inference", "benchmark.csv")
        inf_png = os.path.join(session_root, "artifacts", "inference", "benchmark_bars.png")
        os.makedirs(os.path.dirname(inf_csv), exist_ok=True)
        inf_parts: list[pd.DataFrame] = []
        for g_idx, (group_yaml, group_runs) in enumerate(sorted(run_groups.items()), start=1):
            inf_part_csv = os.path.join(session_root, "artifacts", "inference", f"benchmark_group_{g_idx}.csv")
            ib_ns = argparse.Namespace(
                runs_group_dir=runs_group_dir,
                selected_run_dirs=group_runs,
                data_yaml=group_yaml,
                split="test",
                frames=100,
                device="cpu",
                half=False,
                out_csv=inf_part_csv,
                workspace=args.workspace,
                models_root=args.models_root,
                analytics_session=args.analytics_session,
                reuse_run_cache=True,
                cache_stats_out=os.path.join(session_root, "artifacts", "inference", f"cache_stats_group_{g_idx}.json"),
            )
            cmd_inference_benchmark(ib_ns)
            if os.path.isfile(inf_part_csv):
                try:
                    part_df = pd.read_csv(inf_part_csv)
                    part_df["dataset_yaml_used"] = group_yaml
                    if "run_name" not in part_df.columns:
                        part_df["run_name"] = part_df.get("run_dir", pd.Series(dtype=str)).astype(str).map(
                            lambda p: os.path.basename(str(p).rstrip(os.sep))
                        )
                    if "benchmark_status" not in part_df.columns:
                        part_df["benchmark_status"] = "ok"
                    inf_parts.append(part_df)
                except Exception as e:
                    _record_failure(
                        stage="speed",
                        status="failed",
                        reason_code="benchmark_group_read_failed",
                        reason_detail=str(e),
                        split="test",
                    )
        if inf_parts:
            inf_df = pd.concat(inf_parts, ignore_index=True)
            present = {
                os.path.abspath(str(p))
                for p in inf_df.get("run_dir", pd.Series(dtype=str)).astype(str).tolist()
                if str(p).strip()
            }
            present_by_name: dict[str, str] = {}
            for _, row in inf_df.iterrows():
                rname = str(row.get("run_name") or "").strip()
                if not rname:
                    continue
                status = str(row.get("benchmark_status") or "ok").strip()
                current = present_by_name.get(rname)
                if current is None or (current != "ok" and status == "ok"):
                    present_by_name[rname] = status
            for run_dir in selected_run_dirs:
                rd = os.path.abspath(run_dir)
                run_name = os.path.basename(run_dir.rstrip(os.sep))
                if rd in present:
                    continue
                if present_by_name.get(run_name) == "ok":
                    _record_failure(
                        stage="speed",
                        status="missing",
                        reason_code="run_dir_mismatch",
                        reason_detail="benchmark row matched by run_name but run_dir differs",
                        run_dir=run_dir,
                        split="test",
                    )
                    continue
                inf_df = pd.concat(
                    [
                        inf_df,
                        pd.DataFrame(
                            [
                                {
                                    "model": os.path.basename(run_dir.rstrip(os.sep)),
                                    "run_name": os.path.basename(run_dir.rstrip(os.sep)),
                                    "run_dir": run_dir,
                                    "dataset_yaml_used": run_data_yaml_map.get(run_dir, ""),
                                    "benchmark_status": "missing_or_failed",
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
                _record_failure(
                    stage="speed",
                    status="missing",
                    reason_code="benchmark_missing_or_failed",
                    reason_detail="benchmark row was not produced for selected run",
                    run_dir=run_dir,
                    split="test",
                )
            inf_df.to_csv(inf_csv, index=False, encoding="utf-8")
        else:
            inf_df = pd.DataFrame(
                [
                    {
                        "model": os.path.basename(run_dir.rstrip(os.sep)),
                        "run_name": os.path.basename(run_dir.rstrip(os.sep)),
                        "run_dir": run_dir,
                        "dataset_yaml_used": run_data_yaml_map.get(run_dir, ""),
                        "benchmark_status": "missing_or_failed",
                    }
                    for run_dir in selected_run_dirs
                ]
            )
            inf_df.to_csv(inf_csv, index=False, encoding="utf-8")
        if os.path.isfile(lb_csv) and os.path.isfile(inf_csv):
            try:
                lb_df = pd.read_csv(lb_csv)
                inf_df = pd.read_csv(inf_csv)
                if "run_dir" in lb_df.columns and "run_dir" in inf_df.columns:
                    speed_series = (
                        inf_df.assign(
                            speed_metric=pd.to_numeric(inf_df.get("avg_inference_fps"), errors="coerce")
                        )
                        .dropna(subset=["run_dir", "speed_metric"])
                        .groupby("run_dir", as_index=True)["speed_metric"]
                        .max()
                    )
                    if not speed_series.empty:
                        existing_speed = pd.to_numeric(lb_df.get("speed_metric"), errors="coerce")
                        direct_speed = lb_df["run_dir"].map(speed_series)
                        bench_by_name = {
                            os.path.basename(str(k).rstrip(os.sep)): float(v)
                            for k, v in speed_series.items()
                        }
                        by_name_speed = lb_df["run_dir"].astype(str).map(
                            lambda p: bench_by_name.get(os.path.basename(str(p).rstrip(os.sep)))
                        )
                        lb_df["speed_metric"] = direct_speed.combine_first(by_name_speed).combine_first(existing_speed)
                        if "quality_metric" in lb_df.columns:
                            qv = pd.to_numeric(lb_df.get("quality_metric"), errors="coerce")
                            sv = pd.to_numeric(lb_df.get("speed_metric"), errors="coerce")
                            stable = (
                                pd.Series([1.0] * len(lb_df))
                                if "training_ok" not in lb_df.columns and "testing_ok" not in lb_df.columns
                                else (
                                    pd.to_numeric(lb_df.get("training_ok"), errors="coerce").fillna(0.0)
                                    * pd.to_numeric(lb_df.get("testing_ok"), errors="coerce").fillna(0.0)
                                )
                            )
                            speed_component = sv.where(sv.isna(), sv)
                            denom = 0.6 + 0.25 + 0.15
                            lb_df["composite_score"] = (
                                (0.6 * qv.fillna(0.0)) + (0.25 * speed_component.fillna(0.0)) + (0.15 * stable.fillna(0.0))
                            ) / denom
                        lb_df = lb_df.sort_values("composite_score", ascending=False)
                        lb_df.to_csv(lb_csv, index=False, encoding="utf-8")
            except Exception:
                _record_failure(
                    stage="speed",
                    status="failed",
                    reason_code="leaderboard_speed_merge_failed",
                    reason_detail="failed to merge speed benchmark into leaderboard",
                    split="test",
                )
        ip_ns = argparse.Namespace(
            csv=inf_csv,
            metric="avg_inference_ms_per_frame",
            out_png=inf_png,
            workspace=args.workspace,
            models_root=args.models_root,
            analytics_session=args.analytics_session,
        )
        cmd_inference_plot(ip_ns)
        artifacts.extend(
            [
                {"role": "inference_csv", "path": os.path.relpath(inf_csv, session_root)},
                {"role": "inference_png", "path": os.path.relpath(inf_png, session_root)},
            ]
        )
        for g_idx in range(1, len(run_groups) + 1):
            cache_stats_path = os.path.join(session_root, "artifacts", "inference", f"cache_stats_group_{g_idx}.json")
            if os.path.isfile(cache_stats_path):
                try:
                    cache_events.extend(json.load(open(cache_stats_path, "r", encoding="utf-8")).get("cache", []))
                except Exception as e:
                    _record_failure(
                        stage="speed",
                        status="failed",
                        reason_code="cache_stats_read_failed",
                        reason_detail=str(e),
                        split="test",
                    )
        if os.path.isfile(inf_csv):
            try:
                if os.path.getsize(inf_csv) > 0:
                    sq = _write_speed_quality_artifacts(
                        session_root,
                        inf_csv,
                        [baseline] + others,
                        metric_sources_payload,
                        scatter_x=str(getattr(args, "scatter_x", "avg_inference_ms_per_frame")),
                        scatter_y=str(getattr(args, "scatter_y", "mAP50-95")),
                        run_data_yaml_map=run_data_yaml_map,
                    )
                    if sq:
                        artifacts.extend(
                            [
                                {"role": "speed_quality_csv", "path": sq["csv"]},
                                {"role": "speed_quality_png", "path": sq["png"]},
                            ]
                        )
                        try:
                            sq_abs = os.path.join(session_root, sq["csv"])
                            sq_df = pd.read_csv(sq_abs)
                            expected = {
                                os.path.basename(str(r).rstrip(os.sep))
                                for r in selected_run_dirs
                            }
                            actual = set(sq_df.get("model", pd.Series(dtype=str)).astype(str).tolist())
                            if len(actual) < len(expected):
                                _record_failure(
                                    stage="speed_quality",
                                    status="failed",
                                    reason_code="png_incomplete_series",
                                    reason_detail=f"speed_quality models={sorted(actual)} expected={sorted(expected)}",
                                    split="test",
                                )
                        except Exception as e:
                            _record_failure(
                                stage="speed_quality",
                                status="failed",
                                reason_code="speed_quality_validation_failed",
                                reason_detail=str(e),
                                split="test",
                            )
            except Exception:
                _record_failure(
                    stage="speed_quality",
                    status="failed",
                    reason_code="speed_quality_write_failed",
                    reason_detail="failed to build speed-quality artifacts",
                    split="test",
                )

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
