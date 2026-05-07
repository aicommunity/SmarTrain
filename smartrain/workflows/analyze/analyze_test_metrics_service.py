from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_test_metrics_plot_png(
    workspace_cli: str | None,
    out_dir_cli: str | None,
    runs_group_dir: str,
    metric: str,
    *,
    resolve_workspace_root_cb,
    workspace_layout_cls,
) -> str:
    if out_dir_cli:
        base = os.path.abspath(os.path.expanduser(out_dir_cli))
    else:
        try:
            workspace = resolve_workspace_root_cb(workspace_cli)
            base = os.path.join(workspace_layout_cls(workspace).analytics, "metrics_comparison")
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


def run_test_metrics_plot(
    args: argparse.Namespace,
    *,
    prompt_text_cb,
    resolve_selected_run_dirs_cb,
    latest_test_metrics_path_cb,
    load_recompute_status_cb,
    save_recompute_status_cb,
    resolve_data_yaml_for_run_cb,
    recompute_run_test_metrics_cb,
    compute_fingerprint_cb,
    run_cache_root_cb,
    data_yaml_hash_cb,
    append_cache_entry_cb,
    resolve_workspace_root_cb,
    workspace_layout_cls,
) -> None:
    if (not getattr(args, "runs_group_dir", None) or not getattr(args, "metrics", None)) and sys.stdin.isatty():
        args.runs_group_dir = prompt_text_cb("Runs group dir", default=str(args.models_root)).strip() or str(args.models_root)
        raw_metrics = prompt_text_cb("Metrics (comma separated)", default="mAP50-95,Box-F1").strip()
        args.metrics = [m.strip() for m in raw_metrics.split(",") if m.strip()]
    if not getattr(args, "runs_group_dir", None) or not getattr(args, "metrics", None):
        print("[ERROR] Incomplete arguments: --runs-group-dir and --metrics are required.", file=sys.stderr)
        sys.exit(2)
    runs_group_dir = os.path.abspath(os.path.expanduser(args.runs_group_dir))
    if not os.path.isdir(runs_group_dir):
        print(f"[ERROR] Models directory not found: {runs_group_dir}", file=sys.stderr)
        sys.exit(1)

    run_dirs = resolve_selected_run_dirs_cb(runs_group_dir, getattr(args, "selected_run_dirs", None))
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
        tm = latest_test_metrics_path_cb(run_dir)

        row = {"model": model_name, "run_dir": run_dir}
        if tm:
            try:
                df = pd.read_csv(tm)
                df.columns = [str(c).strip() for c in df.columns]
                if len(df) > 0:
                    row.update(df.iloc[0].to_dict())
                else:
                    print(f"[WARN] {model_name}: empty CSV {tm}")
            except Exception as exc:
                print(f"[WARN] {model_name}: failed to read {tm}: {exc}")
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
            data_yaml = str(run_data_yaml_map.get(run_dir) or "").strip() or resolve_data_yaml_for_run_cb(run_dir, args.workspace)[0]
            if not data_yaml:
                print(f"[WARN] {os.path.basename(run_dir)}: no data.yaml detected, cannot recompute {missing_for_run}")
                for m in missing_for_run:
                    metric_sources.setdefault(run_dir, {})[m] = "missing"
                recompute_status_by_run[run_dir] = "skipped_no_data_yaml"
                save_recompute_status_cb(
                    run_dir,
                    data_yaml=os.path.join(run_dir, "_missing_data_yaml_"),
                    split=split,
                    requested_metrics=requested_metrics,
                    resolved=[],
                    unresolved=missing_for_run,
                    status="missing_data_yaml",
                )
                continue
            cached_status = load_recompute_status_cb(run_dir, data_yaml, split, requested_metrics)
            if cached_status and isinstance(cached_status, dict):
                unresolved_prev = set(cached_status.get("unresolved_metrics") or [])
                if unresolved_prev and set(missing_for_run).issubset(unresolved_prev):
                    for m in missing_for_run:
                        metric_sources.setdefault(run_dir, {})[m] = "missing"
                    recompute_status_by_run[run_dir] = "skipped_known_unresolved"
                    continue
            try:
                recomputed_csv = os.path.join(run_dir, "test_metrics_recomputed.csv")
                fp_metrics = compute_fingerprint_cb(
                    {
                        "tool": "analyze-v2",
                        "task": "test_metrics_recompute",
                        "split": split,
                        "val_batch": int(getattr(args, "val_batch", 1)),
                        "val_imgsz": int(getattr(args, "val_imgsz", 640)),
                        "val_half": bool(getattr(args, "val_half", True)),
                        "data_yaml_hash": data_yaml_hash_cb(data_yaml),
                    }
                )
                cache_metrics_csv = os.path.join(run_cache_root_cb(run_dir), "metrics", f"recomputed_{fp_metrics}.csv")
                if os.path.isfile(cache_metrics_csv):
                    cdf = pd.read_csv(cache_metrics_csv)
                    recomputed = cdf.iloc[0].to_dict() if len(cdf) else {}
                else:
                    recomputed = recompute_run_test_metrics_cb(
                        run_dir,
                        data_yaml=data_yaml,
                        split=split,
                        val_batch=int(getattr(args, "val_batch", 1)),
                        val_imgsz=int(getattr(args, "val_imgsz", 640)),
                        val_half=bool(getattr(args, "val_half", True)),
                        gpu_only=bool(getattr(args, "gpu_only_val", True)),
                    )
                    if recomputed:
                        recomputed = dict(recomputed)
                        recomputed["source"] = "recomputed"
                    os.makedirs(os.path.dirname(cache_metrics_csv), exist_ok=True)
                    pd.DataFrame([recomputed]).to_csv(cache_metrics_csv, index=False, encoding="utf-8")
                    append_cache_entry_cb(
                        run_dir,
                        {
                            "artifact": "metrics.recomputed",
                            "fingerprint": fp_metrics,
                            "path": os.path.relpath(cache_metrics_csv, run_dir),
                            "status": "miss",
                        },
                    )
                    try:
                        pd.DataFrame([recomputed]).to_csv(recomputed_csv, index=False, encoding="utf-8")
                    except Exception:
                        pass
                if not recomputed:
                    raise RuntimeError("recompute produced no metrics")
            except Exception as exc:
                print(f"[WARN] {os.path.basename(run_dir)}: recompute failed: {exc}")
                for m in missing_for_run:
                    metric_sources.setdefault(run_dir, {})[m] = "missing"
                recompute_status_by_run[run_dir] = "error"
                save_recompute_status_cb(
                    run_dir,
                    data_yaml=data_yaml,
                    split=split,
                    requested_metrics=requested_metrics,
                    resolved=[],
                    unresolved=missing_for_run,
                    status="error",
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
            save_recompute_status_cb(
                run_dir,
                data_yaml=data_yaml,
                split=split,
                requested_metrics=requested_metrics,
                resolved=resolved_now,
                unresolved=unresolved_now,
                status="ok" if not unresolved_now else "partial",
            )
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
        out_png = resolve_test_metrics_plot_png(
            args.workspace,
            args.out_dir,
            runs_group_dir,
            metric,
            resolve_workspace_root_cb=resolve_workspace_root_cb,
            workspace_layout_cls=workspace_layout_cls,
        )

        plt.figure(figsize=(10, 6))
        x = range(len(plot_df))
        bars = plt.bar(x, vals, tick_label=plot_df["model"].tolist())
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(metric)
        plt.title("Test Metrics Comparison")
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)

        ymax = max(vals) if vals else 0.0
        y_pad = ymax * 0.015 if ymax > 0 else 0.01
        for bar, value in zip(bars, vals):
            x_text = bar.get_x() + bar.get_width() / 2.0
            y_text = bar.get_height()
            plt.text(
                x_text,
                y_text + y_pad,
                f"{float(value):.4f}",
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
        with open(out_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        print(f"[OK] Metric sources: {out_path}")

