#!/usr/bin/env python3
"""
Scan training runs, summary CSVs, and compare metrics (CSV + PNG).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from glob import glob
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from smartrain.cli_argparse import CliArgumentParser
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root


DEFAULT_MAP_COL = "metrics/mAP50-95(B)"


def resolve_models_scan_root(workspace_cli: str | None, models_root_cli: str | None) -> str:
    """Explicit --models-root, otherwise workspace/runs, otherwise current directory."""
    if models_root_cli is not None:
        return os.path.abspath(os.path.expanduser(models_root_cli))
    try:
        ws = resolve_workspace_root(workspace_cli)
        return WorkspaceLayout(ws).runs
    except ValueError:
        return os.path.abspath(os.getcwd())


def find_run_directories(models_root: str) -> list[str]:
    runs: list[str] = []
    models_root = os.path.abspath(models_root)
    if not os.path.isdir(models_root):
        return runs
    for dirpath, _, filenames in os.walk(models_root):
        if "training_metadata.json" in filenames:
            runs.append(dirpath)
    return sorted(runs)


def load_metadata(run_dir: str) -> dict[str, Any]:
    path = os.path.join(run_dir, "training_metadata.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def latest_test_metrics_path(run_dir: str) -> str | None:
    candidates = sorted(glob(os.path.join(run_dir, "test_metrics*.csv")))
    return candidates[-1] if candidates else None


def results_csv_path(run_dir: str) -> str | None:
    p = os.path.join(run_dir, "train", "results.csv")
    return p if os.path.exists(p) else None


def flatten_metadata(md: dict[str, Any], run_dir: str) -> dict[str, Any]:
    row: dict[str, Any] = {"run_dir": run_dir}
    ti = md.get("training_info") or {}
    row["model"] = ti.get("model")
    row["dataset_name"] = (ti.get("dataset") or {}).get("name")
    row["dataset_hash"] = (ti.get("dataset") or {}).get("hash")
    hp = ti.get("hyperparameters") or {}
    row["epochs"] = hp.get("epochs")
    row["batch_size"] = hp.get("batch_size")
    row["train_image_size"] = hp.get("image_size")
    inf = md.get("inference") or {}
    row["val_imgsz"] = inf.get("imgsz")
    row["val_conf"] = inf.get("conf")
    row["val_iou"] = inf.get("iou")
    st = md.get("status") or {}
    row["training_ok"] = (st.get("training") or {}).get("success")
    row["testing_ok"] = (st.get("testing") or {}).get("success")
    ts = md.get("timestamps") or {}
    row["training_duration_s"] = (ts.get("training") or {}).get("duration_seconds")
    return row


def pick_map_column(df: pd.DataFrame) -> str | None:
    df.columns = [str(c).strip() for c in df.columns]
    for c in (DEFAULT_MAP_COL, "metrics/mAP50(B)"):
        if c in df.columns:
            return c
    for c in df.columns:
        if "mAP50" in c and "B" in c:
            return c
    return None


def cmd_scan(args: argparse.Namespace) -> None:
    runs = find_run_directories(args.models_root)
    if not runs:
        print("(runs with training_metadata.json not found)")
        return
    print(f"{'#':>4}  {'model':<14}  {'dataset':<24}  {'run_dir'}")
    print("-" * 100)
    for i, rd in enumerate(runs, start=1):
        try:
            md = load_metadata(rd)
            ti = md.get("training_info") or {}
            m = ti.get("model") or "?"
            ds = (ti.get("dataset") or {}).get("name") or "?"
            print(f"{i:4d}  {str(m)[:14]:<14}  {str(ds)[:24]:<24}  {rd}")
        except OSError as e:
            print(f"{i:4d} {'?':<14} {'?':<24} {rd} [error: {e}]")


def cmd_export_table(args: argparse.Namespace) -> None:
    runs = find_run_directories(args.models_root)
    out_path = args.output
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
    rows: list[dict[str, Any]] = []
    for rd in runs:
        try:
            md = load_metadata(rd)
        except OSError as e:
            print(f"[WARN] {rd}: {e}", file=sys.stderr)
            continue
        row = flatten_metadata(md, rd)
        tm = latest_test_metrics_path(rd)
        if tm:
            try:
                tdf = pd.read_csv(tm)
                tdf.columns = [str(c).strip() for c in tdf.columns]
                if len(tdf) > 0:
                    for col in tdf.columns:
                        row[f"test_{col}"] = tdf[col].iloc[0]
            except Exception as e:
                row["test_read_error"] = str(e)
        rc = results_csv_path(rd)
        if rc:
            try:
                rdf = pd.read_csv(rc)
                rdf.columns = [str(c).strip() for c in rdf.columns]
                mcol = pick_map_column(rdf)
                if mcol and "epoch" in rdf.columns and len(rdf) > 0:
                    last = rdf.iloc[-1]
                    row["train_last_epoch"] = last.get("epoch")
                    row[f"train_last_{mcol}"] = last.get(mcol)
            except Exception as e:
                row["train_read_error"] = str(e)
        rows.append(row)
    if not rows:
        print("[ERROR] No data to export.", file=sys.stderr)
        sys.exit(1)
    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Pivot table: {out_path} ({len(df)} rows)")
    if analytics_dir is not None:
        manifest = {
            "scan_root": args.models_root,
            "run_directories": runs,
            "summary_csv": out_path,
        }
        with open(os.path.join(analytics_dir, "session.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[OK] Session manifest: {os.path.join(analytics_dir, 'session.json')}")


def _finalize_compare_analytics_session(
    args: argparse.Namespace,
    baseline: str,
    others: list[str],
    out_csv: str,
    out_png: str,
    bar_path: str | None,
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
    print(f"[OK] Session manifest compare: {sj}")


def _read_test_metrics_row(run_dir: str) -> dict[str, Any]:
    tm = latest_test_metrics_path(run_dir)
    if not tm:
        return {}
    df = pd.read_csv(tm)
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) == 0:
        return {}
    return df.iloc[0].to_dict()


def cmd_compare(args: argparse.Namespace) -> None:
    baseline = os.path.abspath(args.baseline)
    others = [os.path.abspath(p) for p in args.others]
    bar_path: str | None = None
    all_runs = [baseline] + others
    for p in all_runs:
        if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "training_metadata.json")):
            print(f"[ERROR] Not run (no training_metadata.json): {p}", file=sys.stderr)
            sys.exit(1)

    base_metrics = _read_test_metrics_row(baseline)
    if not base_metrics:
        print("[WARN] The database does not have test_metrics*.csv - deltas only for train/results.csv", file=sys.stderr)

    delta_rows: list[dict[str, Any]] = []
    for other in others:
        om = _read_test_metrics_row(other)
        row: dict[str, Any] = {"baseline": baseline, "other": other}
        keys = set(base_metrics) | set(om)
        for k in keys:
            if k is None or str(k).strip() == "":
                continue
            try:
                bv = float(base_metrics[k]) if k in base_metrics and pd.notna(base_metrics.get(k)) else None
                ov = float(om[k]) if k in om and pd.notna(om.get(k)) else None
            except (TypeError, ValueError):
                continue
            if bv is not None and ov is not None:
                row[f"delta_{k}"] = ov - bv
        delta_rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    pd.DataFrame(delta_rows).to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] Metric comparison (test): {args.out_csv}")

    metric_col = args.metric_column
    plt.figure(figsize=(12, 7))
    plotted = False
    labels: list[str] = []
    for i, rd in enumerate(all_runs):
        rc = results_csv_path(rd)
        label = os.path.basename(rd.rstrip(os.sep))[:40]
        labels.append(label)
        if not rc:
            print(f"[WARN] No train/results.csv: {rd}")
            continue
        try:
            df = pd.read_csv(rc)
            df.columns = [str(c).strip() for c in df.columns]
            mcol = metric_col if metric_col in df.columns else pick_map_column(df)
            if mcol is None or "epoch" not in df.columns:
                print(f"[WARN] No epoch/mAP columns in {rc}")
                continue
            plt.plot(df["epoch"], df[mcol], label=label, linewidth=2)
            plotted = True
        except Exception as e:
            print(f"[WARN] {rc}: {e}")

    if plotted:
        plt.title("Metrics by epoch (comparison of runs)")
        plt.xlabel("Epoch")
        plt.ylabel(metric_col or DEFAULT_MAP_COL)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="persecution", fontsize=9)
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=200)
        plt.close()
        print(f"[OK] Graph: {args.out_png}")
    else:
        plt.close()

    # bar graph for latest mAP from results.csv
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
        plt.title("The Last Era: Comparison")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        bar_path = re.sub(r"\.png$", "_bars.png", args.out_png)
        plt.savefig(bar_path, dpi=200)
        plt.close()
        print(f"[OK] Bar graph: {bar_path}")

    _finalize_compare_analytics_session(
        args, baseline, others, args.out_csv, args.out_png, bar_path
    )


def cmd_interactive(args: argparse.Namespace) -> None:
    runs = find_run_directories(args.models_root)
    if not runs:
        print("No runs found.")
        return
    for i, rd in enumerate(runs, start=1):
        print(f"  {i}. {rd}")
    try:
        bi = int(input("Baseline number: ").strip())
        oi = input("Rest numbers separated by commas: ").strip()
        idxs = [int(x.strip()) for x in oi.split(",") if x.strip()]
    except ValueError:
        print("Invalid input.")
        sys.exit(1)
    if bi < 1 or bi > len(runs):
        sys.exit(1)
    baseline = runs[bi - 1]
    others = []
    for j in idxs:
        if 1 <= j <= len(runs) and runs[j - 1] != baseline:
            others.append(runs[j - 1])
    if not others:
        print("There are no runs to compare.")
        sys.exit(1)
    out_dir = args.output_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(baseline.rstrip(os.sep))[:30]
    out_csv = os.path.join(out_dir, f"compare_{base_name}.csv")
    out_png = os.path.join(out_dir, f"compare_{base_name}.png")
    ns = argparse.Namespace(
        baseline=baseline,
        others=others,
        out_csv=out_csv,
        out_png=out_png,
        metric_column=args.metric_column,
        workspace=args.workspace,
        analytics_session=args.analytics_session,
        models_root=args.models_root,
    )
    cmd_compare(ns)


def _extract_pr_curve_from_metrics(metrics_obj: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Tries to get the all-classes PR curve from the Ultralytics metrics object."""
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

            # We are looking for the PR curve (Recall -> Precision).
            if "recall" not in marker or "precision" not in marker:
                continue

            if y.ndim >= 2:
                # Typically shape: (num_classes, points); average by class.
                y = np.nanmean(y, axis=0)
            if x.ndim > 1:
                x = np.ravel(x)
            if y.ndim > 1:
                y = np.ravel(y)

            n = min(len(x), len(y))
            if n == 0:
                continue
            return x[:n], y[:n]
    return None


def _resolve_pr_output_png(
    workspace_cli: str | None,
    out_png_cli: str | None,
    runs_group_dir: str,
) -> str:
    if out_png_cli:
        return os.path.abspath(os.path.expanduser(out_png_cli))
    try:
        ws = resolve_workspace_root(workspace_cli)
        analytics_dir = WorkspaceLayout(ws).analytics
    except ValueError:
        analytics_dir = os.path.join(os.path.dirname(os.path.abspath(runs_group_dir)), "analytics")
    os.makedirs(analytics_dir, exist_ok=True)
    ds_name = os.path.basename(os.path.normpath(runs_group_dir))
    return os.path.join(analytics_dir, f"pr_all_classes_{ds_name}.png")


def cmd_pr_curves(args: argparse.Namespace) -> None:
    runs_group_dir = os.path.abspath(os.path.expanduser(args.runs_group_dir))
    if not os.path.isdir(runs_group_dir):
        print(f"[ERROR] Models folder not found: {runs_group_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.data_yaml:
        print("[ERROR] Specify --data-yaml (path to data.yaml for split=test).", file=sys.stderr)
        sys.exit(1)
    data_yaml = os.path.abspath(os.path.expanduser(args.data_yaml))
    if not os.path.isfile(data_yaml):
        print(f"[ERROR] Data.yaml not found: {data_yaml}", file=sys.stderr)
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[ERROR] Failed to import ultralytics: {e}", file=sys.stderr)
        sys.exit(1)

    run_dirs = sorted(
        d for d in glob(os.path.join(runs_group_dir, "*"))
        if os.path.isdir(d)
    )
    if not run_dirs:
        print(f"[ERROR] There are no run directories in the folder: {runs_group_dir}", file=sys.stderr)
        sys.exit(1)

    curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    for run_dir in run_dirs:
        label = os.path.basename(run_dir.rstrip(os.sep))
        best_pt = os.path.join(run_dir, "train", "weights", "best.pt")
        if not os.path.isfile(best_pt):
            print(f"[WARN] {label}: no best.pt, skip ({best_pt})")
            continue
        print(f"[INFO] {label}: val(split=test) ...")
        model = YOLO(best_pt)
        try:
            metrics = model.val(
                data=data_yaml,
                split="test",
                plots=False,
                save=False,
                verbose=False,
            )
        except Exception as e:
            print(f"[WARN] {label}: error val(): {e}")
            continue

        pr = _extract_pr_curve_from_metrics(metrics)
        if pr is None:
            print(f"[WARN] {label}: PR curve not available in metrics object, skip")
            continue
        recall, precision = pr
        curves.append((label, recall, precision))

        pr_dir = os.path.join(run_dir, "test")
        os.makedirs(pr_dir, exist_ok=True)
        pr_csv = os.path.join(pr_dir, "pr.csv")
        pd.DataFrame({"recall": recall, "precision": precision}).to_csv(
            pr_csv, index=False, encoding="utf-8"
        )
        print(f"[OK] {label}: saved {pr_csv}")

    if not curves:
        print("[ERROR] Could not get any PR curves.", file=sys.stderr)
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
    print(f"[OK] General PR graph: {out_png}")


def _collect_split_images(data_yaml_path: str, split_name: str, limit: int) -> list[str]:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Incorrect YAML: {data_yaml_path}")
    split_rel = data.get(split_name)
    if not split_rel or not isinstance(split_rel, str):
        raise ValueError(f"There is no path for split={split_name!r} in data.yaml")

    base_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    split_path = os.path.abspath(os.path.join(base_dir, split_rel))
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"Directory split not found: {split_path}")

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
    runs_group_dir = os.path.abspath(os.path.expanduser(args.runs_group_dir))
    if not os.path.isdir(runs_group_dir):
        print(f"[ERROR] Models folder not found: {runs_group_dir}", file=sys.stderr)
        sys.exit(1)
    data_yaml = os.path.abspath(os.path.expanduser(args.data_yaml))
    if not os.path.isfile(data_yaml):
        print(f"[ERROR] Data.yaml not found: {data_yaml}", file=sys.stderr)
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
                    f"[WARN] CUDA unavailable (torch.cuda.is_available()=False). "
                    f"Switching from device={effective_device!r} to 'cpu'."
                )
                effective_device = "cpu"
        except Exception as e:
            print(f"[WARN] Failed to check CUDA via torch ({e}); using CPU.")
            effective_device = "cpu"
    if effective_device.lower() == "cpu" and effective_half:
        print("[WARN] --half is not used on the CPU; I disable half.")
        effective_half = False

    try:
        images = _collect_split_images(data_yaml, args.split, args.frames)
    except Exception as e:
        print(f"[ERROR] Failed to obtain frames for test: {e}", file=sys.stderr)
        sys.exit(1)
    if not images:
        print("[ERROR] No images found for inference.", file=sys.stderr)
        sys.exit(1)

    run_dirs = sorted(d for d in glob(os.path.join(runs_group_dir, "*")) if os.path.isdir(d))
    if not run_dirs:
        print(f"[ERROR] There are no run directories in the folder: {runs_group_dir}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        model_name = os.path.basename(run_dir.rstrip(os.sep))
        best_pt = os.path.join(run_dir, "train", "weights", "best.pt")
        if not os.path.isfile(best_pt):
            print(f"[WARN] {model_name}: no best.pt, skip")
            continue
        print(f"[INFO] {model_name}: benchmark for {len(images)} frames ...")
        try:
            model = YOLO(best_pt)
            # Light warm-up to reduce skew in the first iteration.
            model.predict(
                source=images[0],
                verbose=False,
                device=effective_device,
                half=effective_half,
            )
            timings_ms: list[float] = []
            prep_ms: list[float] = []
            infer_ms: list[float] = []
            post_ms: list[float] = []
            for img_path in images:
                t0 = time.perf_counter()
                results = model.predict(
                    source=img_path,
                    verbose=False,
                    device=effective_device,
                    half=effective_half,
                )
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
            if avg_infer is not None:
                print(
                    f"[OK] {model_name}: total={avg_ms:.2f} ms/frame, "
                    f"infer={avg_infer:.2f} ms/frame"
                )
            else:
                print(f"[OK] {model_name}: total={avg_ms:.2f} ms/frame")
        except Exception as e:
            print(f"[WARN] {model_name}: benchmark error: {e}")

    if not rows:
        print("[ERROR] No benchmark results.", file=sys.stderr)
        sys.exit(1)

    out_csv = _resolve_inference_csv_path(args.workspace, args.out_csv, runs_group_dir)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    sort_col = "avg_inference_ms_per_frame" if any(
        r.get("avg_inference_ms_per_frame") is not None for r in rows
    ) else "avg_total_ms_per_frame"
    pd.DataFrame(rows).sort_values(sort_col).to_csv(
        out_csv, index=False, encoding="utf-8"
    )
    print(f"[OK] CSV with results: {out_csv}")


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
    csv_path = os.path.abspath(os.path.expanduser(args.csv))
    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print(f"[ERROR] CSV is empty: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if "model" not in df.columns:
        print("[ERROR] There is no 'model' column in the CSV.", file=sys.stderr)
        sys.exit(1)
    metric = args.metric
    if metric not in df.columns:
        print(
            f"[ERROR] There is no {metric!r} column in CSV."
            f"Available: {', '.join(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    plot_df = df[["model", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    if len(plot_df) == 0:
        print(f"[ERROR] There are no numeric values ​​in column {metric!r}.", file=sys.stderr)
        sys.exit(1)

    # For ms less is better, for fps more is better.
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
    plt.title(f"Inference benchmark: {metric}")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Value labels above columns.
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


def build_analyze_arg_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Workspace root (aka {WORKSPACE_ENV_VAR}) for default run root",
    )
    common.add_argument(
        "--models-root",
        type=str,
        default=None,
        help="Explicit directory search root with training_metadata.json",
    )
    common.add_argument(
        "--analytics-session",
        type=str,
        default=None,
        help="Subdirectory workspace/analytics/: artifacts and session.json (export-table, compare, interactive)",
    )

    parser = CliArgumentParser(description="Analysis of YOLO training results (Ultralytics)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", parents=[common], help="List of runs")
    p_scan.set_defaults(func=cmd_scan)

    p_exp = sub.add_parser("export-table", parents=[common], help="Summary CSV for all runs")
    p_exp.add_argument("-o", "--output", type=str, default="runs_summary.csv")
    p_exp.set_defaults(func=cmd_export_table)

    p_cmp = sub.add_parser(
        "compare",
        parents=[common],
        help="Comparing the base run with others (CSV + graphs)",
    )
    p_cmp.add_argument("--baseline", type=str, required=True, help="Catalog of Persecution (Baseline)")
    p_cmp.add_argument(
        "--others",
        type=str,
        nargs="+",
        required=True,
        help="One or more run directories to compare",
    )
    p_cmp.add_argument("-o", "--out-csv", type=str, default="compare_delta.csv")
    p_cmp.add_argument("--out-png", type=str, default="compare_curves.png")
    p_cmp.add_argument(
        "--metric-column",
        type=str,
        default=DEFAULT_MAP_COL,
        help="Column from train/results.csv for graph",
    )
    p_cmp.set_defaults(func=cmd_compare)

    p_int = sub.add_parser(
        "interactive",
        parents=[common],
        help="Interactive selection of runs in the terminal",
    )
    p_int.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save compare_*.csv/png",
    )
    p_int.add_argument("--metric-column", type=str, default=DEFAULT_MAP_COL)
    p_int.set_defaults(func=cmd_interactive)

    p_pr = sub.add_parser(
        "pr-curves",
        parents=[common],
        help="Repeated test-val for all models in the folder + pr.csv by run + general PR-graph",
    )
    p_pr.add_argument(
        "--runs-group-dir",
        type=str,
        required=True,
        help="A folder like runs/<dataset_name>/, inside which there are model catalogs.",
    )
    p_pr.add_argument(
        "--data-yaml",
        type=str,
        required=True,
        help="Path to data.yaml dataset for split=test.",
    )
    p_pr.add_argument(
        "--out-png",
        type=str,
        default=None,
        help="Where to save the general graph. Default: workspace/analytics/pr_all_classes_<dataset>.png",
    )
    p_pr.set_defaults(func=cmd_pr_curves)

    p_inf = sub.add_parser(
        "inference-benchmark",
        parents=[common],
        help="Inference speed test of all models in the runs/<dataset>/ folder (average over N frames)",
    )
    p_inf.add_argument(
        "--runs-group-dir",
        type=str,
        required=True,
        help="A folder like runs/<dataset_name>/, inside which there are model catalogs.",
    )
    p_inf.add_argument(
        "--data-yaml",
        type=str,
        required=True,
        help="Path to data.yaml dataset.",
    )
    p_inf.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Which split to use for benchmark frames.",
    )
    p_inf.add_argument(
        "--frames",
        type=int,
        default=100,
        help="How many frames to use to calculate the average time.",
    )
    p_inf.add_argument(
        "--device",
        type=str,
        default="0",
        help="Inference device (for example: cpu, 0, 0,1).",
    )
    p_inf.add_argument(
        "--half",
        action="store_true",
        help="FP16 inference (mainly relevant for GPUs).",
    )
    p_inf.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Where to save the CSV. Default: workspace/analytics/inference_tests/<dataset>.csv",
    )
    p_inf.set_defaults(func=cmd_inference_benchmark)

    p_inf_plot = sub.add_parser(
        "inference-plot",
        parents=[common],
        help="Build a bar chart using the CSV of the inference benchmark",
    )
    p_inf_plot.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the CSV generated by analyze inference-benchmark.",
    )
    p_inf_plot.add_argument(
        "--metric",
        type=str,
        default="avg_inference_ms_per_frame",
        help=(
            "Column for a chart, for example: avg_inference_ms_per_frame,"
            "avg_total_ms_per_frame, avg_total_fps, avg_inference_fps."
        ),
    )
    p_inf_plot.add_argument(
        "--out-png",
        type=str,
        default=None,
        help="Where to save PNG. Default: analytics/inference_tests/<csv_name>_bars.png",
    )
    p_inf_plot.set_defaults(func=cmd_inference_plot)

    return parser


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_analyze_arg_parser()
    args = parser.parse_args(argv)
    args.models_root = resolve_models_scan_root(args.workspace, args.models_root)
    args.func(args)


if __name__ == "__main__":
    main()
