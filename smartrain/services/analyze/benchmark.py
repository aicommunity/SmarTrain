from __future__ import annotations

import argparse
import json
import os
import sys
import time
from glob import glob
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def resolve_selected_run_dirs(
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
    for path in (selected_run_dirs or []):
        value = str(path).strip()
        if not value:
            continue
        abs_path = os.path.abspath(os.path.expanduser(value))
        if abs_path not in selected_ordered:
            selected_ordered.append(abs_path)
    if not selected_ordered:
        return all_run_dirs
    explicit_existing = [d for d in selected_ordered if os.path.isdir(d)]
    if explicit_existing:
        return explicit_existing
    selected_norm = set(selected_ordered)
    return [d for d in all_run_dirs if os.path.abspath(d) in selected_norm]


def collect_split_images(data_yaml_path: str, split_name: str, limit: int) -> list[str]:
    with open(data_yaml_path, "r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
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


def resolve_inference_csv_path(
    workspace_cli: str | None,
    out_csv_cli: str | None,
    runs_group_dir: str,
    *,
    resolve_workspace_root_cb,
    workspace_layout_cls,
) -> str:
    if out_csv_cli:
        return os.path.abspath(os.path.expanduser(out_csv_cli))
    try:
        workspace = resolve_workspace_root_cb(workspace_cli)
        base = os.path.join(workspace_layout_cls(workspace).analytics, "inference_tests")
    except ValueError:
        base = os.path.join(os.path.dirname(os.path.abspath(runs_group_dir)), "analytics", "inference_tests")
    os.makedirs(base, exist_ok=True)
    group_name = os.path.basename(os.path.normpath(runs_group_dir))
    return os.path.join(base, f"{group_name}.csv")


def resolve_inference_plot_png(
    workspace_cli: str | None,
    out_png_cli: str | None,
    csv_path: str,
    *,
    resolve_workspace_root_cb,
    workspace_layout_cls,
) -> str:
    if out_png_cli:
        return os.path.abspath(os.path.expanduser(out_png_cli))
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    try:
        workspace = resolve_workspace_root_cb(workspace_cli)
        base = os.path.join(workspace_layout_cls(workspace).analytics, "inference_tests")
    except ValueError:
        base = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{csv_name}_bars.png")


def run_inference_benchmark(
    args: argparse.Namespace,
    *,
    prompt_text_cb,
    resolve_workspace_root_cb,
    workspace_layout_cls,
    preferred_run_model_path_cb,
    run_cache_root_cb,
    compute_fingerprint_cb,
    data_yaml_hash_cb,
    weights_hash_cb,
    append_cache_entry_cb,
    ultralytics_sidecar_dir_cb,
    clear_gpu_memory_cb,
) -> None:
    if (not getattr(args, "runs_group_dir", None) or not getattr(args, "data_yaml", None)) and sys.stdin.isatty():
        args.runs_group_dir = prompt_text_cb("Runs group dir", default=str(args.models_root)).strip() or str(args.models_root)
        args.data_yaml = prompt_text_cb("Path to data.yaml", default=str(getattr(args, "data_yaml", ""))).strip()
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
    except ImportError as exc:
        print(f"[ERROR] Failed to import ultralytics: {exc}", file=sys.stderr)
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
        except Exception as exc:
            print(f"[WARN] Could not validate CUDA via torch ({exc}); using CPU.")
            effective_device = "cpu"
    if effective_device.lower() == "cpu" and effective_half:
        print("[WARN] --half is not used on CPU; disabling half.")
        effective_half = False

    try:
        images = collect_split_images(data_yaml, args.split, args.frames)
    except Exception as exc:
        print(f"[ERROR] Failed to load test frames: {exc}", file=sys.stderr)
        sys.exit(1)
    if not images:
        print("[ERROR] No images found for inference.", file=sys.stderr)
        sys.exit(1)

    run_dirs = resolve_selected_run_dirs(runs_group_dir, getattr(args, "selected_run_dirs", None))
    if not run_dirs:
        print(f"[ERROR] No run directories found for scope in: {runs_group_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Benchmark scope: {len(run_dirs)} run(s) selected")

    rows: list[dict[str, Any]] = []
    cache_stats: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        model_name = os.path.basename(run_dir.rstrip(os.sep))
        best_pt = preferred_run_model_path_cb(run_dir, ".pt")
        if not os.path.isfile(best_pt):
            print(f"[WARN] {model_name}: missing run model, skipping")
            continue
        cache_root = run_cache_root_cb(run_dir)
        fp = compute_fingerprint_cb(
            {
                "tool": "analyze-v2",
                "task": "inference_benchmark",
                "split": args.split,
                "frames": int(args.frames),
                "device": effective_device,
                "half": bool(effective_half),
                "data_yaml_hash": data_yaml_hash_cb(data_yaml),
                "weights_hash": weights_hash_cb(run_dir),
            }
        )
        cache_csv = os.path.join(cache_root, "inference", f"bench_{fp}.csv")
        os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
        if bool(getattr(args, "reuse_run_cache", True)) and os.path.isfile(cache_csv):
            cached_df = pd.read_csv(cache_csv)
            if len(cached_df) > 0:
                rows.append(cached_df.iloc[0].to_dict())
                cache_stats.append({"run_dir": run_dir, "artifact": "inference.benchmark", "status": "hit"})
                print(f"[INFO] {model_name}: benchmark cache hit")
                continue
        print(f"[INFO] {model_name}: benchmarking on {len(images)} frames ...")
        try:
            clear_gpu_memory_cb()
            model = YOLO(best_pt)
            pred_proj = ultralytics_sidecar_dir_cb(run_dir, ".ultralytics_predict_scratch")
            pred_kw = dict(
                verbose=False,
                device=effective_device,
                half=effective_half,
                save=False,
                project=pred_proj,
                name="infer-bench",
                exist_ok=True,
            )
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
                start = time.perf_counter()
                results = model.predict(source=img_path, **pred_kw)
                end = time.perf_counter()
                timings_ms.append((end - start) * 1000.0)
                if results:
                    speed = getattr(results[0], "speed", None)
                    if isinstance(speed, dict):
                        prep = speed.get("preprocess")
                        infer = speed.get("inference")
                        post = speed.get("postprocess")
                        if prep is not None:
                            prep_ms.append(float(prep))
                        if infer is not None:
                            infer_ms.append(float(infer))
                        if post is not None:
                            post_ms.append(float(post))
            avg_ms = float(np.mean(timings_ms))
            avg_prep = float(np.mean(prep_ms)) if prep_ms else None
            avg_infer = float(np.mean(infer_ms)) if infer_ms else None
            avg_post = float(np.mean(post_ms)) if post_ms else None
            row = {
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
            rows.append(row)
            pd.DataFrame([row]).to_csv(cache_csv, index=False, encoding="utf-8")
            append_cache_entry_cb(
                run_dir,
                {"artifact": "inference.benchmark", "fingerprint": fp, "path": os.path.relpath(cache_csv, run_dir)},
            )
            cache_stats.append({"run_dir": run_dir, "artifact": "inference.benchmark", "status": "miss"})
            if avg_infer is not None:
                print(f"[OK] {model_name}: total={avg_ms:.2f} ms/frame, infer={avg_infer:.2f} ms/frame")
            else:
                print(f"[OK] {model_name}: total={avg_ms:.2f} ms/frame")
            clear_gpu_memory_cb()
        except Exception as exc:
            print(f"[WARN] {model_name}: benchmark error: {exc}")
            clear_gpu_memory_cb()
        finally:
            try:
                from smartrain.core.runtime.ultralytics_ephemeral import prune_empty_sidecar_dirs

                prune_empty_sidecar_dirs(run_dir)
            except Exception:
                pass

    if not rows:
        print("[ERROR] No benchmark results produced.", file=sys.stderr)
        sys.exit(1)

    out_csv = resolve_inference_csv_path(
        args.workspace,
        args.out_csv,
        runs_group_dir,
        resolve_workspace_root_cb=resolve_workspace_root_cb,
        workspace_layout_cls=workspace_layout_cls,
    )
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    sort_col = "avg_inference_ms_per_frame" if any(
        row.get("avg_inference_ms_per_frame") is not None for row in rows
    ) else "avg_total_ms_per_frame"
    pd.DataFrame(rows).sort_values(sort_col).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Results CSV: {out_csv}")
    stats_out = str(getattr(args, "cache_stats_out", "") or "").strip()
    if stats_out:
        os.makedirs(os.path.dirname(stats_out) or ".", exist_ok=True)
        with open(stats_out, "w", encoding="utf-8") as file_obj:
            json.dump({"cache": cache_stats}, file_obj, ensure_ascii=False, indent=2)


def run_inference_plot(
    args: argparse.Namespace,
    *,
    prompt_text_cb,
    default_relative_output_cb,
    resolve_workspace_root_cb,
    workspace_layout_cls,
) -> None:
    if not getattr(args, "csv", None) and sys.stdin.isatty():
        default_csv = default_relative_output_cb(
            args.workspace, args.analytics_session, "inference", "benchmark.csv", None
        )
        args.csv = prompt_text_cb("Path to benchmark CSV", default=default_csv).strip()
    if not getattr(args, "csv", None):
        print("[ERROR] Incomplete arguments: --csv is required.", file=sys.stderr)
        sys.exit(2)
    csv_path = os.path.abspath(os.path.expanduser(args.csv))
    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    data_frame = pd.read_csv(csv_path)
    if len(data_frame) == 0:
        print(f"[ERROR] CSV is empty: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if "model" not in data_frame.columns:
        print("[ERROR] CSV has no 'model' column.", file=sys.stderr)
        sys.exit(1)
    label_col = "display_label" if "display_label" in data_frame.columns else "model"
    metric = args.metric
    if metric not in data_frame.columns:
        print(
            f"[ERROR] CSV has no column {metric!r}. Available: {', '.join(data_frame.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    plot_df = data_frame[[label_col, metric]].copy()
    plot_df = plot_df.rename(columns={label_col: "model"})
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    if len(plot_df) == 0:
        print(f"[ERROR] No numeric values in column {metric!r}.", file=sys.stderr)
        sys.exit(1)

    ascending = "fps" not in metric.lower()
    plot_df = plot_df.sort_values(metric, ascending=ascending)

    out_png = resolve_inference_plot_png(
        args.workspace,
        args.out_png,
        csv_path,
        resolve_workspace_root_cb=resolve_workspace_root_cb,
        workspace_layout_cls=workspace_layout_cls,
    )
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    plt.figure(figsize=(10, 6))
    indices = range(len(plot_df))
    values = plot_df[metric].tolist()
    bars = plt.bar(indices, values, tick_label=plot_df["model"].tolist())
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(metric)
    plt.title("Inference benchmark")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)

    max_value = max(values) if values else 0.0
    y_pad = max_value * 0.015 if max_value > 0 else 0.01
    for bar, value in zip(bars, values):
        x_text = bar.get_x() + bar.get_width() / 2.0
        y_text = bar.get_height()
        plt.text(
            x_text,
            y_text + y_pad,
            f"{float(value):.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[OK] Bar chart: {out_png}")

