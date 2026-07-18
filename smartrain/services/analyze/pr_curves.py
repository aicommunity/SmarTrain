from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smartrain.core.runtime.path_portable import posix_relpath


def resolve_pr_output_png(
    workspace_cli: str | None,
    out_png_cli: str | None,
    runs_group_dir: str,
    *,
    resolve_workspace_root_cb,
    workspace_layout_cls,
) -> str:
    if out_png_cli:
        return os.path.abspath(os.path.expanduser(out_png_cli))
    try:
        workspace = resolve_workspace_root_cb(workspace_cli)
        base = os.path.join(workspace_layout_cls(workspace).analytics, "pr_curves")
    except ValueError:
        base = os.path.join(os.path.dirname(os.path.abspath(runs_group_dir)), "analytics", "pr_curves")
    os.makedirs(base, exist_ok=True)
    group_name = os.path.basename(os.path.normpath(runs_group_dir))
    return os.path.join(base, f"{group_name}_all_classes.png")


def run_pr_curves(
    args: argparse.Namespace,
    *,
    prompt_text_cb,
    resolve_selected_run_dirs_cb,
    load_dataset_class_names_cb,
    preferred_run_model_path_cb,
    run_cache_root_cb,
    compute_fingerprint_cb,
    data_yaml_hash_cb,
    weights_hash_cb,
    clear_gpu_memory_cb,
    resolve_run_val_profile_cb,
    ultralytics_sidecar_dir_cb,
    run_val_memory_safe_cb,
    extract_pr_curve_cb,
    extract_pr_curve_per_class_cb,
    append_cache_entry_cb,
    safe_name_cb,
    resolve_workspace_root_cb,
    workspace_layout_cls,
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
    if not args.data_yaml:
        print("[ERROR] Please provide --data-yaml (path to data.yaml for split=test).", file=sys.stderr)
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

    run_dirs = resolve_selected_run_dirs_cb(runs_group_dir, getattr(args, "selected_run_dirs", None))
    if not run_dirs:
        print(f"[ERROR] No run directories found for scope in: {runs_group_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] PR scope: {len(run_dirs)} run(s) selected")

    curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    per_class_rows: list[dict[str, Any]] = []
    class_names = load_dataset_class_names_cb(data_yaml)
    per_class_enabled = bool(getattr(args, "pr_per_class", True))
    reuse_cache = bool(getattr(args, "reuse_run_cache", True))
    cache_stats: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        label = os.path.basename(run_dir.rstrip(os.sep))
        best_pt = preferred_run_model_path_cb(run_dir, ".pt")
        if not os.path.isfile(best_pt):
            print(f"[WARN] {label}: missing run model, skipping ({best_pt})")
            continue
        cache_root = run_cache_root_cb(run_dir)
        fp = compute_fingerprint_cb(
            {
                "tool": "analyze-v2",
                "task": "pr_curves",
                "data_yaml_hash": data_yaml_hash_cb(data_yaml),
                "split": "test",
                "weights_hash": weights_hash_cb(run_dir),
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
            cached_df = pd.read_csv(cache_agg)
            if {"recall", "precision"}.issubset(set(cached_df.columns)):
                recall = cached_df["recall"].to_numpy(dtype=float)
                precision = cached_df["precision"].to_numpy(dtype=float)
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
            clear_gpu_memory_cb()
            model = YOLO(best_pt)
            try:
                batch, imgsz, half = resolve_run_val_profile_cb(
                    run_dir,
                    default_batch=int(getattr(args, "val_batch", 1)),
                    default_imgsz=int(getattr(args, "val_imgsz", 640)),
                    default_half=bool(getattr(args, "val_half", True)),
                )
                ultra_proj = ultralytics_sidecar_dir_cb(run_dir, ".ultralytics_scratch")
                metrics = run_val_memory_safe_cb(
                    model,
                    data_yaml=data_yaml,
                    split="test",
                    val_batch=batch,
                    val_imgsz=imgsz,
                    val_half=half,
                    gpu_only=bool(getattr(args, "gpu_only_val", True)),
                    ultra_project=ultra_proj,
                    ultra_name="val-pr-curves",
                )
            except Exception as exc:
                print(f"[WARN] {label}: val() error: {exc}")
                clear_gpu_memory_cb()
                continue
            finally:
                try:
                    from smartrain.core.runtime.ultralytics_ephemeral import prune_empty_sidecar_dirs

                    prune_empty_sidecar_dirs(run_dir)
                except Exception:
                    pass
            pr = extract_pr_curve_cb(metrics)
            if pr is None:
                print(f"[WARN] {label}: PR curve not available in metrics object, skipping")
                continue
            recall, precision = pr
            pd.DataFrame({"recall": recall, "precision": precision}).to_csv(cache_agg, index=False, encoding="utf-8")
            append_cache_entry_cb(
                run_dir,
                {"artifact": "pr.aggregate", "fingerprint": fp, "path": posix_relpath(cache_agg, run_dir)},
            )
            cache_stats.append({"run_dir": run_dir, "artifact": "pr.aggregate", "status": "miss"})
            if per_class_enabled:
                per_class = extract_pr_curve_per_class_cb(metrics)
                if per_class is not None:
                    rx, y2d = per_class
                    rows: list[dict[str, Any]] = []
                    for class_id in range(y2d.shape[0]):
                        class_name = class_names.get(class_id, f"class_{class_id}")
                        ap = float(np.trapz(y2d[class_id], rx))
                        for idx in range(len(rx)):
                            rows.append(
                                {
                                    "run_dir": run_dir,
                                    "model": label,
                                    "class_id": class_id,
                                    "class_name": class_name,
                                    "recall": float(rx[idx]),
                                    "precision": float(y2d[class_id][idx]),
                                    "ap": ap,
                                }
                            )
                    per_class_df = pd.DataFrame(rows)
                    per_class_df.to_csv(cache_pc, index=False, encoding="utf-8")
                    append_cache_entry_cb(
                        run_dir,
                        {"artifact": "pr.per_class", "fingerprint": fp, "path": posix_relpath(cache_pc, run_dir)},
                    )
                    cache_stats.append({"run_dir": run_dir, "artifact": "pr.per_class", "status": "miss"})
            clear_gpu_memory_cb()

        curves.append((label, recall, precision))
        if per_class_df is not None and len(per_class_df) > 0:
            per_class_rows.extend(per_class_df.to_dict(orient="records"))
        print(f"[OK] {label}: PR curve cached at {cache_agg}")

    if not curves:
        if bool(getattr(args, "soft_fail", False)):
            print("[WARN] Failed to obtain any PR curves; skipping PR artifacts for this group.")
            stats_out = str(getattr(args, "cache_stats_out", "") or "").strip()
            if stats_out:
                os.makedirs(os.path.dirname(stats_out) or ".", exist_ok=True)
                with open(stats_out, "w", encoding="utf-8") as file_obj:
                    json.dump({"cache": cache_stats, "skipped": True}, file_obj, ensure_ascii=False, indent=2)
            return
        print("[ERROR] Failed to obtain any PR curves.", file=sys.stderr)
        sys.exit(1)

    out_png = resolve_pr_output_png(
        args.workspace,
        args.out_png,
        runs_group_dir,
        resolve_workspace_root_cb=resolve_workspace_root_cb,
        workspace_layout_cls=workspace_layout_cls,
    )
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
            for model_name, model_df in cls_df.groupby("model"):
                model_df = model_df.sort_values("recall")
                plt.plot(model_df["recall"], model_df["precision"], linewidth=1.8, label=model_name)
            plt.title(f"PR per class: {class_name} (id={class_id})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(fontsize=8)
            plt.tight_layout()
            cls_png = os.path.join(out_base_dir, f"pr_class_{int(class_id)}_{safe_name_cb(str(class_name))}.png")
            plt.savefig(cls_png, dpi=220)
            plt.close()
        print(f"[OK] Per-class PR artifacts: {out_base_dir}")

    stats_out = str(getattr(args, "cache_stats_out", "") or "").strip()
    if stats_out:
        os.makedirs(os.path.dirname(stats_out) or ".", exist_ok=True)
        with open(stats_out, "w", encoding="utf-8") as file_obj:
            json.dump({"cache": cache_stats}, file_obj, ensure_ascii=False, indent=2)

