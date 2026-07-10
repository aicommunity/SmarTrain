from __future__ import annotations

import argparse
import json
import os
from glob import glob
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_all_pr_stage(
    *,
    args: Any,
    profile: str,
    selected_run_dirs: list[str],
    session_root: str,
    runs_group_dir: str,
    run_data_yaml_map: dict[str, str],
    record_failure_cb: Callable[..., None],
    group_runs_by_data_yaml_cb: Callable[[list[str], dict[str, str]], tuple[dict[str, list[str]], list[str]]],
    cmd_pr_curves_cb: Callable[[argparse.Namespace], None],
    safe_name_cb: Callable[[str], str],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    if profile != "full":
        return [], []

    artifacts: list[dict[str, str]] = []
    cache_events: list[dict[str, Any]] = []

    pr_groups, unresolved_for_pr = group_runs_by_data_yaml_cb(selected_run_dirs, run_data_yaml_map)
    if unresolved_for_pr:
        print("[WARN] PR stage: skipped runs without resolved data.yaml:")
        for rd in unresolved_for_pr:
            print(f"[WARN]  - {os.path.basename(rd.rstrip(os.sep))}")
            record_failure_cb(
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
            soft_fail=True,
        )
        cmd_pr_curves_cb(pr_ns)
        if os.path.isfile(pr_png):
            artifacts.append({"role": "pr_png", "path": os.path.relpath(pr_png, session_root)})
            pr_png_written = True

        part_csv = os.path.join(group_pr_dir, "per_class", "pr_per_class.csv")
        if os.path.isfile(part_csv):
            try:
                pr_per_class_frames.append(pd.read_csv(part_csv))
            except Exception as e:
                record_failure_cb(
                    stage="pr",
                    status="failed",
                    reason_code="per_class_csv_read_failed",
                    reason_detail=str(e),
                    split="test",
                )
        else:
            for rd in group_runs:
                record_failure_cb(
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
                with open(pr_cache_stats, "r", encoding="utf-8") as f:
                    cache_events.extend(json.load(f).get("cache", []))
            except Exception as e:
                record_failure_cb(
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
                present_models_local: set[str] = set()
                for model_name, mdf in cdf.groupby("model"):
                    mdf = mdf.sort_values("recall_num")
                    if len(mdf) == 0:
                        continue
                    present_models_local.add(str(model_name))
                    plt.plot(
                        mdf["recall_num"],
                        mdf["precision_num"],
                        linewidth=1.8,
                        label=str(model_name),
                    )
                if not present_models_local:
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
                    f"pr_class_{class_id_int}_{safe_name_cb(str(class_name))}_all_runs.png",
                )
                plt.savefig(out_png, dpi=220)
                plt.close()
                artifacts.append({"role": "pr_per_class_png", "path": os.path.relpath(out_png, session_root)})
                if len(present_models_local) < len(expected_models):
                    record_failure_cb(
                        stage="pr",
                        status="failed",
                        reason_code="png_incomplete_series",
                        reason_detail=(
                            f"pr_per_class_combined class={class_name} "
                            f"models={sorted(present_models_local)} expected={sorted(expected_models)}"
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
                record_failure_cb(
                    stage="pr",
                    status="failed",
                    reason_code="png_incomplete_series",
                    reason_detail=f"pr_per_class models={sorted(actual)} expected={sorted(expected)}",
                    split="test",
                )
        except Exception as e:
            record_failure_cb(
                stage="pr",
                status="failed",
                reason_code="pr_series_validation_failed",
                reason_detail=str(e),
                split="test",
            )

    if not pr_png_written:
        print("[WARN] PR stage completed without PR plot artifacts.")
        record_failure_cb(
            stage="pr",
            status="missing",
            reason_code="pr_plot_missing",
            reason_detail="no PR plot artifacts were produced",
            split="test",
        )

    for p in sorted(glob(os.path.join(session_root, "artifacts", "pr", "**", "per_class", "*.png"), recursive=True)):
        artifacts.append({"role": "pr_per_class_png", "path": os.path.relpath(p, session_root)})

    return artifacts, cache_events

