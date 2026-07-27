from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable

from smartrain.core.runtime.path_portable import posix_relpath


def run_all_quality_stage(
    *,
    args: Any,
    profile: str,
    baseline: str,
    others: list[str],
    selected_run_dirs: list[str],
    session_root: str,
    runs_group_dir: str,
    data_yaml: str,
    run_data_yaml_map: dict[str, str],
    collect_missing_metrics_recompute_plan_cb: Callable[..., dict[str, Any]],
    cmd_test_metrics_plot_cb: Callable[[argparse.Namespace], None],
    refresh_runs_summary_cb: Callable[[argparse.Namespace], None] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any] | None, bool]:
    artifacts: list[dict[str, str]] = []
    metric_sources_payload: dict[str, Any] | None = None
    recompute_missing_metrics = True

    if profile in ("quality", "full"):
        metric_sources_json = os.path.join(session_root, "artifacts", "metrics", "metric_sources.json")
        recompute_plan = collect_missing_metrics_recompute_plan_cb(
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
        cmd_test_metrics_plot_cb(tm_ns)
        artifacts.append({"role": "metrics_dir", "path": posix_relpath(tm_ns.out_dir, session_root)})
        artifacts.append({"role": "metric_sources", "path": posix_relpath(metric_sources_json, session_root)})
        if os.path.isfile(metric_sources_json):
            try:
                with open(metric_sources_json, "r", encoding="utf-8") as f:
                    metric_sources_payload = json.load(f)
            except Exception:
                metric_sources_payload = None

        if refresh_runs_summary_cb is not None:
            exp_csv = os.path.join(session_root, "artifacts", "table", "runs_summary.csv")
            refresh_runs_summary_cb(
                argparse.Namespace(
                    output=exp_csv,
                    workspace=args.workspace,
                    models_root=args.models_root,
                    analytics_session=args.analytics_session,
                )
            )
            artifacts.append({"role": "summary_csv", "path": posix_relpath(exp_csv, session_root)})

    return artifacts, metric_sources_payload, recompute_missing_metrics

