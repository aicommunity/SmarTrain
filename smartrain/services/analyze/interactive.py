from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable

import pandas as pd

from smartrain.services.analyze.all_selection import _display_model_column


def run_interactive_workflow(
    *,
    args: argparse.Namespace,
    indexed: list[tuple[str, Any]],
    session_artifacts_dir: Callable[[str | None, str | None, str], str],
    cmd_compare: Callable[[argparse.Namespace], None],
    cmd_test_metrics_plot: Callable[[argparse.Namespace], None],
    cmd_inference_benchmark: Callable[[argparse.Namespace], None],
    cmd_inference_plot: Callable[[argparse.Namespace], None],
    cmd_pr_curves: Callable[[argparse.Namespace], None],
    runs_with_missing_metrics: Callable[..., list[str]],
    auto_select_data_yaml: Callable[..., str | None],
    prompt_choice: Callable[..., str],
) -> None:
    model_w = 20
    print(f"{'#':>4}  {'model':<{model_w}}  {'dataset':<24}  {'mAP50-95':>9}  {'Box-F1':>9}  {'run_dir'}")
    print("-" * 150)
    for i, (run_dir, rec) in enumerate(indexed, start=1):
        q = rec.test_metrics.get("mAP50-95")
        f1 = rec.test_metrics.get("Box-F1")
        q_str = f"{float(q):.4f}" if q is not None and pd.notna(q) else "-"
        f1_str = f"{float(f1):.4f}" if f1 is not None and pd.notna(f1) else "-"
        print(
            f"{i:4d}  {_display_model_column(run_dir, rec, width=model_w):<{model_w}}  {str(rec.dataset_name or '?')[:24]:<24}  "
            f"{q_str:>9}  {f1_str:>9}  {run_dir}"
        )
    try:
        baseline_idx = int(input("Baseline run number: ").strip())
        others_raw = input("Other run numbers (comma-separated): ").strip()
        other_indexes = [int(x.strip()) for x in others_raw.split(",") if x.strip()]
    except ValueError:
        print("Invalid input.")
        sys.exit(1)
    if baseline_idx < 1 or baseline_idx > len(indexed):
        sys.exit(1)
    baseline = indexed[baseline_idx - 1][0]
    others: list[str] = []
    for j in other_indexes:
        if 1 <= j <= len(indexed) and indexed[j - 1][0] != baseline:
            others.append(indexed[j - 1][0])
    if not others:
        print("No runs selected for comparison.")
        sys.exit(1)

    if args.output_dir:
        out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    else:
        out_dir = session_artifacts_dir(args.workspace, args.analytics_session, "compare")
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(baseline.rstrip(os.sep))[:30]
    out_csv = os.path.join(out_dir, f"compare_{base_name}.csv")
    out_png = os.path.join(out_dir, f"compare_{base_name}.png")
    out_insights = os.path.join(out_dir, f"compare_{base_name}_insights.txt")
    compare_ns = argparse.Namespace(
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
    cmd_compare(compare_ns)

    preset = args.preset
    selected_data_yaml: str | None = args.data_yaml
    if preset in ("quality", "full"):
        metric_list = [m.strip() for m in (args.quality_metrics or "").split(",") if m.strip()]
        if metric_list:
            runs_group_dir = os.path.dirname(baseline)
            recompute_missing_metrics = bool(getattr(args, "recompute_missing_metrics", False))
            if sys.stdin.isatty():
                missing_runs = runs_with_missing_metrics(
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
            data_yaml = auto_select_data_yaml(
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
