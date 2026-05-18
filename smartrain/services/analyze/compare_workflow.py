from __future__ import annotations

import os
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd


def run_compare_workflow(
    *,
    baseline: str,
    others: list[str],
    out_csv: str,
    out_insights: str,
    out_png: str,
    metric_column: str,
    read_test_metrics_for_run: Callable[[str], dict[str, Any]],
    build_delta_rows: Callable[[str, dict[str, Any], list[str], list[dict[str, Any]]], list[dict[str, Any]]],
    generate_compare_insights: Callable[[str, list[str], list[dict[str, Any]]], list[str]],
    results_csv_path: Callable[[str], str | None],
    pick_map_column: Callable[[pd.DataFrame], str | None],
    default_map_col: str,
) -> tuple[str | None, list[dict[str, Any]]]:
    base_metrics = read_test_metrics_for_run(baseline)
    other_rows = [read_test_metrics_for_run(other) for other in others]
    delta_rows = build_delta_rows(baseline, base_metrics, others, other_rows)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    pd.DataFrame(delta_rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Test metrics comparison: {out_csv}")

    insight_lines = generate_compare_insights(baseline, others, delta_rows)
    os.makedirs(os.path.dirname(out_insights) or ".", exist_ok=True)
    with open(out_insights, "w", encoding="utf-8") as f:
        f.write("\n".join(insight_lines).rstrip() + "\n")
    print(f"[OK] Insights: {out_insights}")

    all_runs = [baseline] + others
    plt.figure(figsize=(12, 7))
    plotted = False
    labels: list[str] = []
    for run_dir in all_runs:
        results_csv = results_csv_path(run_dir)
        label = os.path.basename(run_dir.rstrip(os.sep))[:40]
        labels.append(label)
        if not results_csv:
            print(f"[WARN] Missing train/results.csv: {run_dir}")
            continue
        try:
            df = pd.read_csv(results_csv)
            df.columns = [str(c).strip() for c in df.columns]
            metric_col = metric_column if metric_column in df.columns else pick_map_column(df)
            if metric_col is None or "epoch" not in df.columns:
                print(f"[WARN] Missing epoch / mAP columns in {results_csv}")
                continue
            plt.plot(df["epoch"], df[metric_col], label=label, linewidth=2)
            plotted = True
        except Exception as exc:
            print(f"[WARN] {results_csv}: {exc}")

    if plotted:
        plt.title("Metrics Comparison Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric_column or default_map_col)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Model", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[OK] Plot: {out_png}")
    else:
        plt.close()

    last_vals: list[float] = []
    last_labs: list[str] = []
    for run_dir, label in zip(all_runs, labels):
        results_csv = results_csv_path(run_dir)
        if not results_csv:
            continue
        try:
            df = pd.read_csv(results_csv)
            df.columns = [str(c).strip() for c in df.columns]
            metric_col = metric_column if metric_column in df.columns else pick_map_column(df)
            if metric_col and len(df) > 0:
                value = df[metric_col].iloc[-1]
                if pd.notna(value):
                    last_vals.append(float(value))
                    last_labs.append(label)
        except Exception:
            pass

    bar_path: str | None = None
    if len(last_vals) >= 2:
        plt.figure(figsize=(10, 5))
        x = range(len(last_labs))
        plt.bar(x, last_vals, tick_label=last_labs)
        plt.ylabel(metric_column or default_map_col)
        plt.title("Last Epoch Comparison")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        bar_path = out_png[:-4] + "_bars.png" if out_png.lower().endswith(".png") else f"{out_png}_bars.png"
        plt.savefig(bar_path, dpi=200)
        plt.close()
        print(f"[OK] Bar chart: {bar_path}")

    return bar_path, delta_rows
