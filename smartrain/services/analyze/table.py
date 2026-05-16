from __future__ import annotations

import os
import sys
from typing import Any, Callable

import pandas as pd


def run_scan_command(
    *,
    models_root: str,
    find_run_directories_fn: Callable[[str], list[str]],
    flat_row_for_run: Callable[[str], dict[str, Any]],
) -> None:
    runs = find_run_directories_fn(models_root)
    scan_runs(runs=runs, flat_row_for_run=flat_row_for_run)


def scan_runs(
    *,
    runs: list[str],
    flat_row_for_run: Callable[[str], dict[str, Any]],
) -> None:
    if not runs:
        print("(no runs with training_metadata.json found)")
        return
    print(f"{'#':>4}  {'model':<14}  {'dataset':<24}  {'run_dir'}")
    print("-" * 100)
    for i, run_dir in enumerate(runs, start=1):
        try:
            flat = flat_row_for_run(run_dir)
            model = flat.get("model") or "?"
            dataset = flat.get("dataset_name") or "?"
            print(f"{i:4d}  {str(model)[:14]:<14}  {str(dataset)[:24]:<24}  {run_dir}")
        except Exception as exc:
            print(f"{i:4d}  {'?':<14}  {'?':<24}  {run_dir}  [error: {exc}]")


def export_runs_table(
    *,
    runs: list[str],
    out_path: str,
    latest_test_metrics_path: Callable[[str], str | None],
    results_csv_path: Callable[[str], str | None],
    pick_map_column: Callable[[pd.DataFrame], str | None],
    flat_row_for_run: Callable[[str], dict[str, Any]],
) -> int:
    rows: list[dict[str, Any]] = []
    for run_dir in runs:
        try:
            row = flat_row_for_run(run_dir)
        except Exception as exc:
            print(f"[WARN] {run_dir}: {exc}", file=sys.stderr)
            continue
        test_metrics_path = latest_test_metrics_path(run_dir)
        if test_metrics_path:
            try:
                test_df = pd.read_csv(test_metrics_path)
                test_df.columns = [str(c).strip() for c in test_df.columns]
                if len(test_df) > 0:
                    for col in test_df.columns:
                        row[f"test_{col}"] = test_df[col].iloc[0]
            except Exception as exc:
                row["test_read_error"] = str(exc)
        train_results_path = results_csv_path(run_dir)
        if train_results_path:
            try:
                train_df = pd.read_csv(train_results_path)
                train_df.columns = [str(c).strip() for c in train_df.columns]
                map_col = pick_map_column(train_df)
                if map_col and "epoch" in train_df.columns and len(train_df) > 0:
                    last = train_df.iloc[-1]
                    row["train_last_epoch"] = last.get("epoch")
                    row[f"train_last_{map_col}"] = last.get(map_col)
            except Exception as exc:
                row["train_read_error"] = str(exc)
        rows.append(row)
    if not rows:
        print("[ERROR] No data to export.", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Summary table: {out_path} ({len(df)} rows)")
    return 0
