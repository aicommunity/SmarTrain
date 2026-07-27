from __future__ import annotations

import os
import sys
from typing import Any, Callable

import pandas as pd

from smartrain.core.analyze.run_metrics_discovery import resolve_recomputed_metrics_csv


def run_scan_command(
    *,
    runs: list[str],
    flat_row_for_run: Callable[[str], dict[str, Any]],
) -> None:
    scan_runs(runs=runs, flat_row_for_run=flat_row_for_run)


def scan_runs(
    *,
    runs: list[str],
    flat_row_for_run: Callable[[str], dict[str, Any]],
) -> None:
    if not runs:
        print("(no runs or promoted models found)")
        return
    rows: list[dict[str, Any]] = []
    for run_dir in runs:
        try:
            flat = flat_row_for_run(run_dir)
            rows.append({"run_dir": run_dir, **flat})
        except Exception as exc:
            rows.append({"run_dir": run_dir, "model": "?", "dataset_name": "?", "release_comment": "", "_error": exc})
    show_comment = any(str(r.get("release_comment") or "").strip() for r in rows)
    if show_comment:
        print(f"{'#':>4}  {'model':<14}  {'dataset':<24}  {'comment':<32}  {'path'}")
        print("-" * 130)
        for i, flat in enumerate(rows, start=1):
            run_dir = flat["run_dir"]
            if flat.get("_error"):
                print(f"{i:4d}  {'?':<14}  {'?':<24}  {'':<32}  {run_dir}  [error: {flat['_error']}]")
                continue
            model = flat.get("model") or "?"
            dataset = flat.get("dataset_name") or "?"
            comment = str(flat.get("release_comment") or "")[:32]
            print(f"{i:4d}  {str(model)[:14]:<14}  {str(dataset)[:24]:<24}  {comment:<32}  {run_dir}")
    else:
        print(f"{'#':>4}  {'model':<14}  {'dataset':<24}  {'path'}")
        print("-" * 100)
        for i, flat in enumerate(rows, start=1):
            run_dir = flat["run_dir"]
            if flat.get("_error"):
                print(f"{i:4d}  {'?':<14}  {'?':<24}  {run_dir}  [error: {flat['_error']}]")
                continue
            model = flat.get("model") or "?"
            dataset = flat.get("dataset_name") or "?"
            print(f"{i:4d}  {str(model)[:14]:<14}  {str(dataset)[:24]:<24}  {run_dir}")


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
        if not test_metrics_path:
            test_metrics_path = resolve_recomputed_metrics_csv(run_dir)
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
                    map_series = pd.to_numeric(train_df[map_col], errors="coerce")
                    if map_series.notna().any():
                        best_idx = int(map_series.idxmax())
                        best = train_df.loc[best_idx]
                        row["train_best_epoch"] = best.get("epoch")
                        row[f"train_best_{map_col}"] = best.get(map_col)
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
