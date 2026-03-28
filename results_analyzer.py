#!/usr/bin/env python3
"""
Сканирование прогонов обучения, сводные CSV и сравнение метрик (CSV + PNG).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from glob import glob
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_MAP_COL = "metrics/mAP50-95(B)"


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
        print("(прогоны с training_metadata.json не найдены)")
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
            print(f"{i:4d}  {'?':<14}  {'?':<24}  {rd}  [ошибка: {e}]")


def cmd_export_table(args: argparse.Namespace) -> None:
    runs = find_run_directories(args.models_root)
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
        print("[ERROR] Нет данных для экспорта.", file=sys.stderr)
        sys.exit(1)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[OK] Сводная таблица: {args.output} ({len(df)} строк)")


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
    all_runs = [baseline] + others
    for p in all_runs:
        if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "training_metadata.json")):
            print(f"[ERROR] Не прогон (нет training_metadata.json): {p}", file=sys.stderr)
            sys.exit(1)

    base_metrics = _read_test_metrics_row(baseline)
    if not base_metrics:
        print("[WARN] У базы нет test_metrics*.csv — дельты только по train/results.csv", file=sys.stderr)

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
    print(f"[OK] Сравнение метрик (test): {args.out_csv}")

    metric_col = args.metric_column
    plt.figure(figsize=(12, 7))
    plotted = False
    labels: list[str] = []
    for i, rd in enumerate(all_runs):
        rc = results_csv_path(rd)
        label = os.path.basename(rd.rstrip(os.sep))[:40]
        labels.append(label)
        if not rc:
            print(f"[WARN] Нет train/results.csv: {rd}")
            continue
        try:
            df = pd.read_csv(rc)
            df.columns = [str(c).strip() for c in df.columns]
            mcol = metric_col if metric_col in df.columns else pick_map_column(df)
            if mcol is None or "epoch" not in df.columns:
                print(f"[WARN] Нет колонок epoch / mAP в {rc}")
                continue
            plt.plot(df["epoch"], df[mcol], label=label, linewidth=2)
            plotted = True
        except Exception as e:
            print(f"[WARN] {rc}: {e}")

    if plotted:
        plt.title("Метрики по эпохам (сравнение прогонов)")
        plt.xlabel("Эпоха")
        plt.ylabel(metric_col or DEFAULT_MAP_COL)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Прогон", fontsize=9)
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=200)
        plt.close()
        print(f"[OK] График: {args.out_png}")
    else:
        plt.close()

    # столбчатый график по последнему mAP из results.csv
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
        plt.title("Последняя эпоха: сравнение")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        bar_path = re.sub(r"\.png$", "_bars.png", args.out_png)
        plt.savefig(bar_path, dpi=200)
        plt.close()
        print(f"[OK] Столбчатый график: {bar_path}")


def cmd_interactive(args: argparse.Namespace) -> None:
    runs = find_run_directories(args.models_root)
    if not runs:
        print("Прогоны не найдены.")
        return
    for i, rd in enumerate(runs, start=1):
        print(f"  {i}. {rd}")
    try:
        bi = int(input("Номер базового прогона (baseline): ").strip())
        oi = input("Номера остальных через запятую: ").strip()
        idxs = [int(x.strip()) for x in oi.split(",") if x.strip()]
    except ValueError:
        print("Некорректный ввод.")
        sys.exit(1)
    if bi < 1 or bi > len(runs):
        sys.exit(1)
    baseline = runs[bi - 1]
    others = []
    for j in idxs:
        if 1 <= j <= len(runs) and runs[j - 1] != baseline:
            others.append(runs[j - 1])
    if not others:
        print("Нет прогонов для сравнения.")
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
    )
    cmd_compare(ns)


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--models-root",
        type=str,
        default=".",
        help="Корень поиска каталогов с training_metadata.json",
    )

    parser = argparse.ArgumentParser(description="Анализ результатов обучения YOLO (Ultralytics)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", parents=[common], help="Список прогонов")
    p_scan.set_defaults(func=cmd_scan)

    p_exp = sub.add_parser("export-table", parents=[common], help="Сводный CSV по всем прогонам")
    p_exp.add_argument("-o", "--output", type=str, default="runs_summary.csv")
    p_exp.set_defaults(func=cmd_export_table)

    p_cmp = sub.add_parser(
        "compare",
        parents=[common],
        help="Сравнение базового прогона с другими (CSV + графики)",
    )
    p_cmp.add_argument("--baseline", type=str, required=True, help="Каталог прогона (baseline)")
    p_cmp.add_argument(
        "--others",
        type=str,
        nargs="+",
        required=True,
        help="Один или несколько каталогов прогонов для сравнения",
    )
    p_cmp.add_argument("-o", "--out-csv", type=str, default="compare_delta.csv")
    p_cmp.add_argument("--out-png", type=str, default="compare_curves.png")
    p_cmp.add_argument(
        "--metric-column",
        type=str,
        default=DEFAULT_MAP_COL,
        help="Колонка из train/results.csv для графика",
    )
    p_cmp.set_defaults(func=cmd_compare)

    p_int = sub.add_parser(
        "interactive",
        parents=[common],
        help="Интерактивный выбор прогонов в терминале",
    )
    p_int.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Куда сохранить compare_*.csv/png",
    )
    p_int.add_argument("--metric-column", type=str, default=DEFAULT_MAP_COL)
    p_int.set_defaults(func=cmd_interactive)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
