"""Markdown builders for analyze session reports."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from smartrain.tasks.metric_columns import (
    compare_delta_column_names,
    detect_task_type_from_columns,
    format_metrics_table_column_names,
    primary_quality_columns,
    runs_summary_test_column_names,
)
from smartrain.services.analyze.schema_contracts import ensure_analyze_session_manifest

def _read_template(lang: str) -> dict[str, str]:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "templates", "analyze_report"))
    path = os.path.join(base, f"{lang}.md")
    if not os.path.isfile(path):
        return {}
    text = open(path, "r", encoding="utf-8").read()
    chunks = re.split(r"(?m)^<!--\s*BLOCK:\s*([A-Z_]+)\s*-->\s*$", text)
    out: dict[str, str] = {}
    if len(chunks) < 3:
        return out
    i = 1
    while i + 1 < len(chunks):
        out[chunks[i].strip()] = chunks[i + 1].strip()
        i += 2
    return out


def _column_display_name(name: str, is_ru: bool) -> str:
    common = {
        "run_dir": "Запуск" if is_ru else "Run",
        "run_name": "Запуск" if is_ru else "Run",
        "run": "Запуск" if is_ru else "Run",
        "baseline": "Базовый запуск" if is_ru else "Baseline run",
        "other": "Сравниваемый запуск" if is_ru else "Compared run",
        "model": "Модель" if is_ru else "Model",
        "dataset_name": "Имя датасета" if is_ru else "Dataset name",
        "dataset_hash": "Хэш датасета" if is_ru else "Dataset hash",
        "alias": "Алиас" if is_ru else "Alias",
        "format": "Формат" if is_ru else "Format",
        "split": "Подвыборка" if is_ru else "Split",
        "target_path": "Путь артефакта" if is_ru else "Target path",
        "delta_mAP50-95": "Дельта mAP50-95" if is_ru else "Delta mAP50-95",
        "delta_Box-F1": "Дельта Box-F1" if is_ru else "Delta Box-F1",
        "delta_mAP50": "Дельта mAP50" if is_ru else "Delta mAP50",
        "delta_Box-P": "Дельта Box-P" if is_ru else "Delta Box-P",
        "delta_Box-R": "Дельта Box-R" if is_ru else "Delta Box-R",
        "delta_mask_mAP50-95": "Дельта mask mAP50-95" if is_ru else "Delta mask mAP50-95",
        "delta_mask_mAP50": "Дельта mask mAP50" if is_ru else "Delta mask mAP50",
        "delta_Mask-F1": "Дельта Mask-F1" if is_ru else "Delta Mask-F1",
        "delta_Mask-P": "Дельта Mask-P" if is_ru else "Delta Mask-P",
        "delta_Mask-R": "Дельта Mask-R" if is_ru else "Delta Mask-R",
        "test_mAP50-95": "test mAP50-95" if is_ru else "test mAP50-95",
        "test_mAP50": "test mAP50" if is_ru else "test mAP50",
        "test_Box-F1": "test Box-F1" if is_ru else "test Box-F1",
        "test_Box-P": "test Box-P" if is_ru else "test Box-P",
        "test_Box-R": "test Box-R" if is_ru else "test Box-R",
        "test_mask_mAP50-95": "test mask mAP50-95" if is_ru else "test mask mAP50-95",
        "test_mask_mAP50": "test mask mAP50" if is_ru else "test mask mAP50",
        "test_Mask-F1": "test Mask-F1" if is_ru else "test Mask-F1",
        "test_Mask-P": "test Mask-P" if is_ru else "test Mask-P",
        "test_Mask-R": "test Mask-R" if is_ru else "test Mask-R",
        "mask_mAP50-95": "mask mAP50-95" if is_ru else "mask mAP50-95",
        "mask_mAP50": "mask mAP50" if is_ru else "mask mAP50",
        "Mask-F1": "Mask-F1",
        "Mask-P": "Mask-P",
        "Mask-R": "Mask-R",
        "backend_status": "Бэкенд" if is_ru else "Backend",
        "eval_imgsz": "Размер изображения" if is_ru else "Image size",
        "eval_conf": "Порог confidence" if is_ru else "Confidence threshold",
        "eval_iou": "Порог IoU" if is_ru else "IoU threshold",
        "recommended_conf": "Рекомендованный confidence" if is_ru else "Recommended confidence",
        "target_metric": "Целевая метрика" if is_ru else "Target metric",
        "precision": "Точность" if is_ru else "Precision",
        "recall": "Полнота" if is_ru else "Recall",
        "f1": "F1",
        "status": "Статус" if is_ru else "Status",
        "performance_status": "Статус performance" if is_ru else "Performance status",
        "class_name": "Класс" if is_ru else "Class",
        "class_id": "ID класса" if is_ru else "Class ID",
        "best_run": "Лучший запуск" if is_ru else "Best run",
        "worst_run": "Худший запуск" if is_ru else "Worst run",
        "best_ap": "Лучший AP" if is_ru else "Best AP",
        "worst_ap": "Худший AP" if is_ru else "Worst AP",
        "ap_gap": "Разница AP" if is_ru else "AP gap",
        "composite_score": "Итоговый балл" if is_ru else "Composite score",
        "quality_metric": "Метрика качества" if is_ru else "Quality metric",
        "speed_metric": "Метрика скорости" if is_ru else "Speed metric",
        "epochs": "Эпохи" if is_ru else "Epochs",
        "batch_size": "Размер batch" if is_ru else "Batch size",
        "train_image_size": "Размер train image" if is_ru else "Train image size",
        "val_imgsz": "Размер val image" if is_ru else "Val image size",
        "avg_inference_ms_per_frame": "мс/кадр (инференс)" if is_ru else "ms/frame (inference)",
        "avg_total_ms_per_frame": "мс/кадр (полный)" if is_ru else "ms/frame (total)",
        "avg_inference_fps": "FPS (инференс)" if is_ru else "FPS (inference)",
        "avg_total_fps": "FPS (полный)" if is_ru else "FPS (total)",
        "throughput_img_s": "Пропускная способность, img/s" if is_ru else "Throughput, img/s",
        "latency_p50_ms": "Задержка p50, мс" if is_ru else "Latency p50, ms",
        "latency_p95_ms": "Задержка p95, мс" if is_ru else "Latency p95, ms",
        "perf_preprocess_ms_per_frame": "мс/кадр (preprocess)" if is_ru else "ms/frame (preprocess)",
        "perf_inference_ms_per_frame": "мс/кадр (inference stage)" if is_ru else "ms/frame (inference stage)",
        "perf_postprocess_ms_per_frame": "мс/кадр (postprocess)" if is_ru else "ms/frame (postprocess)",
        "perf_total_ms_per_frame": "мс/кадр (pipeline total)" if is_ru else "ms/frame (pipeline total)",
        "perf_warmup_images": "Warmup (кадров)" if is_ru else "Warmup images",
        "perf_sample_count": "Измерено кадров" if is_ru else "Measured frames",
        "perf_batch": "Batch (eval)" if is_ru else "Batch (eval)",
        "perf_device": "Устройство" if is_ru else "Device",
        "pure_inference_ms_per_frame": "мс/кадр (чистый инференс)" if is_ru else "ms/frame (pure inference)",
        "pure_inference_fps": "FPS (чистый инференс)" if is_ru else "FPS (pure inference)",
        "pipeline_end_to_end_ms_per_frame": "мс/кадр (pipeline e2e)" if is_ru else "ms/frame (pipeline e2e)",
        "pipeline_end_to_end_fps": "FPS (pipeline e2e)" if is_ru else "FPS (pipeline e2e)",
        "perf_io_load_ms_per_frame": "мс/кадр (I/O загрузка, diag)" if is_ru else "ms/frame (I/O load, diag)",
        "perf_diag_alloc_ms_per_frame": "мс/кадр (alloc, diag)" if is_ru else "ms/frame (alloc, diag)",
        "perf_diag_h2d_ms_per_frame": "мс/кадр (H2D, diag)" if is_ru else "ms/frame (H2D, diag)",
        "perf_diag_execute_ms_per_frame": "мс/кадр (execute, diag)" if is_ru else "ms/frame (execute, diag)",
        "perf_diag_d2h_ms_per_frame": "мс/кадр (D2H, diag)" if is_ru else "ms/frame (D2H, diag)",
        "perf_diag_session_init_ms": "Инициализация сессии, мс (diag)" if is_ru else "Session init, ms (diag)",
        "perf_diag_engine_init_ms": "Инициализация engine, мс (diag)" if is_ru else "Engine init, ms (diag)",
        "perf_diag_worker_wall_ms": "Время worker, мс (diag)" if is_ru else "Worker wall time, ms (diag)",
        "perf_diag_retries_count": "Повторы, шт (diag)" if is_ru else "Retries, count (diag)",
        "perf_diag_retry_sleep_ms": "Сон при повторе, мс (diag)" if is_ru else "Retry sleep, ms (diag)",
        "perf_diag_provider_switched_to_cpu": "Переключение на CPU (diag)" if is_ru else "Switched to CPU (diag)",
        "scatter_x_metric": "Ось X (метрика)" if is_ru else "X axis (metric name)",
        "scatter_x_value": "Ось X (значение)" if is_ru else "X axis (value)",
        "scatter_y_metric": "Ось Y (метрика)" if is_ru else "Y axis (metric name)",
        "scatter_y_value": "Ось Y (значение)" if is_ru else "Y axis (value)",
        "quality_source": "Источник оценки качества" if is_ru else "Quality source",
        "inference_source": "Источник инференса" if is_ru else "Inference source",
        "gt_source": "Источник разметки" if is_ru else "Ground-truth source",
        "nms_profile": "Профиль NMS" if is_ru else "NMS profile",
        "metrics_read_policy": "Политика чтения метрик" if is_ru else "Metrics read policy",
        "metrics_source": "Файл/источник метрик" if is_ru else "Metrics source",
        "performance_reason": "Причина performance" if is_ru else "Performance reason",
        "train_last_epoch": "Последняя эпоха (train)" if is_ru else "Last train epoch",
        "train_last_metrics/mAP50-95(B)": "mAP50-95 (последняя эпоха train)" if is_ru else "mAP50-95 (last train epoch)",
    }
    return common.get(name, name)


def _pretty_value(v: Any, abbreviations: dict[str, str], *, is_ru: bool, float_decimals: int = 4) -> str:
    if pd.isna(v):
        return "нет данных" if is_ru else "no data"
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.{int(max(0, float_decimals))}f}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    sv = str(v)
    if sv.strip().lower() in {"nan", "none", "null", "", "-"}:
        return "нет данных" if is_ru else "no data"
    return abbreviations.get(sv, sv)


def _diag_cell_placeholder(col: str) -> bool:
    c = str(col)
    return c.startswith("perf_diag_") or c == "perf_io_load_ms_per_frame"


def _pretty_diag_table_value(
    v: Any,
    abbreviations: dict[str, str],
    *,
    col: str,
    is_ru: bool,
    float_decimals: int,
) -> str:
    if _diag_cell_placeholder(col):
        if pd.isna(v):
            return "н/п" if is_ru else "n/a"
        if isinstance(v, float) and pd.isna(v):
            return "н/п" if is_ru else "n/a"
        sv = str(v).strip()
        if sv.lower() in {"nan", "none", "null", "", "-"}:
            return "н/п" if is_ru else "n/a"
    return _pretty_value(v, abbreviations, is_ru=is_ru, float_decimals=float_decimals)


def _drop_all_nan_columns(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0 or len(df.columns) == 0:
        return df
    keep: list[str] = []
    for c in df.columns:
        ser = df[c]
        if ser.isna().all():
            continue
        if ser.dtype == object:
            s2 = ser.astype(str).str.strip().str.lower()
            if s2.isin(["", "nan", "none", "nat"]).all():
                continue
        keep.append(str(c))
    return df[keep].copy() if keep else df.copy()


def _should_drop_split_column(rel: str, df: pd.DataFrame) -> bool:
    if "split" not in df.columns or len(df) == 0:
        return False
    low = rel.replace("\\", "/").lower()
    if "format_metrics_compare_pt_uni" in low:
        return False
    try:
        if int(df["split"].nunique(dropna=False)) > 1:
            return False
    except Exception:
        return False
    if "format_metrics_compare_test" in low or "format_metrics_compare_val" in low:
        return True
    if "format_performance_compare_test" in low or "format_performance_compare_val" in low:
        return True
    return False


def _os_display_train_profile_row(row: pd.Series) -> Any:
    parts: list[str] = []
    for x in (row.get("sys_os"), row.get("sys_os_release")):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            continue
        s = str(x).strip()
        if s.lower() in {"", "nan", "none"}:
            continue
        parts.append(s)
    if not parts:
        return None
    return " ".join(parts)


def _md_table_from_df(
    df: pd.DataFrame,
    abbreviations: dict[str, str],
    limit: int | None = None,
    *,
    is_ru: bool = True,
    float_decimals: int = 4,
    diag_style: bool = False,
) -> list[str]:
    if len(df) == 0:
        return ["_No data._"]
    preview = df.head(limit).copy() if isinstance(limit, int) and limit > 0 else df.copy()
    cols = [_column_display_name(abbreviations.get(str(c), str(c)), is_ru) for c in preview.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in preview.iterrows():
        vals = []
        for real_col in preview.columns:
            v = row.get(real_col)
            if diag_style:
                vals.append(
                    _pretty_diag_table_value(
                        v, abbreviations, col=str(real_col), is_ru=is_ru, float_decimals=float_decimals
                    )
                )
            else:
                vals.append(_pretty_value(v, abbreviations, is_ru=is_ru, float_decimals=float_decimals))
        lines.append("| " + " | ".join(vals) + " |")
    if isinstance(limit, int) and limit > 0 and len(df) > limit:
        lines.append(f"_Showing first {limit} rows out of {len(df)}._")
    return lines


def _abbrev_value(v: Any, abbreviations: dict[str, str]) -> str:
    if isinstance(v, (int, float)) or v is None:
        return str(v)
    s = str(v)
    if s in abbreviations:
        return abbreviations[s]
    if "/" in s and len(s) > 28:
        base = os.path.basename(s.rstrip("/"))
        if base:
            return abbreviations.get(base, base)
    return s


def _abbrev_df(df: pd.DataFrame, abbreviations: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_abbrev_value(c, abbreviations) for c in out.columns]
    for c in out.columns:
        if str(out[c].dtype) == "object":
            out[c] = out[c].map(lambda x: _abbrev_value(x, abbreviations))
    return out


def _select_table_columns(rel: str, df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    lower = rel.lower()
    if "compare_delta" in lower:
        preferred = [
            "baseline",
            "other",
            *compare_delta_column_names(),
        ]
    elif "leaderboard" in lower:
        preferred = [
            "model",
            "composite_score",
            "quality_metric",
            "speed_metric",
            "run_dir",
        ]
    elif "speed_quality" in lower:
        preferred = [
            "model",
            "scatter_x_metric",
            "scatter_x_value",
            "scatter_y_metric",
            "scatter_y_value",
            "quality_source",
        ]
    elif "format_metrics_compare" in lower:
        preferred = [
            "alias",
            "run_name",
            "split",
            "format",
            "backend_status",
            *format_metrics_table_column_names(),
        ]
    elif "format_performance_compare" in lower:
        preferred = [
            "alias",
            "run_name",
            "split",
            "format",
            "avg_inference_ms_per_frame",
            "avg_inference_fps",
            "latency_p95_ms",
            "perf_preprocess_ms_per_frame",
            "perf_inference_ms_per_frame",
            "perf_postprocess_ms_per_frame",
            "perf_total_ms_per_frame",
            "perf_warmup_images",
            "perf_sample_count",
            "perf_batch",
            "perf_device",
            "perf_io_load_ms_per_frame",
            "perf_diag_alloc_ms_per_frame",
            "perf_diag_h2d_ms_per_frame",
            "perf_diag_execute_ms_per_frame",
            "perf_diag_d2h_ms_per_frame",
            "perf_diag_session_init_ms",
            "perf_diag_engine_init_ms",
            "perf_diag_worker_wall_ms",
            "perf_diag_retries_count",
            "perf_diag_retry_sleep_ms",
            "perf_diag_provider_switched_to_cpu",
        ]
    elif "format_eval_settings" in lower:
        preferred = [
            "run_name",
            "split",
            "format",
            "eval_imgsz",
            "eval_conf",
            "eval_iou",
            "inference_source",
            "gt_source",
            "nms_profile",
        ]
    elif "pr_per_class" in lower:
        preferred = ["model", "class_id", "class_name", "ap", "recall", "precision"]
    elif "confidence_recommendations_" in lower:
        preferred = [
            "run_name",
            "split",
            "level",
            "class_id",
            "class_name",
            "recommended_conf",
            "target_metric",
            "precision",
            "recall",
            "f1",
            "status",
        ]
    elif "runs_summary" in lower:
        preferred = [
            "run_name",
            "model",
            "dataset_name",
            "train_last_epoch",
            "epochs",
            "batch_size",
            "train_image_size",
            "val_imgsz",
            "train_last_metrics/mAP50-95(B)",
            *runs_summary_test_column_names(),
        ]
    elif "test_system_profile" in lower:
        preferred = [
            "run_name",
            "model",
            "format",
            "test_backend",
            "test_provider",
            "sys_cpu_model",
            "sys_ram_total_gb",
            "sys_gpu_0_name",
            "sys_gpu_0_vram_gb",
            "sys_os",
            "sys_os_release",
        ]
    elif "system_profile" in lower:
        preferred = [
            "run_name",
            "model",
            "dataset_name",
            "sys_cpu_model",
            "sys_cpu_logical_cores",
            "sys_ram_total_gb",
            "sys_gpu_0_name",
            "sys_gpu_0_vram_gb",
            "sys_gpu_total_vram_gb",
            "sys_disk_fs",
            "sys_disk_total_gb",
            "sys_os",
            "sys_os_release",
        ]
    else:
        preferred = cols[:8]
    chosen = [c for c in preferred if c in cols]
    if "runs_summary" in lower:
        _rs_drop = {
            "test_Unnamed: 0",
            "test_class_name",
            "test_box-p",
            "test_box-r",
            "test_box-f1",
            "test_box-map",
            "test_box-map50",
            "test_box-map75",
        }
        chosen = [c for c in chosen if c not in _rs_drop]
    if not chosen:
        if "runs_summary" in lower:
            chosen = [c for c in ("run_name", "model", "dataset_name") if c in cols]
        if not chosen:
            chosen = cols[:8]
    out = df[chosen].copy()
    return out


def _build_pr_per_class_summary(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"model", "class_name", "ap"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    g = df.groupby(["class_name", "model"], as_index=False)["ap"].mean()
    pivot = g.pivot(index="class_name", columns="model", values="ap")
    rows: list[dict[str, Any]] = []
    for class_name, row in pivot.iterrows():
        row_valid = row.dropna()
        if len(row_valid) == 0:
            continue
        best_model = str(row_valid.idxmax())
        best_ap = float(row_valid.max())
        worst_model = str(row_valid.idxmin())
        worst_ap = float(row_valid.min())
        rows.append(
            {
                "class_name": class_name,
                "best_run": best_model,
                "best_ap": best_ap,
                "worst_run": worst_model,
                "worst_ap": worst_ap,
                "ap_gap": best_ap - worst_ap,
            }
        )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values("ap_gap", ascending=False)


def _should_hide_system_profile_table(df: pd.DataFrame) -> bool:
    sys_cols = [c for c in df.columns if str(c).startswith("sys_")]
    if not sys_cols:
        return False
    # If key hardware/os columns are absent, table is considered too sparse.
    key_cols = {"sys_cpu_model", "sys_gpu_0_name", "sys_ram_total_gb", "sys_os", "sys_os_release"}
    if not key_cols.issubset(set(sys_cols)):
        return True
    scoped = df[sys_cols].copy()
    scoped = scoped.replace("", np.nan)
    total = int(scoped.size)
    if total <= 0:
        return False
    empty = int(scoped.isna().sum().sum())
    return (empty / float(total)) > 0.70


def _system_profile_text_summary(df: pd.DataFrame, is_ru: bool) -> list[str]:
    lines: list[str] = []
    if len(df) == 0:
        return lines
    for _, row in df.iterrows():
        run_name = str(row.get("run_name") or row.get("run_dir") or "-")
        cpu = str(row.get("sys_cpu_model") or "-")
        gpu = str(row.get("sys_gpu_0_name") or "-")
        ram = row.get("sys_ram_total_gb")
        os_name = str(row.get("sys_os") or "-")
        os_rel = str(row.get("sys_os_release") or "-")
        ram_txt = "-"
        try:
            if pd.notna(ram):
                ram_txt = f"{float(ram):.1f} GB"
        except Exception:
            ram_txt = str(ram)
        if is_ru:
            lines.append(f"- `{run_name}`: CPU={cpu}; GPU={gpu}; RAM={ram_txt}; OS={os_name} {os_rel}".strip())
        else:
            lines.append(f"- `{run_name}`: CPU={cpu}; GPU={gpu}; RAM={ram_txt}; OS={os_name} {os_rel}".strip())
    return lines


def _pick_test_metric_columns(df: pd.DataFrame) -> list[str]:
    seg_cols = [
        c
        for c in runs_summary_test_column_names()
        if c.startswith("test_mask_") or c.startswith("test_Mask-")
    ]
    if any(c in df.columns for c in seg_cols):
        return [c for c in seg_cols if c in df.columns]
    det_cols = [
        c
        for c in runs_summary_test_column_names()
        if c.startswith("test_mAP") or c.startswith("test_Box")
    ]
    present = [c for c in det_cols if c in df.columns]
    if present:
        return present
    return [c for c in ("test_top1_acc", "test_top5_acc") if c in df.columns]


def _build_test_metrics_summary(df: pd.DataFrame, abbreviations: dict[str, str]) -> pd.DataFrame:
    metric_cols = _pick_test_metric_columns(df)
    present = [c for c in metric_cols if c in df.columns]
    if not present:
        return pd.DataFrame()
    run_col = "run_name" if "run_name" in df.columns else ("run_dir" if "run_dir" in df.columns else None)
    if not run_col:
        return pd.DataFrame()
    out = df[[run_col] + present].copy()
    out = out.rename(columns={run_col: "run"})
    out = _abbrev_df(out, abbreviations)
    for c in present:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _selected_run_identity(manifest: dict[str, Any]) -> tuple[set[str], set[str]]:
    selected_dirs: set[str] = set()
    selected_names: set[str] = set()
    for p in [manifest.get("baseline")] + list(manifest.get("others") or []):
        sp = str(p or "").strip()
        if not sp:
            continue
        selected_dirs.add(sp)
        selected_names.add(os.path.basename(sp.rstrip("/")))
    return selected_dirs, selected_names


def _filter_runs_summary_for_selection(df: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    if len(df) == 0:
        return df
    selected_dirs, selected_names = _selected_run_identity(manifest)
    work = df.copy()
    keep = pd.Series([False] * len(work), index=work.index)
    if "run_dir" in work.columns:
        run_dir_abs = work["run_dir"].astype(str).str.strip()
        keep = keep | run_dir_abs.isin(selected_dirs)
    if "run_name" in work.columns:
        run_name = work["run_name"].astype(str).str.strip()
        keep = keep | run_name.isin(selected_names)
    if "run_dir" in work.columns:
        run_dir_base = work["run_dir"].astype(str).str.rstrip("/").str.split("/").str[-1]
        keep = keep | run_dir_base.isin(selected_names)
    filtered = work[keep].copy()
    if len(filtered) == 0:
        return work
    return filtered


def _filter_generic_table_for_selection(df: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    if len(df) == 0:
        return df
    selected_dirs, selected_names = _selected_run_identity(manifest)
    if not selected_dirs and not selected_names:
        return df
    work = df.copy()
    keep = pd.Series([False] * len(work), index=work.index)
    candidate_cols = ["run_dir", "run_name", "model", "baseline", "other", "best_run", "worst_run"]
    for col in candidate_cols:
        if col not in work.columns:
            continue
        values = work[col].astype(str).str.strip()
        keep = keep | values.isin(selected_dirs) | values.isin(selected_names)
        keep = keep | values.str.rstrip("/").str.split("/").str[-1].isin(selected_names)
    filtered = work[keep].copy()
    if len(filtered) == 0:
        return work
    return filtered


def _quality_metric_comments(df: pd.DataFrame, is_ru: bool) -> list[str]:
    lines: list[str] = []
    if len(df) == 0 or "run" not in df.columns:
        return lines
    for metric in [c for c in df.columns if c != "run"]:
        s = pd.to_numeric(df[metric], errors="coerce")
        if s.notna().sum() < 2:
            continue
        metric_label = _column_display_name(metric, is_ru)
        best_idx = int(s.idxmax())
        worst_idx = int(s.idxmin())
        best_run = str(df.loc[best_idx, "run"])
        worst_run = str(df.loc[worst_idx, "run"])
        best_val = float(s.loc[best_idx])
        worst_val = float(s.loc[worst_idx])
        if is_ru:
            lines.append(
                f"- {metric_label}: лучший запуск **{best_run}** ({best_val:.4f}), худший **{worst_run}** ({worst_val:.4f})."
            )
        else:
            lines.append(
                f"- {metric_label}: best run **{best_run}** ({best_val:.4f}), worst **{worst_run}** ({worst_val:.4f})."
            )
    return lines


def _justify_block(text: str) -> list[str]:
    t = str(text or "").strip()
    if not t:
        return []
    return ['::: {style="text-align:justify; text-indent:1cm;"}', t, ":::", ""]


def _center_open() -> list[str]:
    return ['::: {style="text-align:center;"}']


def _center_close() -> list[str]:
    return [":::", ""]


MAX_NARRATIVE_BULLETS = 5
MAX_SPREAD_METRICS = 3

_ID_COLS_FOR_ROW_LABEL = ("run_name", "alias", "model", "run", "run_dir")
_NUMERIC_TAKEAWAY_SKIP = frozenset(
    {
        "run_name",
        "run_dir",
        "alias",
        "model",
        "split",
        "format",
        "level",
        "status",
        "class_id",
        "class_name",
        "backend_status",
        "performance_status",
        "performance_reason",
    }
)


def _subsection_intro_lines(tpl: dict[str, str], key: str) -> list[str]:
    t = str(tpl.get(key) or "").strip()
    if not t:
        return []
    return _justify_block(t)


def _preamble_template_key(kind: str) -> str:
    if kind == "alias_legend":
        return "NARR_PREAMBLE_ALIAS"
    if kind == "eval_settings":
        return "NARR_PREAMBLE_EVAL"
    return "NARR_PREAMBLE_GENERIC"


def _table_preamble_lines(
    rel: str,
    df: pd.DataFrame | None,
    kind: str,
    is_ru: bool,
    tpl: dict[str, str],
) -> list[str]:
    key = _preamble_template_key(kind)
    static = str(tpl.get(key) or tpl.get("NARR_PREAMBLE_GENERIC") or "").strip()
    out: list[str] = []
    if static:
        out.extend(_justify_block(static))
    if out and not out[-1] == "":
        out.append("")
    return out


def _row_label_from_df(df: pd.DataFrame, idx: Any) -> str:
    for c in _ID_COLS_FOR_ROW_LABEL:
        if c in df.columns:
            v = df.loc[idx, c]
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                s = str(v).strip()
                if s:
                    return s
    return str(idx)


def _numeric_spread_takeaways(df: pd.DataFrame, is_ru: bool) -> list[str]:
    lines: list[str] = []
    if df is None or len(df) < 2:
        return lines
    candidates: list[tuple[str, float, Any, Any, float, float]] = []
    for col in df.columns:
        if col in _NUMERIC_TAKEAWAY_SKIP:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() < 2:
            continue
        spread = float(s.max() - s.min())
        if spread <= 0 or not np.isfinite(spread):
            continue
        imax = s.idxmax()
        imin = s.idxmin()
        candidates.append((col, spread, imax, imin, float(s.loc[imax]), float(s.loc[imin])))
    candidates.sort(key=lambda x: -x[1])
    label_fn = lambda c: _column_display_name(c, is_ru)
    for col, _sp, imax, imin, vmax, vmin in candidates[:MAX_SPREAD_METRICS]:
        rmax = _row_label_from_df(df, imax)
        rmin = _row_label_from_df(df, imin)
        lab = label_fn(col)
        if is_ru:
            lines.append(f"- {lab}: максимум у **{rmax}** ({vmax:.4g}), минимум у **{rmin}** ({vmin:.4g}).")
        else:
            lines.append(f"- {lab}: max **{rmax}** ({vmax:.4g}), min **{rmin}** ({vmin:.4g}).")
    return lines


def _format_metrics_takeaways(df: pd.DataFrame, is_ru: bool) -> list[str]:
    lines: list[str] = []
    if df is None or len(df) == 0:
        return lines
    task_type = detect_task_type_from_columns(df.columns) or "detection"
    for col in primary_quality_columns(task_type):
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        idx = s.idxmax()
        if pd.isna(s.loc[idx]):
            continue
        row_lab = "?"
        for rc in ("alias", "format", "run_name", "model"):
            if rc in df.columns:
                v = df.loc[idx, rc]
                if v is not None and str(v).strip():
                    row_lab = str(v).strip()
                    break
        if is_ru:
            lines.append(f"- Лучший **{row_lab}** по {_column_display_name(col, is_ru)}: **{float(s.loc[idx]):.4f}**.")
        else:
            lines.append(f"- Best **{row_lab}** on {_column_display_name(col, is_ru)}: **{float(s.loc[idx]):.4f}**.")
        break
    if not lines and is_ru:
        lines.append("- Качество по выбранным колонкам недоступно для сравнения (все значения пропущены).")
    elif not lines:
        lines.append("- Quality comparison unavailable (all values missing).")
    return lines[:MAX_NARRATIVE_BULLETS]


def _perf_status_takeaways(df: pd.DataFrame, is_ru: bool) -> list[str]:
    lines: list[str] = []
    if df is None or "performance_status" not in df.columns:
        return lines
    st = df["performance_status"].astype(str).str.strip().str.lower()
    bad = int((st != "ok").sum())
    if bad > 0:
        if is_ru:
            lines.append(f"- Строк со статусом производительности не `ok`: **{bad}** из {len(df)}.")
        else:
            lines.append(f"- Rows with performance_status not `ok`: **{bad}** of {len(df)}.")
    return lines


def _speed_quality_takeaways(df: pd.DataFrame, is_ru: bool, abbreviations: dict[str, str] | None = None) -> list[str]:
    lines: list[str] = []
    abbreviations = abbreviations or {}
    if df is None or len(df) == 0:
        return lines
    x, y = "scatter_x_value", "scatter_y_value"
    mcol = "model" if "model" in df.columns else None
    if mcol and y in df.columns:
        sy = pd.to_numeric(df[y], errors="coerce")
        if sy.notna().sum() > 0:
            best = df.loc[sy.idxmax()]
            model_label = abbreviations.get(str(best[mcol]), str(best[mcol]))
            if is_ru:
                lines.append(f"- Лучший компромисс по качеству (**{y}**): **{model_label}** ({float(sy.max()):.4f}).")
            else:
                lines.append(f"- Best quality (**{y}**): **{model_label}** ({float(sy.max()):.4f}).")
    if mcol and x in df.columns:
        sx = pd.to_numeric(df[x], errors="coerce")
        if sx.notna().sum() > 0:
            fast = df.loc[sx.idxmin()]
            model_label = abbreviations.get(str(fast[mcol]), str(fast[mcol]))
            if is_ru:
                lines.append(f"- Наиболее быстрый по **{x}**: **{model_label}** ({float(sx.min()):.4f}).")
            else:
                lines.append(f"- Fastest on **{x}**: **{model_label}** ({float(sx.min()):.4f}).")
    return lines[:MAX_NARRATIVE_BULLETS]


def _leaderboard_takeaways(df: pd.DataFrame, is_ru: bool) -> list[str]:
    lines: list[str] = []
    if df is None or len(df) == 0:
        return lines
    if "composite_score" in df.columns and "model" in df.columns:
        s = pd.to_numeric(df["composite_score"], errors="coerce")
        if s.notna().sum() > 0:
            i = s.idxmax()
            m = str(df.loc[i, "model"])
            v = float(s.loc[i])
            if is_ru:
                lines.append(f"- Лидер по composite_score: **{m}** ({v:.4f}).")
            else:
                lines.append(f"- Leader by composite_score: **{m}** ({v:.4f}).")
    lines.extend(_numeric_spread_takeaways(df, is_ru))
    return lines[:MAX_NARRATIVE_BULLETS]


def _pr_summary_takeaways(df: pd.DataFrame, is_ru: bool) -> list[str]:
    lines: list[str] = []
    if df is None or len(df) == 0:
        return lines
    if {"class_name", "model", "ap"}.issubset(df.columns):
        g = df.groupby(["class_name", "model"], as_index=False)["ap"].mean()
        if len(g) > 0:
            worst = g.sort_values("ap", ascending=True).iloc[0]
            cn = str(worst["class_name"])
            md = str(worst["model"])
            apv = float(worst["ap"])
            if is_ru:
                lines.append(f"- Наименьший средний AP: класс **{cn}**, модель **{md}** ({apv:.4f}).")
            else:
                lines.append(f"- Lowest mean AP: class **{cn}**, model **{md}** ({apv:.4f}).")
    lines.extend(_numeric_spread_takeaways(df, is_ru))
    return lines[:MAX_NARRATIVE_BULLETS]


def _system_sparse_takeaways_fixed(df: pd.DataFrame | None, is_ru: bool) -> list[str]:
    if df is None or len(df) == 0:
        return []
    sys_cols = [c for c in df.columns if str(c).startswith("sys_")]
    if not sys_cols:
        return []
    filled = 0
    total = 0
    for _i, row in df.iterrows():
        for c in sys_cols:
            total += 1
            v = row.get(c)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            if str(v).strip().lower() in {"", "nan", "none"}:
                continue
            filled += 1
    ratio = filled / total if total else 0.0
    if is_ru:
        return [f"- Доля заполненных sys-полей в таблице профиля: **{ratio:.0%}** ({filled}/{total})."]
    return [f"- Share of populated sys_* fields: **{ratio:.0%}** ({filled}/{total})."]


def _table_takeaway_lines(
    rel: str,
    df: pd.DataFrame | None,
    kind: str,
    is_ru: bool,
    *,
    manifest: dict[str, Any],
    report_root: str,
    tpl: dict[str, str],
) -> list[str]:
    lines: list[str] = []
    no_data = str(tpl.get("NARR_TAKEAWAY_NO_DATA") or "").strip()
    if df is None or len(df) == 0:
        if no_data:
            lines.append(f"- {no_data}")
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind == "runs_summary_extra":
        lines.extend(_quality_metric_comments(df, is_ru))
        if len(lines) >= 2:
            return lines[:MAX_NARRATIVE_BULLETS]
        lines.extend(_numeric_spread_takeaways(df, is_ru))
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind in ("format_metrics", "format_metrics_pt_uni"):
        lines.extend(_format_metrics_takeaways(df, is_ru))
        lines.extend(_numeric_spread_takeaways(df, is_ru))
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind.startswith("perf_"):
        lines.extend(_perf_status_takeaways(df, is_ru))
        lines.extend(_numeric_spread_takeaways(df, is_ru))
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind == "speed_quality":
        lines.extend(_speed_quality_takeaways(df, is_ru))
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind == "leaderboard":
        lines.extend(_leaderboard_takeaways(df, is_ru))
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind == "pr_per_class_summary":
        lines.extend(_pr_summary_takeaways(df, is_ru))
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind in ("system_profile_train_sparse", "system_profile_train_cards", "test_system_profile", "system_profile_train"):
        lines.extend(_system_sparse_takeaways_fixed(df, is_ru))
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind == "alias_legend":
        if "alias" in df.columns:
            nu = int(df["alias"].astype(str).nunique(dropna=False))
            if is_ru:
                lines.append(f"- Число уникальных алиасов: **{nu}**.")
            else:
                lines.append(f"- Unique aliases: **{nu}**.")
        return lines[:MAX_NARRATIVE_BULLETS]

    lines.extend(_numeric_spread_takeaways(df, is_ru))
    if not lines and no_data:
        lines.append(f"- {no_data}")
    return lines[:MAX_NARRATIVE_BULLETS]


def _infer_table_kind(
    rel: str,
    *,
    is_runs_summary_extra: bool = False,
    perf_subkind: str | None = None,
) -> str:
    if perf_subkind:
        return perf_subkind
    low = rel.lower()
    if is_runs_summary_extra:
        return "runs_summary_extra"
    if "test_system_profile" in low:
        return "test_system_profile"
    if "runs_summary" in low:
        return "runs_summary"
    if "format_metrics_compare_pt_uni" in low:
        return "format_metrics_pt_uni"
    bn = os.path.basename(low)
    if "format_eval" in low or "eval_settings" in bn or ("eval" in bn and "format" in low and low.endswith(".csv")):
        return "eval_settings"
    if "format_metrics_compare" in low:
        return "format_metrics"
    if "speed_quality" in low and low.endswith(".csv"):
        return "speed_quality"
    if "leaderboard" in low:
        return "leaderboard"
    if "confidence_recommendations" in low or ("confidence" in low and "recommend" in low):
        return "confidence_class"
    if "pr_per_class" in low and low.endswith(".csv"):
        return "pr_per_class_summary"
    if "alias_legend" in low or "format_alias_legend" in low:
        return "alias_legend"
    if any(x in low for x in ("metrics", "compare", "test_metrics")):
        return "generic_metrics"
    return "unknown"


def _figure_narrative_key(rel: str) -> str:
    low = rel.replace("\\", "/").lower()
    if "compare_curves" in low:
        return "NARR_FIG_COMPARE"
    if "benchmark_bars" in low or ("benchmark" in low and low.endswith(".png")):
        return "NARR_FIG_BENCHMARK"
    if "speed_vs_map" in low:
        return "NARR_FIG_SPEED_MAP"
    if "pr_" in low or "pr_all" in low or "per_class" in low:
        return "NARR_FIG_PR"
    return "NARR_FIG_DEFAULT"


def _figure_preamble_lines(rel: str, is_ru: bool, tpl: dict[str, str]) -> list[str]:
    key = _figure_narrative_key(rel)
    t = str(tpl.get(key) or tpl.get("NARR_FIG_DEFAULT") or "").strip()
    if not t:
        return []
    return _justify_block(t)


def _figure_takeaway_lines(
    rel: str,
    is_ru: bool,
    *,
    manifest: dict[str, Any],
    report_root: str,
    tpl: dict[str, str],
) -> list[str]:
    lines: list[str] = []
    low = rel.replace("\\", "/").lower()
    rr = str(report_root or "")
    if "speed_vs_map" in low and rr:
        sq = manifest.get("speed_quality") if isinstance(manifest.get("speed_quality"), dict) else {}
        csv_rel = str((sq or {}).get("csv") or "").strip()
        if csv_rel:
            p = os.path.join(rr, csv_rel)
            if os.path.isfile(p):
                try:
                    df = pd.read_csv(p)
                    lines.extend(_speed_quality_takeaways(df, is_ru))
                except Exception:
                    pass
    if "benchmark" in low and rr:
        fmt = manifest.get("format_comparison") if isinstance(manifest.get("format_comparison"), dict) else {}
        perf_rel = str((fmt or {}).get("perf_test_csv") or "")
        if perf_rel:
            p = os.path.join(rr, perf_rel)
            if os.path.isfile(p):
                try:
                    pdf = pd.read_csv(p)
                    for col in ("avg_inference_fps", "pure_inference_fps", "throughput_img_s"):
                        if col in pdf.columns:
                            s = pd.to_numeric(pdf[col], errors="coerce")
                            if s.notna().sum() > 0:
                                i = s.idxmax()
                                lab = _row_label_from_df(pdf, i)
                                if is_ru:
                                    lines.append(f"- Максимум **{_column_display_name(col, is_ru)}**: **{lab}** ({float(s.max()):.4g}).")
                                else:
                                    lines.append(f"- Max **{_column_display_name(col, is_ru)}**: **{lab}** ({float(s.max()):.4g}).")
                                break
                except Exception:
                    pass
    if "compare_curves" in low:
        fmt = manifest.get("format_comparison") if isinstance(manifest.get("format_comparison"), dict) else {}
        if str((fmt or {}).get("test_csv") or "").strip():
            lines.extend(
                _justify_block(
                    "Сопоставьте кривые с таблицей метрик по форматам (test) в разделе выше."
                    if is_ru
                    else "Cross-check curves with the format metrics (test) table above."
                )
            )
    if _figure_narrative_key(rel) == "NARR_FIG_PR" and rr:
        pr = manifest.get("pr_per_class") if isinstance(manifest.get("pr_per_class"), dict) else {}
        pr_rel = str((pr or {}).get("csv") or "").strip()
        if pr_rel:
            p = os.path.join(rr, pr_rel)
            if os.path.isfile(p):
                try:
                    pdf = pd.read_csv(p)
                    lines.extend(_pr_summary_takeaways(pdf, is_ru))
                except Exception:
                    pass
    if not lines:
        t = str(tpl.get("NARR_TAKEAWAY_NO_DATA") or "").strip()
        if t:
            lines.append(f"- {t}")
    return lines[:MAX_NARRATIVE_BULLETS]


def _append_takeaway_bullets(lines: list[str], bullets: list[str]) -> None:
    if not bullets:
        return
    for b in bullets:
        lines.append(b)
    lines.append("")


def _load_filtered_table_df(rel: str, abs_path: str, manifest: dict[str, Any]) -> pd.DataFrame | None:
    if not abs_path or not os.path.isfile(abs_path):
        return None
    try:
        df = pd.read_csv(abs_path)
        rel_lower = rel.lower()
        if "runs_summary" in rel_lower:
            return _filter_runs_summary_for_selection(df, manifest)
        if any(k in rel_lower for k in ("leaderboard", "speed_quality", "pr_per_class")):
            return _filter_generic_table_for_selection(df, manifest)
        if "confidence_recommendations_" in rel_lower:
            df = _filter_generic_table_for_selection(df, manifest)
            if "level" in df.columns:
                df = df[df["level"].astype(str) == "global"].copy()
            for col in ("level", "class_id", "class_name"):
                if col in df.columns:
                    df = df.drop(columns=[col])
            return df
        return df
    except Exception:
        return None


def _ultralytics_completeness_lines(item: dict[str, Any], is_ru: bool) -> list[str]:
    completeness = str(item.get("completeness") or "").strip().lower()
    note = str(item.get("completeness_note") or "").strip()
    missing = item.get("missing_files") if isinstance(item.get("missing_files"), list) else []
    sources = item.get("artifact_sources") if isinstance(item.get("artifact_sources"), dict) else {}
    lines: list[str] = []
    labels = {
        "complete": ("Полнота: полный набор test-артефактов." if is_ru else "Completeness: full canonical test artifact set."),
        "partial_csv_only": (
            "Полнота: только PR CSV; графики Ultralytics отсутствуют в test-split."
            if is_ru
            else "Completeness: PR CSV only; Ultralytics plots missing from test-split."
        ),
        "train_val_fallback": (
            "Полнота: часть графиков взята из train-ultralytics (val при обучении), не test-split."
            if is_ru
            else "Completeness: some plots resolved from train-ultralytics (training val), not test-split."
        ),
        "missing": ("Полнота: артефакты Ultralytics test не найдены." if is_ru else "Completeness: no Ultralytics test artifacts found."),
    }
    if completeness in labels:
        lines.append("- " + labels[completeness])
    if note:
        lines.append("- " + note)
    if missing:
        miss_txt = ", ".join(str(x) for x in missing[:12])
        if len(missing) > 12:
            miss_txt += ", ..."
        lines.append(
            "- "
            + ("Отсутствуют обязательные файлы: " if is_ru else "Missing required files: ")
            + miss_txt
        )
    if sources:
        prov_labels = {
            "test": "test" if not is_ru else "test-split",
            "legacy": "legacy",
            "train_val_fallback": "train-val" if not is_ru else "train-val",
        }
        sample = []
        for name, prov in sorted(sources.items()):
            sample.append(f"{name}←{prov_labels.get(str(prov), prov)}")
        if sample:
            lines.append(
                "- "
                + ("Источники файлов: " if is_ru else "File sources: ")
                + ", ".join(sample[:10])
                + (" ..." if len(sample) > 10 else "")
            )
    if completeness in {"partial_csv_only", "missing", "train_val_fallback"}:
        run_name = str(item.get("run_name") or item.get("run_code") or "")
        lines.append(
            "- "
            + (
                f"Для полного test-набора выполните: `smartrain model test --run {run_name}`."
                if is_ru
                else f"For a full test artifact set run: `smartrain model test --run {run_name}`."
            )
        )
    return lines


def _ultralytics_per_class_ap_table_lines(
    *,
    report_root: str,
    csv_rel: str,
    is_ru: bool,
    table_no: int,
) -> tuple[list[str], int]:
    abs_path = os.path.join(report_root, csv_rel)
    if not os.path.isfile(abs_path):
        return [], table_no
    try:
        df = pd.read_csv(abs_path)
    except Exception:
        return [], table_no
    if len(df) == 0:
        return [], table_no
    df.columns = [str(c).strip() for c in df.columns]
    if "class_name" not in df.columns or "ap" not in df.columns:
        return [], table_no
    summary = (
        df.groupby("class_name", dropna=False)["ap"]
        .max()
        .reset_index()
        .rename(columns={"ap": "AP"})
        .sort_values("AP", ascending=False)
    )
    if len(summary) > 20:
        summary = summary.head(20)
    lines: list[str] = []
    lines.extend(_center_open())
    lines.append("")
    title = "AP по классам (Ultralytics test)" if is_ru else "Per-class AP (Ultralytics test)"
    lines.append(f"**{('Таблица' if is_ru else 'Table')} {table_no}. {title}**")
    lines.append("")
    lines.extend(_md_table_from_df(summary, {}, limit=None, is_ru=is_ru))
    lines.append("")
    lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{csv_rel}`"))
    lines.append("")
    lines.extend(_center_close())
    return lines, table_no + 1


def _csv_source_label(csv_key: str, is_ru: bool) -> str:
    key = str(csv_key or "").strip().lower()
    if key == "pr.csv":
        return (
            "Источник таблицы PR-кривой (precision/recall по порогам confidence)"
            if is_ru
            else "Data source for PR-curve table (precision/recall across confidence thresholds)"
        )
    if key == "pr_per_class.csv":
        return (
            "Источник таблицы PR по классам"
            if is_ru
            else "Data source for per-class PR table"
        )
    return ("Источник данных" if is_ru else "Data source")


def _table_title(rel: str, is_ru: bool) -> str:
    low = rel.lower()
    if "compare_delta" in low:
        return "Сравнение дельт метрик" if is_ru else "Metric delta comparison"
    if "leaderboard" in low:
        return "Рейтинг моделей" if is_ru else "Model leaderboard"
    if "speed_quality" in low:
        return "Соотношение скорости и качества" if is_ru else "Speed-quality trade-off"
    if "format_metrics_compare_test" in low:
        return "Сравнение метрик по форматам (test)" if is_ru else "Format metrics comparison (test)"
    if "format_performance_compare_test" in low:
        return "Сравнение производительности форматов (test)" if is_ru else "Format performance comparison (test)"
    if "format_metrics_compare_val" in low:
        return "Сравнение метрик по форматам (val)" if is_ru else "Format metrics comparison (val)"
    if "format_metrics_compare_pt_uni" in low:
        return "Сравнение PT и PT-uni (test/val)" if is_ru else "PT vs PT-uni comparison (test/val)"
    if "format_eval_settings" in low:
        return "Параметры расчета метрик по форматам" if is_ru else "Metric calculation settings by format"
    if "format_metrics_compare" in low:
        return "Сравнение метрик по форматам" if is_ru else "Format metrics comparison"
    if "pr_per_class" in low:
        return "Сводка AP по классам" if is_ru else "Per-class AP summary"
    if "confidence_recommendations_" in low:
        m = re.search(r"confidence_recommendations_([abc])\.csv$", low)
        suffix = m.group(1).upper() if m else "?"
        objective_map_ru = {
            "A": "A: максимум F1",
            "B": "B: F-beta (приоритет Recall)",
            "C": "C: F-beta (приоритет Precision)",
        }
        objective_map_en = {
            "A": "A: max F1",
            "B": "B: F-beta (recall-priority)",
            "C": "C: F-beta (precision-priority)",
        }
        objective_label = objective_map_ru.get(suffix, suffix) if is_ru else objective_map_en.get(suffix, suffix)
        return (
            f"Рекомендации confidence ({objective_label})"
            if is_ru
            else f"Confidence recommendations ({objective_label})"
        )
    if "runs_summary" in low:
        return (
            "Сводка параметров и метрик выбранных запусков"
            if is_ru
            else "Selected runs metrics and configuration summary"
        )
    if "test_system_profile" in low:
        return (
            "Сравнение окружения тестирования (железо)"
            if is_ru
            else "Test environment comparison (hardware)"
        )
    if "system_profile" in low:
        return (
            "Сравнение окружения обучения (железо)"
            if is_ru
            else "Training machine profile comparison"
        )
    return os.path.basename(rel)


def _figure_title(rel: str, is_ru: bool) -> str:
    low = rel.lower()
    if "compare_curves" in low:
        return "Кривые сравнения метрик по эпохам" if is_ru else "Metric comparison curves by epoch"
    if "benchmark_bars" in low:
        return "Сравнение скорости инференса" if is_ru else "Inference speed comparison"
    if "speed_vs_map" in low:
        return "Диаграмма скорость-качество" if is_ru else "Speed-vs-quality scatter"
    if "pr_all_classes" in low:
        return "PR-кривые (все классы)" if is_ru else "PR curves (all classes)"
    if "pr_class_" in low:
        m = re.search(r"pr_class_\d+_(.+)\.png$", os.path.basename(rel), flags=re.IGNORECASE)
        cls = m.group(1).replace("_", " ") if m else ""
        if cls:
            return (
                f"PR-кривая по классу: {cls}"
                if is_ru
                else f"Per-class PR curve: {cls}"
            )
        return "PR-кривая по классу" if is_ru else "Per-class PR curve"
    bn = os.path.basename(rel).lower()
    if "boxpr_curve" in bn or bn == "pr_curve.png":
        return "PR-кривая" if is_ru else "Precision-Recall curve"
    if "boxf1_curve" in bn or bn == "f1_curve.png":
        return "F1-кривая" if is_ru else "F1 curve"
    if "boxp_curve" in bn or bn == "p_curve.png":
        return "Кривая precision" if is_ru else "Precision curve"
    if "boxr_curve" in bn or bn == "r_curve.png":
        return "Кривая recall" if is_ru else "Recall curve"
    if bn == "confusion_matrix.png":
        return "Матрица ошибок" if is_ru else "Confusion matrix"
    if bn == "confusion_matrix_normalized.png":
        return "Нормализованная матрица ошибок" if is_ru else "Normalized confusion matrix"
    m_pred = re.match(r"val_batch(\d+)_pred\.jpg$", bn)
    if m_pred:
        n = int(m_pred.group(1))
        return f"Пример предсказаний (batch {n})" if is_ru else f"Prediction sample (batch {n})"
    m_lbl = re.match(r"val_batch(\d+)_labels\.jpg$", bn)
    if m_lbl:
        n = int(m_lbl.group(1))
        return f"Пример разметки (batch {n})" if is_ru else f"Label sample (batch {n})"
    return "Иллюстрация результатов" if is_ru else "Result illustration"


def _build_run_model_abbreviations(manifest: dict[str, Any], abbreviations: dict[str, str]) -> dict[str, str]:
    return dict(abbreviations)


def _path_for_report(path: str, workspace_root: str) -> str:
    p = str(path or "")
    root = str(workspace_root or "")
    if not p:
        return p
    if root:
        try:
            if os.path.abspath(p) == os.path.abspath(root):
                return "."
            return os.path.relpath(p, root)
        except Exception:
            return p
    return p


def _ordered_abbreviations(manifest: dict[str, Any], abbreviations: dict[str, str], is_ru: bool) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    used: set[str] = set()
    baseline = str(manifest.get("baseline") or "")
    if baseline:
        bn = os.path.basename(baseline.rstrip("/"))
        if bn in abbreviations:
            out.append(("baseline", bn, abbreviations[bn]))
            used.add(bn)
    others = manifest.get("others") or []
    if isinstance(others, list):
        for i, item in enumerate(others, start=1):
            bn = os.path.basename(str(item).rstrip("/"))
            if bn in abbreviations and bn not in used:
                out.append((f"other_{i}", bn, abbreviations[bn]))
                used.add(bn)
    for k, v in abbreviations.items():
        if k not in used:
            out.append(("other", k, v))
    return out


def _figure_caption(rel: str, figure_no: int, abbreviations: dict[str, str], manifest: dict[str, Any], is_ru: bool) -> str:
    base = _figure_title(rel, is_ru)
    if "compare_curves" in rel:
        baseline = os.path.basename(str(manifest.get("baseline", "")).rstrip("/"))
        others = [os.path.basename(str(x).rstrip("/")) for x in (manifest.get("others") or [])]
        b = abbreviations.get(baseline, baseline)
        o = ",".join(abbreviations.get(x, x) for x in others)
        suffix = f" ({'базовый' if is_ru else 'baseline'}: {b}; {'сравнение' if is_ru else 'others'}: {o})" if o else ""
    else:
        suffix = ""
    title = "Рисунок" if is_ru else "Figure"
    return f"{title} {figure_no}. {base}{suffix}"


def _insights_from_manifest(manifest: dict[str, Any], lang: str) -> list[str]:
    lines: list[str] = []
    ms = manifest.get("metric_sources") or {}
    sources = ms.get("sources") if isinstance(ms, dict) else {}
    recomputed = 0
    missing = 0
    missing_runtime = 0
    if isinstance(sources, dict):
        for by_metric in sources.values():
            if not isinstance(by_metric, dict):
                continue
            recomputed += sum(1 for v in by_metric.values() if v == "recomputed")
            missing += sum(1 for v in by_metric.values() if v == "missing")
    report_root = str(manifest.get("_report_root") or "")
    fmt = manifest.get("format_comparison") if isinstance(manifest.get("format_comparison"), dict) else {}
    perf_rel = str((fmt or {}).get("perf_test_csv") or "")
    if report_root and perf_rel:
        perf_csv = os.path.join(report_root, perf_rel)
        if os.path.isfile(perf_csv):
            try:
                pdf = pd.read_csv(perf_csv)
                if "performance_status" in pdf.columns:
                    statuses = pdf["performance_status"].astype(str).str.strip().str.lower()
                    missing_runtime += int((statuses != "ok").sum())
                if "performance_reason" in pdf.columns:
                    reasons = (
                        pdf["performance_reason"]
                        .astype(str)
                        .str.strip()
                        .replace("", np.nan)
                        .dropna()
                        .value_counts()
                    )
                    if len(reasons) > 0:
                        top = ", ".join(f"{k}={int(v)}" for k, v in reasons.head(3).items())
                        if lang == "ru":
                            lines.append(f"- Причины runtime-пропусков: {top}.")
                        else:
                            lines.append(f"- Runtime missing reasons: {top}.")
            except Exception:
                pass
    rec_status = ms.get("recompute_status_by_run") if isinstance(ms, dict) else None
    if isinstance(rec_status, dict) and rec_status:
        counts = pd.Series(list(rec_status.values()), dtype="object").value_counts()
        top = ", ".join(f"{str(k)}={int(v)}" for k, v in counts.items())
        if top:
            if lang == "ru":
                lines.append(f"- Статусы пересчёта: {top}.")
            else:
                lines.append(f"- Recompute statuses: {top}.")
    cache = manifest.get("cache") or {}
    hits = int(cache.get("hits", 0)) if isinstance(cache, dict) else 0
    misses = int(cache.get("misses", 0)) if isinstance(cache, dict) else 0
    if lang == "ru":
        lines.append(
            f"- Переоценённых метрик: **{recomputed}**, отсутствующих quality: **{missing}**, runtime: **{missing_runtime}**."
        )
        lines.append(f"- Кэш single-run: **hit={hits}**, **miss={misses}**.")
    else:
        lines.append(
            f"- Recomputed metrics: **{recomputed}**, missing quality: **{missing}**, missing runtime: **{missing_runtime}**."
        )
        lines.append(f"- Single-run cache: **hit={hits}**, **miss={misses}**.")
    sq = manifest.get("speed_quality") if isinstance(manifest.get("speed_quality"), dict) else {}
    sq_csv_rel = str((sq or {}).get("csv") or "")
    if report_root and sq_csv_rel:
        sq_csv = os.path.join(report_root, sq_csv_rel)
        if os.path.isfile(sq_csv):
            try:
                df = pd.read_csv(sq_csv)
                x = "scatter_x_value"
                y = "scatter_y_value"
                if {x, y, "model"}.issubset(df.columns):
                    best = df.sort_values(y, ascending=False).iloc[0]
                    fastest = df.sort_values(x, ascending=True).iloc[0]
                    if lang == "ru":
                        lines.append(f"- Лучшая quality-модель: **{best['model']}** ({best[y]:.4f}).")
                        lines.append(f"- Самая быстрая модель: **{fastest['model']}** ({fastest[x]:.2f}).")
                    else:
                        lines.append(f"- Best quality model: **{best['model']}** ({best[y]:.4f}).")
                        lines.append(f"- Fastest model: **{fastest['model']}** ({fastest[x]:.2f}).")
            except Exception:
                pass
    pr = manifest.get("pr_per_class") if isinstance(manifest.get("pr_per_class"), dict) else {}
    pr_csv_rel = str((pr or {}).get("csv") or "")
    if report_root and pr_csv_rel:
        pr_csv = os.path.join(report_root, pr_csv_rel)
        if os.path.isfile(pr_csv):
            try:
                pdf = pd.read_csv(pr_csv)
                if {"model", "class_name", "ap"}.issubset(pdf.columns):
                    grp = pdf.groupby(["model", "class_name"], as_index=False)["ap"].mean()
                    if len(grp["model"].unique()) >= 2:
                        pivot = grp.pivot(index="class_name", columns="model", values="ap")
                        best_model = grp.groupby("model", as_index=False)["ap"].mean().sort_values("ap", ascending=False).iloc[0]["model"]
                        diff = pivot.sub(pivot[best_model], axis=0).drop(columns=[best_model], errors="ignore")
                        if len(diff.columns) > 0:
                            worst_class = diff.min(axis=1).idxmin()
                            if lang == "ru":
                                lines.append(
                                    f"- Класс с наибольшей деградацией относительно **{best_model}**: **{worst_class}**."
                                )
                            else:
                                lines.append(
                                    f"- Most degraded class vs **{best_model}**: **{worst_class}**."
                                )
            except Exception:
                pass
    return lines


def _missing_reasons_from_manifest(manifest: dict[str, Any], lang: str) -> list[str]:
    lines: list[str] = []
    report_root = str(manifest.get("_report_root") or "")
    if not report_root:
        return lines
    fmt = manifest.get("format_comparison") if isinstance(manifest.get("format_comparison"), dict) else {}
    perf_rel = str((fmt or {}).get("perf_test_csv") or "")
    if perf_rel:
        perf_csv = os.path.join(report_root, perf_rel)
        if os.path.isfile(perf_csv):
            try:
                pdf = pd.read_csv(perf_csv)
                if "performance_reason" in pdf.columns:
                    perf_reasons = pdf["performance_reason"].astype(str).str.strip()
                    perf_reasons = perf_reasons.where((perf_reasons != "") & (perf_reasons != "nan"), np.nan)
                    reasons = (
                        perf_reasons
                        .dropna()
                        .value_counts()
                    )
                    if len(reasons) > 0:
                        top = ", ".join(f"{k}={int(v)}" for k, v in reasons.head(5).items())
                        lines.append(
                            ("- Performance причины: " if lang == "ru" else "- Performance reasons: ") + top
                        )
            except Exception:
                pass
    conf = manifest.get("confidence_recommendations") if isinstance(manifest.get("confidence_recommendations"), dict) else {}
    reason_counts: dict[str, int] = {}
    for rel in conf.values():
        cpath = os.path.join(report_root, str(rel))
        if not os.path.isfile(cpath):
            continue
        try:
            cdf = pd.read_csv(cpath)
            if "reason" not in cdf.columns:
                continue
            conf_reasons = cdf["reason"].astype(str).str.strip()
            conf_reasons = conf_reasons.where((conf_reasons != "") & (conf_reasons != "nan"), np.nan)
            for reason, cnt in (
                conf_reasons.dropna().value_counts().items()
            ):
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + int(cnt)
        except Exception:
            continue
    if reason_counts:
        top = ", ".join(f"{k}={v}" for k, v in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        lines.append(
            ("- Confidence причины: " if lang == "ru" else "- Confidence reasons: ") + top
        )
    failures = manifest.get("artifact_failures") if isinstance(manifest.get("artifact_failures"), list) else []
    if failures:
        by_reason: dict[str, int] = {}
        for item in failures:
            if not isinstance(item, dict):
                continue
            code = str(item.get("reason_code") or "unknown").strip() or "unknown"
            by_reason[code] = by_reason.get(code, 0) + 1
        if by_reason:
            top = ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items(), key=lambda x: x[1], reverse=True)[:8])
            lines.append(("- Диагностические причины: " if lang == "ru" else "- Diagnostic reasons: ") + top)
    return lines


def _perf_not_collected_hint_lines(manifest: dict[str, Any], is_ru: bool, tpl: dict[str, str]) -> list[str]:
    failures = manifest.get("artifact_failures") if isinstance(manifest.get("artifact_failures"), list) else []
    has_perf_gap = any(
        isinstance(item, dict) and str(item.get("reason_code") or "") == "perf_not_collected_for_target"
        for item in failures
    )
    if not has_perf_gap:
        return []
    text = str(tpl.get("NARR_PERF_NOT_COLLECTED") or "").strip()
    if not text:
        return []
    return _justify_block(text)


def _discover_missing_pr_images(report_root: str, manifest_images: list[str]) -> list[str]:
    discovered: list[str] = []
    known = {str(x) for x in manifest_images}
    pr_root = os.path.join(report_root, "artifacts", "pr")
    if not os.path.isdir(pr_root):
        return discovered
    for root, _dirs, files in os.walk(pr_root):
        for name in files:
            if not name.lower().endswith(".png"):
                continue
            abs_path = os.path.join(root, name)
            rel = os.path.relpath(abs_path, report_root)
            if rel in known:
                continue
            if "per_class" in rel or "pr_all_classes" in name:
                discovered.append(rel)
    return sorted(discovered)


def _append_speed_quality_table(
    lines: list[str],
    *,
    manifest: dict[str, Any],
    report_root: str,
    abbreviations: dict[str, str],
    is_ru: bool,
    table_no: int,
    tpl: dict[str, str],
    emit_before_table: Callable[[], None] | None = None,
) -> int:
    sq = manifest.get("speed_quality") if isinstance(manifest.get("speed_quality"), dict) else {}
    rel = str((sq or {}).get("csv") or "").strip()
    if not rel or not report_root:
        return table_no
    abs_path = os.path.join(report_root, rel)
    if not os.path.isfile(abs_path):
        return table_no
    try:
        df = pd.read_csv(abs_path)
        df = _filter_generic_table_for_selection(df, manifest)
        df = _select_table_columns(rel, df)
        df = _abbrev_df(df, abbreviations)
    except Exception:
        return table_no
    if emit_before_table is not None:
        emit_before_table()
    lines.extend(_table_preamble_lines(rel, df, "speed_quality", is_ru, tpl))
    lines.extend(_center_open())
    lines.append("")
    lines.append(f"**{'Таблица' if is_ru else 'Table'} {table_no}. {_table_title(rel, is_ru)}**")
    lines.append("")
    lines.extend(_md_table_from_df(df, abbreviations, limit=None, is_ru=is_ru))
    lines.append("")
    lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{rel}`"))
    lines.append("")
    _append_takeaway_bullets(
        lines,
        _table_takeaway_lines(
            rel,
            df,
            "speed_quality",
            is_ru,
            manifest=manifest,
            report_root=str(report_root),
            tpl=tpl,
        ),
    )
    lines.extend(_center_close())
    return table_no + 1


def _build_markdown_lines(manifest: dict[str, Any], lang: str) -> list[str]:
    is_ru = lang == "ru"
    tpl = _read_template(lang)
    abbreviations = (manifest.get("abbreviations") or {}) if isinstance(manifest, dict) else {}
    if not isinstance(abbreviations, dict):
        abbreviations = {}
    abbreviations = _build_run_model_abbreviations(manifest, abbreviations)
    section_titles = {
        "exec": "Краткое резюме" if is_ru else "Executive Summary",
        "context": "Контекст и цель сравнения" if is_ru else "Comparison Context",
        "quality": "Анализ качества" if is_ru else "Quality Analysis",
        "speed": "Анализ скорости" if is_ru else "Speed Analysis",
        "format_compare": "Сравнение форматов моделей" if is_ru else "Model Format Comparison",
        "per_class": "Анализ по классам" if is_ru else "Per-class Analysis",
        "ultra": "Результаты Ultralytics test" if is_ru else "Ultralytics Test Results",
        "conclusion": "Заключение и рекомендации" if is_ru else "Conclusions and Actions",
    }
    section_order = ["context", "quality", "speed", "format_compare", "per_class", "ultra", "conclusion", "exec"]
    section_index = {k: i + 1 for i, k in enumerate(section_order)}
    def _sec(key: str) -> str:
        return f"## {section_index[key]}. {section_titles[key]}"
    lines: list[str] = ["# " + ("Аналитический отчёт" if is_ru else "Analyze report"), ""]
    workspace_root = str(manifest.get("workspace_root") or os.getcwd())
    lines.append(f"- {('Сгенерировано' if is_ru else 'Generated')}: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- {('Сессия' if is_ru else 'Session')}: `{manifest.get('session_name', '')}`")
    lines.append(f"- {('Профиль' if is_ru else 'Profile')}: `{manifest.get('profile', '')}`")
    lines.append(f"- {('Рабочая папка' if is_ru else 'Workspace root')}: `{workspace_root}`")
    lines.append("")
    report_root = manifest.get("_report_root") or ""
    images = list(manifest.get("images") or [])
    if report_root:
        fallback_pr_images = _discover_missing_pr_images(report_root, images)
        if fallback_pr_images:
            images.extend(fallback_pr_images)
            images = sorted(set(images))
            failures = manifest.setdefault("artifact_failures", [])
            if isinstance(failures, list):
                for rel in fallback_pr_images:
                    failures.append(
                        {
                            "stage": "report",
                            "status": "recovered",
                            "reason_code": "manifest_missed_artifact",
                            "reason_detail": rel,
                            "run_dir": "",
                            "format": "",
                            "split": "",
                        }
                    )
    early_figure1_rel: str | None = None
    if report_root:
        for _img in images:
            if not isinstance(_img, str):
                continue
            if not any(k in _img for k in ("compare", "inference", "speed_quality")):
                continue
            if "speed_vs_map" in _img.replace("\\", "/").lower():
                continue
            if os.path.isfile(os.path.join(str(report_root), _img)):
                early_figure1_rel = _img
                break
    lines.append(_sec("context"))
    lines.append("")
    if tpl.get("INTRO"):
        lines.extend(_justify_block(tpl["INTRO"]))
    baseline = str(manifest.get("baseline", "") or "")
    baseline_name = os.path.basename(baseline.rstrip("/")) if baseline else ""
    baseline_abbr = abbreviations.get(baseline_name, "")
    baseline_display = _path_for_report(baseline, workspace_root)
    if baseline_abbr:
        lines.append(f"- {('Базовый' if is_ru else 'Baseline')} ({baseline_abbr}): `{baseline_display}`")
    else:
        lines.append(f"- {('Базовый' if is_ru else 'Baseline')}: `{baseline_display}`")
    others = manifest.get("others") or []
    single_run_mode = bool(manifest.get("single_run_mode")) or (
        isinstance(others, list) and len(others) == 0 and bool(baseline)
    )
    if single_run_mode:
        lines.append(
            "- "
            + (
                "Режим: отчёт по одному run (без сравнения с кандидатами)."
                if is_ru
                else "Mode: single-run report (no candidate comparison)."
            )
        )
    if isinstance(others, list):
        for item in others:
            item_s = str(item)
            item_name = os.path.basename(item_s.rstrip("/"))
            item_abbr = abbreviations.get(item_name, "")
            item_display = _path_for_report(item_s, workspace_root)
            if item_abbr:
                lines.append(f"- {('Кандидат' if is_ru else 'Candidate')} ({item_abbr}): `{item_display}`")
            else:
                lines.append(f"- {('Кандидат' if is_ru else 'Candidate')}: `{item_display}`")
    dataset_pairs: list[str] = []
    for k, v in abbreviations.items():
        if str(v).startswith("D"):
            dataset_pairs.append(f"{v} = {k}")
    if dataset_pairs:
        lines.append(
            "- "
            + ("Датасеты: " if is_ru else "Datasets: ")
            + "; ".join(sorted(dataset_pairs))
        )
    lines.append("")
    context_idx = section_index["context"]
    lines.append(
        f"### {context_idx}.1 " + ("Датасет" if is_ru else "Dataset")
    )
    lines.append("")
    lines.extend(_subsection_intro_lines(tpl, "SUB_CONTEXT_DATASET"))
    if dataset_pairs:
        for pair in sorted(dataset_pairs):
            lines.append(f"- {pair}")
    lines.append("")
    lines.append(
        f"### {context_idx}.2 " + ("Модели и артефакты" if is_ru else "Models and artifacts")
    )
    lines.append("")
    lines.extend(_subsection_intro_lines(tpl, "SUB_CONTEXT_MODELS"))
    fmt_cmp = manifest.get("format_comparison") if isinstance(manifest.get("format_comparison"), dict) else {}
    alias_rel = str(fmt_cmp.get("alias_legend_csv") or "")
    eval_rel = str(fmt_cmp.get("eval_csv") or "")
    table_no = 1
    figure_no = 1
    early_figure1_inserted = False
    early_figure1_emitted_in_doc = False

    def _emit_figure_1_before_table_11() -> None:
        nonlocal figure_no, early_figure1_inserted, early_figure1_emitted_in_doc
        if early_figure1_inserted or table_no != 11:
            return
        early_figure1_inserted = True
        rel_e = early_figure1_rel
        if not rel_e or not report_root:
            return
        if not os.path.isfile(os.path.join(str(report_root), rel_e)):
            return
        lines.extend(_figure_preamble_lines(rel_e, is_ru, tpl))
        lines.extend(_center_open())
        lines.append(f"![]({os.path.join('..', rel_e)}){{ width=95% }}")
        lines.append(f"*{_figure_caption(rel_e, figure_no, abbreviations, manifest, is_ru)}*")
        figure_no += 1
        lines.append("")
        lines.extend(_center_close())
        _append_takeaway_bullets(
            lines,
            _figure_takeaway_lines(rel_e, is_ru, manifest=manifest, report_root=str(report_root), tpl=tpl),
        )
        early_figure1_emitted_in_doc = True

    if alias_rel:
        alias_abs = os.path.join(report_root, alias_rel)
        if os.path.isfile(alias_abs):
            try:
                alias_df = pd.read_csv(alias_abs)
                alias_df = _filter_generic_table_for_selection(alias_df, manifest)
                preferred_alias = [c for c in ("alias", "run_name", "target_path") if c in alias_df.columns]
                if preferred_alias:
                    alias_df = alias_df[preferred_alias]
                dedup_cols = [c for c in ("run_name", "target_path") if c in alias_df.columns]
                if dedup_cols:
                    alias_df = alias_df.sort_values(by=[c for c in ("run_name", "alias") if c in alias_df.columns])
                    alias_df = alias_df.drop_duplicates(subset=dedup_cols, keep="first")
                alias_df = _abbrev_df(alias_df, abbreviations)
                _emit_figure_1_before_table_11()
                ak = _infer_table_kind(alias_rel)
                lines.extend(_table_preamble_lines(alias_rel, alias_df, ak, is_ru, tpl))
                lines.extend(_center_open())
                lines.append("")
                lines.append(
                    f"**{'Таблица' if is_ru else 'Table'} {table_no}. "
                    + ("Легенда алиасов форматов" if is_ru else "Format alias legend")
                    + "**"
                )
                lines.append("")
                lines.extend(_md_table_from_df(alias_df, abbreviations, limit=None, is_ru=is_ru))
                lines.append("")
                lines.append("<!-- alias_legend_columns: alias,run_name,target_path -->")
                lines.append("")
                lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{alias_rel}`"))
                _append_takeaway_bullets(
                    lines,
                    _table_takeaway_lines(
                        alias_rel, alias_df, ak, is_ru, manifest=manifest, report_root=str(report_root), tpl=tpl
                    ),
                )
                lines.extend(_center_close())
                table_no += 1
            except Exception:
                pass
    if eval_rel:
        eval_abs = os.path.join(report_root, eval_rel)
        if os.path.isfile(eval_abs):
            try:
                eval_df = pd.read_csv(eval_abs)
                eval_df = _filter_generic_table_for_selection(eval_df, manifest)
                keep_cols = [c for c in ("alias", "run_name", "split", "format", "eval_imgsz", "eval_conf", "eval_iou") if c in eval_df.columns]
                if keep_cols:
                    eval_df = eval_df[keep_cols]
                eval_df = eval_df.drop_duplicates()
                eval_df = _abbrev_df(eval_df, abbreviations)
                _emit_figure_1_before_table_11()
                ek = _infer_table_kind(eval_rel)
                lines.extend(_table_preamble_lines(eval_rel, eval_df, ek, is_ru, tpl))
                lines.extend(_center_open())
                lines.append("")
                lines.append(
                    f"**{'Таблица' if is_ru else 'Table'} {table_no}. "
                    + ("Параметры расчета метрик по форматам" if is_ru else "Metric calculation settings by format")
                    + "**"
                )
                lines.append("")
                lines.extend(_md_table_from_df(eval_df, abbreviations, limit=None, is_ru=is_ru))
                lines.append("")
                lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{eval_rel}`"))
                _append_takeaway_bullets(
                    lines,
                    _table_takeaway_lines(
                        eval_rel, eval_df, ek, is_ru, manifest=manifest, report_root=str(report_root), tpl=tpl
                    ),
                )
                lines.extend(_center_close())
                table_no += 1
            except Exception:
                pass
    lines.append("")
    lines.append(_sec("quality"))
    lines.append("")
    quality_idx = section_index["quality"]
    lines.append(f"### {quality_idx}.1 " + ("Общие метрики" if is_ru else "General metrics"))
    lines.append("")
    lines.extend(_subsection_intro_lines(tpl, "SUB_QUALITY_GENERAL"))
    if tpl.get("QUALITY"):
        lines.extend(_justify_block(tpl["QUALITY"]))
    tables = manifest.get("tables") or []
    env_subsection_opened = False
    leaderboard_rel = ""
    run_card_intro_emitted = False
    for rel in tables:
        if not isinstance(rel, str):
            continue
        if "format_metrics_compare" in rel.lower() or "format_eval_settings" in rel.lower():
            # Format-comparison tables/settings are rendered in section 4 only.
            continue
        if "speed_quality" in rel.lower() and rel.lower().endswith(".csv"):
            continue
        if "leaderboard" in rel.lower():
            leaderboard_rel = rel
            continue
        abs_path = os.path.join(report_root, rel) if report_root else ""
        if not any(
            k in rel
            for k in ("compare", "leaderboard", "metrics", "speed_quality", "pr_per_class", "system_profile", "runs_summary")
        ):
            if "confidence_recommendations_" not in rel:
                continue
        if "confidence_recommendations_" in rel:
            # In quality section show only global rows.
            pass
        title = os.path.basename(rel)
        if "pr_per_class" in rel:
            # raw long-format table is too noisy; use aggregated class metrics below
            continue
        if ("system_profile" in rel.lower() or "test_system_profile" in rel.lower()) and not env_subsection_opened:
            lines.append(f"### {quality_idx}.2 " + ("Сравнение окружения" if is_ru else "Environment comparison"))
            lines.append("")
            lines.extend(_subsection_intro_lines(tpl, "SUB_QUALITY_ENV"))
            env_subsection_opened = True
        df_preview = _load_filtered_table_df(rel, abs_path, manifest)
        kind_preview = _infer_table_kind(rel)
        _emit_figure_1_before_table_11()
        lines.extend(_table_preamble_lines(rel, df_preview, kind_preview, is_ru, tpl))
        lines.extend(_center_open())
        lines.append("")
        lines.append(f"**{'Таблица' if is_ru else 'Table'} {table_no}. {_table_title(rel, is_ru)}**")
        lines.append("")
        table_no += 1
        if abs_path and os.path.isfile(abs_path):
            try:
                if df_preview is not None:
                    df = df_preview.copy()
                else:
                    df = pd.read_csv(abs_path)
                rel_lower = rel.lower()
                if "runs_summary" in rel_lower:
                    df = _filter_runs_summary_for_selection(df, manifest)
                elif any(k in rel_lower for k in ("leaderboard", "speed_quality", "pr_per_class")):
                    df = _filter_generic_table_for_selection(df, manifest)
                elif "confidence_recommendations_" in rel_lower:
                    df = _filter_generic_table_for_selection(df, manifest)
                    if "level" in df.columns:
                        df = df[df["level"].astype(str) == "global"].copy()
                    for col in ("level", "class_id", "class_name"):
                        if col in df.columns:
                            df = df.drop(columns=[col])
                if "system_profile_compare.csv" in rel_lower and "test_system_profile" not in rel_lower:
                    if _should_hide_system_profile_table(df):
                        lines.append(
                            "- "
                            + (
                                "Системный профиль не показан: в данных слишком много пропусков по полям hardware/os."
                                if is_ru
                                else "System profile table hidden: data is too sparse in hardware/os fields."
                            )
                        )
                        lines.append("")
                        if not run_card_intro_emitted:
                            lines.extend(_subsection_intro_lines(tpl, "SUB_QUALITY_RUN_CARD"))
                            run_card_intro_emitted = True
                        for _, row in df.iterrows():
                            run_label = str(row.get("run_name") or row.get("run_dir") or "-")
                            run_code = abbreviations.get(run_label, run_label)
                            if run_code != run_label:
                                lines.append(
                                    f"**{('Запуск' if is_ru else 'Run')} `{run_code}` ({run_label})**"
                                )
                            else:
                                lines.append(f"**{('Запуск' if is_ru else 'Run')} `{run_label}`**")
                            lines.append("")
                            cpu_v = row.get("sys_cpu_model")
                            gpu_v = row.get("sys_gpu_0_name")
                            ram_v = row.get("sys_ram_total_gb")
                            os_v = _os_display_train_profile_row(row)
                            card = pd.DataFrame(
                                [
                                    {
                                        "Параметр" if is_ru else "Parameter": "CPU",
                                        "Значение" if is_ru else "Value": cpu_v,
                                    },
                                    {
                                        "Параметр" if is_ru else "Parameter": "GPU",
                                        "Значение" if is_ru else "Value": gpu_v,
                                    },
                                    {
                                        "Параметр" if is_ru else "Parameter": "RAM, GB",
                                        "Значение" if is_ru else "Value": ram_v,
                                    },
                                    {
                                        "Параметр" if is_ru else "Parameter": "OS",
                                        "Значение" if is_ru else "Value": os_v,
                                    },
                                ]
                            )
                            all_na = True
                            for key in (cpu_v, gpu_v, ram_v, os_v):
                                if key is None or (isinstance(key, float) and pd.isna(key)):
                                    continue
                                if str(key).strip().lower() in {"", "nan", "none"}:
                                    continue
                                all_na = False
                                break
                            if all_na:
                                status_msg = (
                                    "В артефактах run отсутствует training_metadata.json или блок system_profile."
                                    if is_ru
                                    else "training_metadata.json or system_profile block is missing in run artifacts."
                                )
                                card = pd.DataFrame(
                                    [
                                        {
                                            "Параметр" if is_ru else "Parameter": ("Статус" if is_ru else "Status"),
                                            "Значение" if is_ru else "Value": status_msg,
                                        },
                                    ]
                                )
                            lines.extend(_md_table_from_df(card, abbreviations, limit=None, is_ru=is_ru))
                            lines.append("")
                        lines.append(
                            "- "
                            + (
                                "См. таблицу окружения тестирования ниже."
                                if is_ru
                                else "See the test environment comparison table below."
                            )
                        )
                        lines.append("")
                        lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{rel}`"))
                        _append_takeaway_bullets(
                            lines,
                            _table_takeaway_lines(
                                rel,
                                df,
                                "system_profile_train_sparse",
                                is_ru,
                                manifest=manifest,
                                report_root=str(report_root),
                                tpl=tpl,
                            ),
                        )
                        lines.extend(_center_close())
                        continue
                    lines.append(
                        "#### " + ("Профиль запуска " if is_ru else "Run profile ") + "(train)"
                    )
                    lines.append("")
                    if not run_card_intro_emitted:
                        lines.extend(_subsection_intro_lines(tpl, "SUB_QUALITY_RUN_CARD"))
                        run_card_intro_emitted = True
                    for _, row in df.iterrows():
                        run_label = str(row.get("run_name") or row.get("run_dir") or "-")
                        run_code = abbreviations.get(run_label, run_label)
                        if run_code != run_label:
                            lines.append(
                                f"**{('Запуск' if is_ru else 'Run')} `{run_code}` ({run_label})**"
                            )
                        else:
                            lines.append(f"**{('Запуск' if is_ru else 'Run')} `{run_label}`**")
                        lines.append("")
                        card = pd.DataFrame(
                            [
                                {"Параметр" if is_ru else "Parameter": "CPU", "Значение" if is_ru else "Value": row.get("sys_cpu_model")},
                                {"Параметр" if is_ru else "Parameter": "GPU", "Значение" if is_ru else "Value": row.get("sys_gpu_0_name")},
                                {"Параметр" if is_ru else "Parameter": "RAM, GB", "Значение" if is_ru else "Value": row.get("sys_ram_total_gb")},
                                {"Параметр" if is_ru else "Parameter": "OS", "Значение" if is_ru else "Value": _os_display_train_profile_row(row)},
                            ]
                        )
                        lines.extend(_md_table_from_df(card, abbreviations, limit=None, is_ru=is_ru))
                        lines.append("")
                    lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{rel}`"))
                    _append_takeaway_bullets(
                        lines,
                        _table_takeaway_lines(
                            rel,
                            df,
                            "system_profile_train_cards",
                            is_ru,
                            manifest=manifest,
                            report_root=str(report_root),
                            tpl=tpl,
                        ),
                    )
                    lines.extend(_center_close())
                    continue
                if "test_system_profile" in rel_lower:
                    grouped = df.groupby("run_name", dropna=False) if "run_name" in df.columns else [("-", df)]
                    for run_name, g in grouped:
                        if not run_card_intro_emitted:
                            lines.extend(_subsection_intro_lines(tpl, "SUB_QUALITY_RUN_CARD"))
                            run_card_intro_emitted = True
                        rn = str(run_name)
                        run_code = abbreviations.get(rn, rn)
                        if run_code != rn:
                            lines.append(f"#### {('Запуск' if is_ru else 'Run')} `{run_code}` ({rn})")
                        else:
                            lines.append(f"#### {('Запуск' if is_ru else 'Run')} `{rn}`")
                        lines.append("")
                        model_name = str(g.iloc[0].get("model") or "-") if len(g) > 0 and "model" in g.columns else "-"
                        card_rows = [
                            {"Параметр" if is_ru else "Parameter": ("Модель" if is_ru else "Model"), "Значение" if is_ru else "Value": model_name},
                            {"Параметр" if is_ru else "Parameter": ("Форматы" if is_ru else "Formats"), "Значение" if is_ru else "Value": ", ".join(sorted(set(g.get("format", pd.Series(dtype=str)).astype(str))))},
                            {"Параметр" if is_ru else "Parameter": "CPU", "Значение" if is_ru else "Value": g.iloc[0].get("sys_cpu_model") if "sys_cpu_model" in g.columns else None},
                            {"Параметр" if is_ru else "Parameter": "GPU", "Значение" if is_ru else "Value": g.iloc[0].get("sys_gpu_0_name") if "sys_gpu_0_name" in g.columns else None},
                            {"Параметр" if is_ru else "Parameter": "RAM, GB", "Значение" if is_ru else "Value": g.iloc[0].get("sys_ram_total_gb") if "sys_ram_total_gb" in g.columns else None},
                            {"Параметр" if is_ru else "Parameter": "OS", "Значение" if is_ru else "Value": _os_display_train_profile_row(g.iloc[0]) if len(g) > 0 else None},
                        ]
                        card = pd.DataFrame(card_rows)
                        lines.extend(_md_table_from_df(card, abbreviations, limit=None, is_ru=is_ru))
                        lines.append("")
                    lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{rel}`"))
                    _append_takeaway_bullets(
                        lines,
                        _table_takeaway_lines(
                            rel, df, "test_system_profile", is_ru, manifest=manifest, report_root=str(report_root), tpl=tpl
                        ),
                    )
                    lines.extend(_center_close())
                    continue
                df = _select_table_columns(rel, df)
                if "system_profile" in rel.lower() and _should_hide_system_profile_table(df):
                    lines.append(
                        "- "
                        + (
                            "Системный профиль не показан: в данных слишком много пропусков по полям hardware/os."
                            if is_ru
                            else "System profile table hidden: data is too sparse in hardware/os fields."
                        )
                    )
                    lines.append("")
                    for _, row in df.iterrows():
                        run_label = str(row.get("run_name") or row.get("run_dir") or "-")
                        run_code = abbreviations.get(run_label, run_label)
                        if run_code != run_label:
                            lines.append(
                                f"**{('Запуск' if is_ru else 'Run')} `{run_code}` ({run_label})**"
                            )
                        else:
                            lines.append(f"**{('Запуск' if is_ru else 'Run')} `{run_label}`**")
                        lines.append("")
                        cpu_v = row.get("sys_cpu_model")
                        gpu_v = row.get("sys_gpu_0_name")
                        ram_v = row.get("sys_ram_total_gb")
                        os_v = _os_display_train_profile_row(row)
                        card = pd.DataFrame(
                            [
                                {"Параметр" if is_ru else "Parameter": "CPU", "Значение" if is_ru else "Value": cpu_v},
                                {"Параметр" if is_ru else "Parameter": "GPU", "Значение" if is_ru else "Value": gpu_v},
                                {"Параметр" if is_ru else "Parameter": "RAM, GB", "Значение" if is_ru else "Value": ram_v},
                                {"Параметр" if is_ru else "Parameter": "OS", "Значение" if is_ru else "Value": os_v},
                            ]
                        )
                        all_na = True
                        for key in (cpu_v, gpu_v, ram_v, os_v):
                            if key is None or (isinstance(key, float) and pd.isna(key)):
                                continue
                            if str(key).strip().lower() in {"", "nan", "none"}:
                                continue
                            all_na = False
                            break
                        if all_na:
                            status_msg = (
                                "В артефактах run отсутствует training_metadata.json или блок system_profile."
                                if is_ru
                                else "training_metadata.json or system_profile block is missing in run artifacts."
                            )
                            card = pd.DataFrame(
                                [
                                    {
                                        "Параметр" if is_ru else "Parameter": ("Статус" if is_ru else "Status"),
                                        "Значение" if is_ru else "Value": status_msg,
                                    },
                                ]
                            )
                        lines.extend(_md_table_from_df(card, abbreviations, limit=None, is_ru=is_ru))
                        lines.append("")
                    lines.append(
                        "- "
                        + (
                            "См. таблицу окружения тестирования ниже."
                            if is_ru
                            else "See the test environment comparison table below."
                        )
                    )
                    lines.append("")
                    lines.append(
                        ("_Источник данных:_ " if is_ru else "_Data source:_ ")
                        + f"`{rel}`"
                    )
                    _append_takeaway_bullets(
                        lines,
                        _table_takeaway_lines(
                            rel,
                            df,
                            "system_profile_train_sparse",
                            is_ru,
                            manifest=manifest,
                            report_root=str(report_root),
                            tpl=tpl,
                        ),
                    )
                    lines.extend(_center_close())
                    continue
                df = _abbrev_df(df, abbreviations)
                lines.extend(_md_table_from_df(df, abbreviations, limit=None, is_ru=is_ru))
                lines.append("")
                lines.append(
                    ("_Источник данных:_ " if is_ru else "_Data source:_ ")
                    + f"`{rel}`"
                )
                tk_kind = "system_profile_train" if "system_profile" in rel.lower() else kind_preview
                _append_takeaway_bullets(
                    lines,
                    _table_takeaway_lines(rel, df, tk_kind, is_ru, manifest=manifest, report_root=str(report_root), tpl=tpl),
                )
            except Exception as e:
                lines.append(f"- {('Ошибка чтения' if is_ru else 'Read error')}: {e}")
        else:
            lines.append(f"- {('Файл не найден' if is_ru else 'File not found')}")
        lines.extend(_center_close())
        if "runs_summary" in rel.lower():
            try:
                full_df = pd.read_csv(abs_path)
                full_df = _filter_runs_summary_for_selection(full_df, manifest)
                test_summary = _build_test_metrics_summary(full_df, abbreviations)
                if len(test_summary) > 0:
                    _emit_figure_1_before_table_11()
                    lines.extend(
                        _table_preamble_lines(
                            rel, test_summary, "runs_summary_extra", is_ru, tpl
                        )
                    )
                    lines.extend(_center_open())
                    lines.append("")
                    lines.append(
                        f"**{'Таблица' if is_ru else 'Table'} {table_no}. "
                        + ("Сводка test-метрик Ultralytics" if is_ru else "Ultralytics test metrics summary")
                        + "**"
                    )
                    lines.append("")
                    lines.extend(_md_table_from_df(test_summary, abbreviations, limit=None, is_ru=is_ru))
                    lines.append("")
                    lines.append(
                        ("_Источник данных:_ " if is_ru else "_Data source:_ ")
                        + f"`{rel}`"
                    )
                    lines.append("")
                    _append_takeaway_bullets(
                        lines,
                        _table_takeaway_lines(
                            rel,
                            test_summary,
                            "runs_summary_extra",
                            is_ru,
                            manifest=manifest,
                            report_root=str(report_root),
                            tpl=tpl,
                        ),
                    )
                    lines.extend(_center_close())
                    table_no += 1
            except Exception:
                pass
    # images list may be extended by fallback PR image discovery.
    lines.append(_sec("format_compare"))
    lines.append("")
    format_idx = section_index["format_compare"]
    lines.append(f"### {format_idx}.1 " + ("Сравнение метрик качества" if is_ru else "Quality metrics comparison"))
    lines.append("")
    lines.extend(_subsection_intro_lines(tpl, "SUB_FORMAT_QUALITY"))
    # Optional deep-diagnostics report (generated by scripts/deep_diagnostics_onnx_map50_95.py).
    baseline_root = str(manifest.get("baseline") or "")
    if baseline_root:
        deep_md = os.path.join(baseline_root, "deep_diagnostics_report", "deep_diagnostics_report.md")
        if os.path.isfile(deep_md):
            lines.append("### " + ("Deep diagnostics (подробный анализ)" if is_ru else "Deep diagnostics (detailed analysis)"))
            lines.append("")
            lines.extend(_subsection_intro_lines(tpl, "SUB_FORMAT_DEEP_DIAG"))
            lines.append(
                "- "
                + (
                    "Отчёт deep-диагностики: "
                    if is_ru
                    else "Deep diagnostics report: "
                )
                + f"`{_path_for_report(deep_md, workspace_root)}`"
            )
            lines.append("")
    fmt_cmp = manifest.get("format_comparison") if isinstance(manifest.get("format_comparison"), dict) else {}
    issues_rel = str(fmt_cmp.get("issues_json") or "")
    if issues_rel:
        issues_abs = os.path.join(report_root, issues_rel)
        if os.path.isfile(issues_abs):
            try:
                with open(issues_abs, "r", encoding="utf-8") as f:
                    issues_payload = json.load(f)
            except Exception:
                issues_payload = []
            if isinstance(issues_payload, list) and issues_payload:
                lines.append("### " + ("Проблемы вычисления форматов" if is_ru else "Format evaluation issues"))
                lines.append("")
                lines.extend(_subsection_intro_lines(tpl, "SUB_FORMAT_ISSUES"))
                lines.append(
                    (
                        "- Сводка причин (код -> количество):"
                        if is_ru
                        else "- Reason summary (code -> count):"
                    )
                )
                reason_counts: dict[str, int] = {}
                for item in issues_payload:
                    if not isinstance(item, dict):
                        continue
                    reason_code = str(item.get("reason_code") or "unknown").strip() or "unknown"
                    reason_counts[reason_code] = int(reason_counts.get(reason_code, 0)) + 1
                for code, cnt in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
                    lines.append(f"  - `{code}`: {cnt}")
                lines.append("")
                lines.append(
                    (
                        "- Краткая сводка по split/format:"
                        if is_ru
                        else "- Compact summary by split/format:"
                    )
                )
                grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
                for item in issues_payload:
                    if not isinstance(item, dict):
                        continue
                    split_name = str(item.get("split") or "-")
                    fmt = str(item.get("format") or "-")
                    reason_code = str(item.get("reason_code") or "unknown").strip() or "unknown"
                    key = (split_name, fmt, reason_code)
                    entry = grouped.setdefault(
                        key,
                        {
                            "count": 0,
                            "runs": set(),
                            "statuses": set(),
                            "reason": str(item.get("reason") or "-").replace("\n", " ").strip(),
                        },
                    )
                    entry["count"] = int(entry.get("count", 0)) + 1
                    entry["runs"].add(str(item.get("run_name") or "-"))
                    entry["statuses"].add(str(item.get("status") or "-"))
                    if entry.get("reason") in {"", "-"}:
                        entry["reason"] = str(item.get("reason") or "-").replace("\n", " ").strip()
                for (split_name, fmt, reason_code), entry in sorted(
                    grouped.items(),
                    key=lambda kv: (
                        -int(kv[1].get("count", 0)),
                        str(kv[0][0]),
                        str(kv[0][1]),
                        str(kv[0][2]),
                    ),
                ):
                    runs_sorted = sorted(str(x) for x in entry.get("runs", set()))
                    runs_txt = ", ".join(f"`{r}`" for r in runs_sorted[:4])
                    if len(runs_sorted) > 4:
                        runs_txt += f"{' и ещё ' if is_ru else ', plus '}{len(runs_sorted) - 4}"
                    status_txt = "/".join(sorted(str(x) for x in entry.get("statuses", set())))
                    reason = str(entry.get("reason") or "-")
                    if reason and reason != "-" and len(reason) > 96:
                        reason = reason[:93].rstrip() + "..."
                    lines.append(
                        f"- `{split_name}` / `{fmt}` / `{reason_code}`: "
                        + (f"{entry['count']} шт." if is_ru else f"{entry['count']} item(s)")
                        + (
                            f"; runs: {runs_txt or '-'}; status: `{status_txt or '-'}`; причина: {reason}"
                            if is_ru
                            else f"; runs: {runs_txt or '-'}; status: `{status_txt or '-'}`; reason: {reason}"
                        )
                    )
                lines.append("")
    eval_rel = str(fmt_cmp.get("eval_csv") or "")
    seen_fmt_csv_paths: set[str] = set()
    for key in ("test_csv", "val_csv", "pt_uni_csv", "csv"):
        fmt_csv_rel = str(fmt_cmp.get(key) or "").strip()
        if not fmt_csv_rel:
            continue
        norm_rel = os.path.normpath(fmt_csv_rel.replace("\\", "/"))
        if norm_rel in seen_fmt_csv_paths:
            continue
        fmt_csv_abs = os.path.join(report_root, fmt_csv_rel)
        if os.path.isfile(fmt_csv_abs):
            try:
                fmt_df = pd.read_csv(fmt_csv_abs)
                fmt_df = _filter_generic_table_for_selection(fmt_df, manifest)
                fmt_df = _select_table_columns(fmt_csv_rel, fmt_df)
                if _should_drop_split_column(fmt_csv_rel, fmt_df):
                    fmt_df = fmt_df.drop(columns=["split"], errors="ignore")
                fmt_df = _abbrev_df(fmt_df, abbreviations)
                fmt_kind = _infer_table_kind(fmt_csv_rel)
                _emit_figure_1_before_table_11()
                lines.extend(_table_preamble_lines(fmt_csv_rel, fmt_df, fmt_kind, is_ru, tpl))
                lines.extend(_center_open())
                lines.append("")
                lines.append(f"**{'Таблица' if is_ru else 'Table'} {table_no}. {_table_title(fmt_csv_rel, is_ru)}**")
                lines.append("")
                lines.extend(_md_table_from_df(fmt_df, abbreviations, limit=None, is_ru=is_ru))
                lines.append("")
                lines.append(
                    ("_Источник данных:_ " if is_ru else "_Data source:_ ")
                    + f"`{fmt_csv_rel}`"
                )
                low_rel = fmt_csv_rel.lower()
                if "format_metrics_compare_test" in low_rel or "format_metrics_compare_val" in low_rel:
                    lines.append(
                        (
                            "- Пустые quality-ячейки для `engine/trt` могут означать `invalid_metrics`: файл метрик найден, "
                            "но все ключевые метрики равны нулю и помечены как невалидные."
                            if is_ru
                            else "- Empty quality cells for `engine/trt` may indicate `invalid_metrics`: metrics file exists, "
                            "but all key metrics are zeros and treated as invalid."
                        )
                    )
                _append_takeaway_bullets(
                    lines,
                    _table_takeaway_lines(
                        fmt_csv_rel,
                        fmt_df,
                        fmt_kind,
                        is_ru,
                        manifest=manifest,
                        report_root=str(report_root),
                        tpl=tpl,
                    ),
                )
                lines.extend(_center_close())
                seen_fmt_csv_paths.add(norm_rel)
                table_no += 1
            except Exception as e:
                lines.append(f"- {('Ошибка чтения' if is_ru else 'Read error')}: {e}")
    lines.append("")
    lines.append(f"### {format_idx}.2 " + ("Сравнение производительности" if is_ru else "Performance comparison"))
    lines.append("")
    lines.extend(_subsection_intro_lines(tpl, "SUB_FORMAT_PERF"))
    lines.extend(_perf_not_collected_hint_lines(manifest, is_ru, tpl))
    lines.append(
        "Сравнение производительности форматов (test)" if is_ru else "Format performance comparison (test)"
    )
    lines.append("")
    lines.append("#### " + ("Анализ скорости" if is_ru else "Speed analysis"))
    lines.append("")
    lines.extend(_subsection_intro_lines(tpl, "SUB_FORMAT_SPEED"))
    if tpl.get("SPEED"):
        lines.extend(_justify_block(tpl["SPEED"]))
    has_combined_per_class = any(
        isinstance(rel, str) and "artifacts/pr/per_class_combined/" in rel and rel.endswith(".png")
        for rel in images
    )
    speed_vs_map_rendered = False
    for rel in images:
        if isinstance(rel, str):
            if early_figure1_emitted_in_doc and early_figure1_rel and rel == early_figure1_rel:
                continue
            if not any(k in rel for k in ("compare", "inference", "speed_quality")):
                continue
            lines.extend(_figure_preamble_lines(rel, is_ru, tpl))
            lines.extend(_center_open())
            lines.append("")
            lines.append(f"![]({os.path.join('..', rel)}){{ width=95% }}")
            lines.append(f"*{_figure_caption(rel, figure_no, abbreviations, manifest, is_ru)}*")
            figure_no += 1
            lines.append("")
            lines.extend(_center_close())
            _append_takeaway_bullets(
                lines,
                _figure_takeaway_lines(rel, is_ru, manifest=manifest, report_root=str(report_root), tpl=tpl),
            )
            if "speed_vs_map" in rel.replace("\\", "/").lower():
                speed_vs_map_rendered = True
                table_no = _append_speed_quality_table(
                    lines,
                    manifest=manifest,
                    report_root=str(report_root),
                    abbreviations=abbreviations,
                    is_ru=is_ru,
                    table_no=table_no,
                    tpl=tpl,
                    emit_before_table=_emit_figure_1_before_table_11,
                )
    if not speed_vs_map_rendered:
        table_no = _append_speed_quality_table(
            lines,
            manifest=manifest,
            report_root=str(report_root),
            abbreviations=abbreviations,
            is_ru=is_ru,
            table_no=table_no,
            tpl=tpl,
            emit_before_table=_emit_figure_1_before_table_11,
        )
    rendered_perf_table = False
    for key in ("perf_test_csv",):
        perf_csv_rel = str(fmt_cmp.get(key) or "")
        if not perf_csv_rel:
            continue
        perf_csv_abs = os.path.join(report_root, perf_csv_rel)
        if os.path.isfile(perf_csv_abs):
            try:
                perf_df = pd.read_csv(perf_csv_abs)
                perf_df = _filter_generic_table_for_selection(perf_df, manifest)
                # Hide legacy target-mismatch rows in the report table to keep
                # the performance view focused on current comparable artifacts.
                if "performance_reason" in perf_df.columns:
                    perf_df = perf_df[
                        perf_df["performance_reason"].astype(str) != "perf_target_mismatch_legacy_variant"
                    ].copy()
                perf_df = _select_table_columns(perf_csv_rel, perf_df)
                perf_df = _abbrev_df(perf_df, abbreviations)
                if _should_drop_split_column(perf_csv_rel, perf_df):
                    perf_df = perf_df.drop(columns=["split"], errors="ignore")
                def _num_or_none(v: Any) -> float | None:
                    try:
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            return None
                        out = float(v)
                        if pd.isna(out):
                            return None
                        return out
                    except Exception:
                        return None

                # Explicitly separate pure inference timing from full pipeline timing.
                pure_ms_series = (
                    perf_df.get("perf_inference_ms_per_frame")
                    if "perf_inference_ms_per_frame" in perf_df.columns
                    else None
                )
                if pure_ms_series is None and "avg_inference_ms_per_frame" in perf_df.columns:
                    pure_ms_series = perf_df.get("avg_inference_ms_per_frame")
                if pure_ms_series is None and "latency_p50_ms" in perf_df.columns:
                    pure_ms_series = perf_df.get("latency_p50_ms")
                if pure_ms_series is None:
                    pure_ms_series = pd.Series([None] * len(perf_df), index=perf_df.index)
                perf_df["pure_inference_ms_per_frame"] = pure_ms_series

                if "throughput_img_s" in perf_df.columns:
                    pure_fps_series = perf_df["throughput_img_s"]
                else:
                    pure_fps_series = perf_df["pure_inference_ms_per_frame"].apply(
                        lambda x: (1000.0 / x) if (_num_or_none(x) is not None and _num_or_none(x) > 0.0) else None
                    )
                perf_df["pure_inference_fps"] = pure_fps_series

                pipeline_ms_series = (
                    perf_df.get("perf_total_ms_per_frame")
                    if "perf_total_ms_per_frame" in perf_df.columns
                    else None
                )
                if pipeline_ms_series is None and "avg_total_ms_per_frame" in perf_df.columns:
                    pipeline_ms_series = perf_df.get("avg_total_ms_per_frame")
                if pipeline_ms_series is None:
                    pipeline_ms_series = pd.Series([None] * len(perf_df), index=perf_df.index)
                perf_df["pipeline_end_to_end_ms_per_frame"] = pipeline_ms_series
                perf_df["pipeline_end_to_end_fps"] = perf_df["pipeline_end_to_end_ms_per_frame"].apply(
                    lambda x: (1000.0 / x) if (_num_or_none(x) is not None and _num_or_none(x) > 0.0) else None
                )

                pure_cols = [
                    c
                    for c in (
                        "alias",
                        "run_name",
                        "split",
                        "format",
                        "pure_inference_ms_per_frame",
                        "pure_inference_fps",
                        "latency_p95_ms",
                        "perf_batch",
                        "perf_device",
                    )
                    if c in perf_df.columns
                ]
                pipeline_cols = [
                    c
                    for c in (
                        "alias",
                        "run_name",
                        "split",
                        "format",
                        "perf_preprocess_ms_per_frame",
                        "perf_inference_ms_per_frame",
                        "perf_postprocess_ms_per_frame",
                        "pipeline_end_to_end_ms_per_frame",
                        "pipeline_end_to_end_fps",
                        "perf_warmup_images",
                        "perf_sample_count",
                    )
                    if c in perf_df.columns
                ]
                pure_df = perf_df[pure_cols].copy() if pure_cols else perf_df.copy()
                pipeline_df = perf_df[pipeline_cols].copy() if pipeline_cols else perf_df.copy()
                _emit_figure_1_before_table_11()
                lines.extend(_table_preamble_lines(perf_csv_rel, pure_df, "perf_pure", is_ru, tpl))
                lines.extend(_center_open())
                lines.append("")
                lines.append(
                    f"**{'Таблица' if is_ru else 'Table'} {table_no}. "
                    + (
                        "Производительность форматов: чистый инференс"
                        if is_ru
                        else "Format performance: pure inference"
                    )
                    + "**"
                )
                lines.append("")
                lines.extend(_md_table_from_df(pure_df, abbreviations, limit=None, is_ru=is_ru, float_decimals=1))
                lines.append("")
                lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{perf_csv_rel}`"))
                lines.append(
                    (
                        "- Основной KPI: сопоставимые runtime-этапы по методике Ultralytics-like. "
                        "I/O загрузки источника и одноразовая инициализация бэкенда сюда не входят."
                        if is_ru
                        else "- Primary KPI: comparable runtime stages in Ultralytics-like methodology. "
                        "Source I/O and one-time backend initialization are excluded."
                    )
                )
                lines.append("")
                legend = [
                    ("Алиас", "Короткий идентификатор артефакта формата") if is_ru else ("Alias", "Short format artifact identifier"),
                    ("Запуск", "Сравниваемый run") if is_ru else ("Run", "Compared run"),
                    ("Подвыборка", "Подмножество датасета (обычно test)") if is_ru else ("Split", "Dataset subset (usually test)"),
                    ("Формат", "Тип экспортированного артефакта") if is_ru else ("Format", "Exported artifact format"),
                    (
                        "мс/кадр (чистый инференс)",
                        "Среднее время ядра инференса на кадр (без preprocess/postprocess)",
                    )
                    if is_ru
                    else ("ms/frame (pure inference)", "Average core inference time per frame (without preprocess/postprocess)"),
                    ("FPS (чистый инференс)", "Расчёт как 1000 / мс на кадр чистого инференса")
                    if is_ru
                    else ("FPS (pure inference)", "Computed as 1000 / pure inference ms per frame"),
                    ("Задержка p95, мс", "95-й перцентиль latency (steady/all fallback)")
                    if is_ru
                    else ("Latency p95, ms", "95th percentile latency (steady/all fallback)"),
                    ("Batch (eval)", "Batch из eval args") if is_ru else ("Batch (eval)", "Batch from eval args"),
                    ("Устройство", "Устройство исполнения (например cpu/0)")
                    if is_ru
                    else ("Device", "Execution device (e.g. cpu/0)"),
                ]
                lines.append("**" + ("Легенда колонок:" if is_ru else "Column legend:") + "**")
                lines.append("")
                for idx, (title, descr) in enumerate(legend, start=1):
                    lines.append(f"{idx}. **{title}** — {descr}")
                lines.append("")
                _append_takeaway_bullets(
                    lines,
                    _table_takeaway_lines(
                        perf_csv_rel,
                        perf_df,
                        "perf_pure",
                        is_ru,
                        manifest=manifest,
                        report_root=str(report_root),
                        tpl=tpl,
                    ),
                )
                table_no += 1
                lines.extend(_center_close())
                lines.append("")
                _emit_figure_1_before_table_11()
                lines.extend(_table_preamble_lines(perf_csv_rel, pipeline_df, "perf_pipeline", is_ru, tpl))
                lines.extend(_center_open())
                lines.append("")
                lines.append(
                    f"**{'Таблица' if is_ru else 'Table'} {table_no}. "
                    + (
                        "Производительность форматов: полный e2e pipeline"
                        if is_ru
                        else "Format performance: full e2e pipeline"
                    )
                    + "**"
                )
                lines.append("")
                lines.extend(_md_table_from_df(pipeline_df, abbreviations, limit=None, is_ru=is_ru, float_decimals=1))
                lines.append("")
                lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{perf_csv_rel}`"))
                lines.append(
                    (
                        "- Здесь показан полный runtime-конвейер (preprocess + inference + postprocess), "
                        "но без одноразовой инициализации."
                        if is_ru
                        else "- This table shows full runtime pipeline "
                        "(preprocess + inference + postprocess), without one-time initialization."
                    )
                )
                lines.append("")
                legend2 = [
                    ("Алиас", "Короткий идентификатор артефакта формата") if is_ru else ("Alias", "Short format artifact identifier"),
                    ("Запуск", "Сравниваемый run") if is_ru else ("Run", "Compared run"),
                    ("Подвыборка", "Подмножество датасета (обычно test)") if is_ru else ("Split", "Dataset subset (usually test)"),
                    ("Формат", "Тип экспортированного артефакта") if is_ru else ("Format", "Exported artifact format"),
                    ("мс/кадр (preprocess)", "Среднее время preprocess на кадр")
                    if is_ru
                    else ("ms/frame (preprocess)", "Average preprocess time per frame"),
                    ("мс/кадр (inference stage)", "Среднее время ядра инференса на кадр")
                    if is_ru
                    else ("ms/frame (inference stage)", "Average core inference time per frame"),
                    ("мс/кадр (postprocess)", "Среднее время decode/NMS на кадр")
                    if is_ru
                    else ("ms/frame (postprocess)", "Average decode/NMS time per frame"),
                    ("мс/кадр (pipeline e2e)", "Среднее суммарное время полного конвейера на кадр")
                    if is_ru
                    else ("ms/frame (pipeline e2e)", "Average total end-to-end pipeline time per frame"),
                    ("FPS (pipeline e2e)", "Расчёт как 1000 / мс полного конвейера")
                    if is_ru
                    else ("FPS (pipeline e2e)", "Computed as 1000 / total e2e ms per frame"),
                    ("Warmup (кадров)", "Число warmup-кадров, исключённых из steady latency")
                    if is_ru
                    else ("Warmup images", "Warmup frames excluded from steady latency"),
                    ("Измерено кадров", "Количество кадров в perf-замере")
                    if is_ru
                    else ("Measured frames", "Number of frames used in perf sampling"),
                ]
                lines.append("**" + ("Легенда колонок (e2e):" if is_ru else "Column legend (e2e):") + "**")
                lines.append("")
                for idx, (title, descr) in enumerate(legend2, start=1):
                    lines.append(f"{idx}. **{title}** — {descr}")
                _append_takeaway_bullets(
                    lines,
                    _table_takeaway_lines(
                        perf_csv_rel,
                        perf_df,
                        "perf_pipeline",
                        is_ru,
                        manifest=manifest,
                        report_root=str(report_root),
                        tpl=tpl,
                    ),
                )
                diag_cols = [
                    c
                    for c in (
                        "alias",
                        "run_name",
                        "split",
                        "format",
                        "perf_io_load_ms_per_frame",
                        "perf_diag_alloc_ms_per_frame",
                        "perf_diag_h2d_ms_per_frame",
                        "perf_diag_execute_ms_per_frame",
                        "perf_diag_d2h_ms_per_frame",
                        "perf_diag_session_init_ms",
                        "perf_diag_engine_init_ms",
                        "perf_diag_worker_wall_ms",
                        "perf_diag_retries_count",
                        "perf_diag_retry_sleep_ms",
                        "perf_diag_provider_switched_to_cpu",
                    )
                    if c in perf_df.columns
                ]
                if diag_cols:
                    diag_df = perf_df[diag_cols].copy()
                    diag_df = _drop_all_nan_columns(diag_df)
                    if len(diag_df.columns) > 0:
                        lines.append("")
                        table_no += 1
                        _emit_figure_1_before_table_11()
                        lines.append(
                            f"**{'Таблица' if is_ru else 'Table'} {table_no}. "
                            + (
                                "Диагностика несопоставимых накладных расходов"
                                if is_ru
                                else "Diagnostics of non-comparable overheads"
                            )
                            + "**"
                        )
                        lines.append("")
                        lines.extend(
                            _md_table_from_df(
                                diag_df, abbreviations, limit=None, is_ru=is_ru, float_decimals=1, diag_style=True
                            )
                        )
                        lines.append("")
                        lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{perf_csv_rel}`"))
                        lines.append(
                            (
                                "- Эти метрики диагностические и не используются в основном сравнении форматов."
                                if is_ru
                                else "- These metrics are diagnostic and are not used for primary format comparison."
                            )
                        )
                        _append_takeaway_bullets(
                            lines,
                            _table_takeaway_lines(
                                perf_csv_rel,
                                perf_df,
                                "perf_diag",
                                is_ru,
                                manifest=manifest,
                                report_root=str(report_root),
                                tpl=tpl,
                            ),
                        )
                lines.extend(_center_close())
                table_no += 1
                rendered_perf_table = True
            except Exception as e:
                lines.append(f"- {('Ошибка чтения' if is_ru else 'Read error')}: {e}")
    if not rendered_perf_table:
        fallback_perf_rel = ""
        for rel in tables:
            if isinstance(rel, str) and rel.lower().endswith("artifacts/inference/benchmark.csv"):
                fallback_perf_rel = rel
                break
        if fallback_perf_rel:
            fallback_perf_abs = os.path.join(report_root, fallback_perf_rel)
            if os.path.isfile(fallback_perf_abs):
                try:
                    perf_df = pd.read_csv(fallback_perf_abs)
                    perf_df = _filter_generic_table_for_selection(perf_df, manifest)
                    keep = [
                        c
                        for c in (
                            "model",
                            "run_dir",
                            "device",
                            "avg_total_ms_per_frame",
                            "avg_inference_ms_per_frame",
                            "avg_total_fps",
                            "avg_inference_fps",
                        )
                        if c in perf_df.columns
                    ]
                    if keep:
                        perf_df = perf_df[keep]
                    perf_df = _abbrev_df(perf_df, abbreviations)
                    _emit_figure_1_before_table_11()
                    lines.extend(_table_preamble_lines(fallback_perf_rel, perf_df, "perf_fallback", is_ru, tpl))
                    lines.extend(_center_open())
                    lines.append("")
                    lines.append(
                        f"**{'Таблица' if is_ru else 'Table'} {table_no}. "
                        + ("Fallback: скорость инференса по benchmark" if is_ru else "Fallback: inference benchmark speed")
                        + "**"
                    )
                    lines.append("")
                    lines.extend(_md_table_from_df(perf_df, abbreviations, limit=None, is_ru=is_ru))
                    lines.append("")
                    lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{fallback_perf_rel}`"))
                    lines.append("")
                    _append_takeaway_bullets(
                        lines,
                        _table_takeaway_lines(
                            fallback_perf_rel,
                            perf_df,
                            "perf_fallback",
                            is_ru,
                            manifest=manifest,
                            report_root=str(report_root),
                            tpl=tpl,
                        ),
                    )
                    lines.extend(_center_close())
                    table_no += 1
                    rendered_perf_table = True
                except Exception as e:
                    lines.append(f"- {('Ошибка чтения' if is_ru else 'Read error')}: {e}")
    if not rendered_perf_table:
        lines.append(
            "- "
            + (
                "Таблица производительности форматов отсутствует: для format compare не найден `perf_test_csv` "
                "и fallback `artifacts/inference/benchmark.csv` недоступен."
                if is_ru
                else "Format performance table is unavailable: `perf_test_csv` is missing and fallback "
                "`artifacts/inference/benchmark.csv` is not available."
            )
        )
        lines.append("")
    lines.append(_sec("per_class"))
    lines.append("")
    if tpl.get("PER_CLASS"):
        lines.extend(_justify_block(tpl["PER_CLASS"]))
    lines.extend(_subsection_intro_lines(tpl, "SUB_PER_CLASS"))
    pr_csv_rel = str(((manifest.get("pr_per_class") or {}) if isinstance(manifest.get("pr_per_class"), dict) else {}).get("csv") or "")
    if pr_csv_rel:
        pr_abs = os.path.join(report_root, pr_csv_rel)
        if os.path.isfile(pr_abs):
            try:
                pr_df = pd.read_csv(pr_abs)
                pr_df = _filter_generic_table_for_selection(pr_df, manifest)
                pr_sum = _build_pr_per_class_summary(pr_df)
                if len(pr_sum) > 0:
                    pr_sum = _abbrev_df(pr_sum, abbreviations)
                    _emit_figure_1_before_table_11()
                    lines.extend(_table_preamble_lines(pr_csv_rel, pr_sum, "pr_per_class_summary", is_ru, tpl))
                    lines.extend(_center_open())
                    lines.append("")
                    lines.append(f"**{'Таблица' if is_ru else 'Table'} {table_no}. {_table_title(pr_csv_rel, is_ru)}**")
                    lines.append("")
                    lines.extend(_md_table_from_df(pr_sum, abbreviations, limit=None, is_ru=is_ru))
                    lines.append("")
                    lines.append(
                        ("_Источник данных:_ " if is_ru else "_Data source:_ ")
                        + f"`{pr_csv_rel}`"
                    )
                    _append_takeaway_bullets(
                        lines,
                        _table_takeaway_lines(
                            pr_csv_rel,
                            pr_sum,
                            "pr_per_class_summary",
                            is_ru,
                            manifest=manifest,
                            report_root=str(report_root),
                            tpl=tpl,
                        ),
                    )
                    lines.extend(_center_close())
                    table_no += 1
            except Exception as e:
                lines.append(f"- {('Ошибка чтения' if is_ru else 'Read error')}: {e}")
    conf_map = (
        manifest.get("confidence_recommendations")
        if isinstance(manifest.get("confidence_recommendations"), dict)
        else {}
    )
    for objective in ("A", "B", "C"):
        rel = str(conf_map.get(objective) or "")
        if not rel:
            continue
        abs_path = os.path.join(report_root, rel) if report_root else ""
        if not abs_path or not os.path.isfile(abs_path):
            continue
        try:
            cdf = pd.read_csv(abs_path)
            cdf = _filter_generic_table_for_selection(cdf, manifest)
            if "level" in cdf.columns:
                cdf = cdf[cdf["level"].astype(str) == "class"].copy()
            if len(cdf) == 0:
                continue
            group_col = "run_name" if "run_name" in cdf.columns else None
            grouped_items: list[tuple[str, pd.DataFrame]]
            if group_col:
                grouped_items = []
                for run_name, gdf in cdf.groupby(group_col, dropna=False):
                    grouped_items.append((str(run_name), gdf.copy()))
            else:
                grouped_items = [("-", cdf.copy())]

            for run_name, run_df in grouped_items:
                local_df = run_df.copy()
                for col in ("run_name", "objective", "level"):
                    if col in local_df.columns:
                        local_df = local_df.drop(columns=[col])
                preferred_pc = [
                    "class_name",
                    "split",
                    "class_id",
                    "recommended_conf",
                    "target_metric",
                    "precision",
                    "recall",
                    "f1",
                    "status",
                ]
                chosen_pc = [c for c in preferred_pc if c in local_df.columns]
                if chosen_pc:
                    local_df = local_df[chosen_pc]
                local_df = _abbrev_df(local_df, abbreviations)
                conf_kind = _infer_table_kind(rel)
                _emit_figure_1_before_table_11()
                lines.extend(_table_preamble_lines(rel, local_df, conf_kind, is_ru, tpl))
                lines.extend(_center_open())
                lines.append("")
                objective_title = _table_title(rel, is_ru)
                if is_ru:
                    full_title = f"{objective_title} — run {run_name}"
                else:
                    full_title = f"{objective_title} — run {run_name}"
                lines.append(f"**{'Таблица' if is_ru else 'Table'} {table_no}. {full_title}**")
                lines.append("")
                lines.extend(_md_table_from_df(local_df, abbreviations, limit=None, is_ru=is_ru))
                lines.append("")
                lines.append(
                    ("_Источник данных:_ " if is_ru else "_Data source:_ ")
                    + f"`{rel}`"
                )
                lines.append("")
                _append_takeaway_bullets(
                    lines,
                    _table_takeaway_lines(
                        rel,
                        local_df,
                        conf_kind,
                        is_ru,
                        manifest=manifest,
                        report_root=str(report_root),
                        tpl=tpl,
                    ),
                )
                lines.extend(_center_close())
                table_no += 1
        except Exception as e:
            lines.append(f"- {('Ошибка чтения' if is_ru else 'Read error')}: {e}")
    for rel in images:
        if isinstance(rel, str) and ("artifacts/pr/" in rel and rel.endswith("pr_all_classes.png")):
            lines.extend(_figure_preamble_lines(rel, is_ru, tpl))
            lines.extend(_center_open())
            lines.append("")
            lines.append(f"![]({os.path.join('..', rel)}){{ width=95% }}")
            lines.append(f"*{_figure_caption(rel, figure_no, abbreviations, manifest, is_ru)}*")
            figure_no += 1
            lines.append("")
            lines.extend(_center_close())
            _append_takeaway_bullets(
                lines,
                _figure_takeaway_lines(rel, is_ru, manifest=manifest, report_root=str(report_root), tpl=tpl),
            )
        # Prefer unified per-class set when present; otherwise keep legacy
        # per-class rendering for backward compatibility.
        if isinstance(rel, str) and (
            ("artifacts/pr/per_class_combined/" in rel and rel.endswith(".png"))
            or (
                (not has_combined_per_class)
                and ("artifacts/pr/" in rel and "per_class/" in rel and rel.endswith(".png"))
            )
        ):
            lines.extend(_figure_preamble_lines(rel, is_ru, tpl))
            lines.extend(_center_open())
            lines.append("")
            lines.append(f"![]({os.path.join('..', rel)}){{ width=95% }}")
            lines.append(f"*{_figure_caption(rel, figure_no, abbreviations, manifest, is_ru)}*")
            figure_no += 1
            lines.append("")
            lines.extend(_center_close())
            _append_takeaway_bullets(
                lines,
                _figure_takeaway_lines(rel, is_ru, manifest=manifest, report_root=str(report_root), tpl=tpl),
            )
    lines.append(_sec("ultra"))
    lines.append("")
    ultra_rows = manifest.get("ultralytics_test") or []
    if isinstance(ultra_rows, list) and ultra_rows:
        table_rows: list[dict[str, Any]] = []
        for item in ultra_rows:
            if not isinstance(item, dict):
                continue
            table_rows.append(
                {
                    "run": item.get("run_code") or os.path.basename(str(item.get("run_name") or "")),
                    "pr.csv": "yes" if (item.get("csv") or {}).get("pr.csv") else "no",
                    "pr_per_class.csv": "yes" if (item.get("csv") or {}).get("pr_per_class.csv") else "no",
                    "images_count": len(item.get("images") or []),
                }
            )
        if table_rows:
            # Keep Ultralytics section compact: no generic summary table.
            pass
        for run_pos, item in enumerate(ultra_rows, start=1):
            if not isinstance(item, dict):
                continue
            run_code = str(item.get("run_code") or item.get("run_name") or "")
            ultra_sub_idx = section_index["ultra"]
            lines.append(f"### {ultra_sub_idx}.{run_pos} {('Запуск' if is_ru else 'Run')} {run_code}")
            lines.append("")
            lines.extend(_subsection_intro_lines(tpl, "SUB_ULTRA_RUN"))
            comp_lines = _ultralytics_completeness_lines(item, is_ru)
            if comp_lines:
                lines.append("")
                lines.extend(_subsection_intro_lines(tpl, "SUB_ULTRA_COMPLETENESS"))
                lines.extend(comp_lines)
            run_info = item.get("run_info") or {}
            if isinstance(run_info, dict):
                model = str(run_info.get("model") or "").strip()
                dataset_name = str(run_info.get("dataset_name") or "").strip()
                epochs = run_info.get("epochs")
                batch = run_info.get("batch_size")
                train_img = run_info.get("train_image_size")
                val_img = run_info.get("val_imgsz")
                if any([model, dataset_name, epochs is not None, batch is not None, train_img is not None, val_img is not None]):
                    if is_ru:
                        lines.append(
                            "- Параметры запуска: "
                            + f"модель={model or '-'}, датасет={_abbrev_value(dataset_name, abbreviations) if dataset_name else '-'}, "
                            + f"epochs={epochs if epochs is not None else '-'}, batch={batch if batch is not None else '-'}, "
                            + f"imgsz_train={train_img if train_img is not None else '-'}, imgsz_val={val_img if val_img is not None else '-'}."
                        )
                    else:
                        lines.append(
                            "- Run config: "
                            + f"model={model or '-'}, dataset={_abbrev_value(dataset_name, abbreviations) if dataset_name else '-'}, "
                            + f"epochs={epochs if epochs is not None else '-'}, batch={batch if batch is not None else '-'}, "
                            + f"imgsz_train={train_img if train_img is not None else '-'}, imgsz_val={val_img if val_img is not None else '-'}."
                        )
            machine_info = item.get("machine_info") or {}
            if isinstance(machine_info, dict):
                cpu = str(machine_info.get("sys_cpu_model") or "").strip()
                cores = machine_info.get("sys_cpu_logical_cores")
                ram = machine_info.get("sys_ram_total_gb")
                gpu = str(machine_info.get("sys_gpu_0_name") or "").strip()
                vram = machine_info.get("sys_gpu_0_vram_gb")
                os_name = str(machine_info.get("sys_os") or "").strip()
                os_rel = str(machine_info.get("sys_os_release") or "").strip()
                if any([cpu, cores is not None, ram is not None, gpu, vram is not None, os_name, os_rel]):
                    if is_ru:
                        lines.append(
                            "- Машина: "
                            + f"CPU={cpu or '-'} ({cores if cores is not None else '-'} cores), "
                            + f"RAM={ram if ram is not None else '-'} GB, "
                            + f"GPU={gpu or '-'} ({vram if vram is not None else '-'} GB), "
                            + f"OS={os_name or '-'} {os_rel}".strip()
                        )
                    else:
                        lines.append(
                            "- Machine: "
                            + f"CPU={cpu or '-'} ({cores if cores is not None else '-'} cores), "
                            + f"RAM={ram if ram is not None else '-'} GB, "
                            + f"GPU={gpu or '-'} ({vram if vram is not None else '-'} GB), "
                            + f"OS={os_name or '-'} {os_rel}".strip()
                        )
            csv_map = item.get("csv") or {}
            if isinstance(csv_map, dict):
                for key in ("pr.csv", "pr_per_class.csv"):
                    rel = str(csv_map.get(key) or "")
                    if rel:
                        lines.append(
                            "- "
                            + _csv_source_label(key, is_ru)
                            + f": `{rel}`"
                        )
                pc_rel = str(csv_map.get("pr_per_class.csv") or "")
                if pc_rel and str(report_root):
                    lines.append("")
                    lines.extend(_subsection_intro_lines(tpl, "SUB_ULTRA_PER_CLASS_TABLE"))
                    pc_lines, table_no = _ultralytics_per_class_ap_table_lines(
                        report_root=str(report_root),
                        csv_rel=pc_rel,
                        is_ru=is_ru,
                        table_no=table_no,
                    )
                    lines.extend(pc_lines)
            if lines and not lines[-1] == "":
                lines.append("")
            for rel in item.get("images") or []:
                rel = str(rel)
                if not rel:
                    continue
                lines.extend(_figure_preamble_lines(rel, is_ru, tpl))
                lines.extend(_center_open())
                lines.append("")
                lines.append(f"![]({os.path.join('..', rel)}){{ width=95% }}")
                lines.append(f"*{_figure_caption(rel, figure_no, abbreviations, manifest, is_ru)}*")
                figure_no += 1
                lines.append("")
                lines.extend(_center_close())
                _append_takeaway_bullets(
                    lines,
                    _figure_takeaway_lines(rel, is_ru, manifest=manifest, report_root=str(report_root), tpl=tpl),
                )
    else:
        lines.append("- " + ("Артефакты Ultralytics test не обнаружены." if is_ru else "No Ultralytics test artifacts found."))
        lines.append("")
    lines.append(_sec("conclusion"))
    lines.append("")
    if leaderboard_rel:
        lb_abs = os.path.join(report_root, leaderboard_rel)
        if os.path.isfile(lb_abs):
            try:
                lb_df = pd.read_csv(lb_abs)
                lb_df = _filter_generic_table_for_selection(lb_df, manifest)
                lb_keep = [c for c in ("model", "run_name", "run_dir", "composite_score", "quality_metric", "speed_metric") if c in lb_df.columns]
                if lb_keep:
                    lb_df = lb_df[lb_keep]
                lb_df = _abbrev_df(lb_df, abbreviations)
                _emit_figure_1_before_table_11()
                lines.extend(_table_preamble_lines(leaderboard_rel, lb_df, "leaderboard", is_ru, tpl))
                lines.extend(_center_open())
                lines.append("")
                lines.append(f"**{'Таблица' if is_ru else 'Table'} {table_no}. {_table_title(leaderboard_rel, is_ru)}**")
                lines.append("")
                lines.extend(_md_table_from_df(lb_df, abbreviations, limit=None, is_ru=is_ru))
                lines.append("")
                lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{leaderboard_rel}`"))
                lines.append("")
                _append_takeaway_bullets(
                    lines,
                    _table_takeaway_lines(
                        leaderboard_rel,
                        lb_df,
                        "leaderboard",
                        is_ru,
                        manifest=manifest,
                        report_root=str(report_root),
                        tpl=tpl,
                    ),
                )
                lines.extend(_center_close())
                table_no += 1
            except Exception:
                pass
    if tpl.get("CONCLUSION"):
        lines.extend(_justify_block(tpl["CONCLUSION"]))
    else:
        lines.append("- " + ("Рекомендуется использовать выводы выше для выбора trade-off качества/скорости." if is_ru else "Use the findings above to select the quality/speed trade-off."))
    missing_lines = _missing_reasons_from_manifest(manifest, lang)
    if missing_lines:
        lines.append("")
        lines.append("### " + ("Пропуски и причины" if is_ru else "Missing values and reasons"))
        lines.append("")
        lines.extend(_subsection_intro_lines(tpl, "SUB_CONCLUSION_MISSING"))
        lines.extend(missing_lines)
    lines.append("")
    lines.append(_sec("exec"))
    lines.append("")
    if tpl.get("EXECUTIVE_SUMMARY"):
        lines.extend(_justify_block(tpl["EXECUTIVE_SUMMARY"]))
    lines.extend(_insights_from_manifest(manifest, lang))
    lines.append("")
    lines.append("")
    return lines

