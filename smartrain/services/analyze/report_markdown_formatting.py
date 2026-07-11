"""Markdown formatting helpers for analyze session reports."""

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


__all__ = [
    "_read_template",
    "_column_display_name",
    "_pretty_value",
    "_diag_cell_placeholder",
    "_pretty_diag_table_value",
    "_drop_all_nan_columns",
    "_should_drop_split_column",
    "_os_display_train_profile_row",
    "_md_table_from_df",
    "_abbrev_value",
    "_abbrev_df",
    "_select_table_columns",
    "_build_pr_per_class_summary",
    "_should_hide_system_profile_table",
    "_system_profile_text_summary",
    "_pick_test_metric_columns",
    "_build_test_metrics_summary",
    "_selected_run_identity",
    "_filter_runs_summary_for_selection",
    "_filter_generic_table_for_selection",
    "_quality_metric_comments",
    "_justify_block",
    "_center_open",
    "_center_close",
    "_subsection_intro_lines",
    "_preamble_template_key",
    "_table_preamble_lines",
    "_row_label_from_df",
    "_numeric_spread_takeaways",
    "_format_metrics_takeaways",
    "_perf_status_takeaways",
    "_speed_quality_takeaways",
    "_leaderboard_takeaways",
    "_pr_summary_takeaways",
    "_system_sparse_takeaways_fixed",
    "_table_takeaway_lines",
    "MAX_NARRATIVE_BULLETS",
    "MAX_SPREAD_METRICS",
    "_ID_COLS_FOR_ROW_LABEL",
    "_NUMERIC_TAKEAWAY_SKIP",
]


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


def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df.columns) < 2:
        return df
    out = df.copy()
    drop: list[str] = []
    cols = list(out.columns)
    for i, left in enumerate(cols):
        if left in drop:
            continue
        left_series = out[left].astype(str)
        for right in cols[i + 1 :]:
            if right in drop:
                continue
            if left_series.equals(out[right].astype(str)):
                drop.append(right)
    if drop:
        out = out.drop(columns=[c for c in drop if c in out.columns])
    return out


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
    return _drop_duplicate_columns(out)


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
                f"{metric_label}: лучший запуск **{best_run}** ({best_val:.4f}), худший **{worst_run}** ({worst_val:.4f})."
            )
        else:
            lines.append(
                f"{metric_label}: best run **{best_run}** ({best_val:.4f}), worst **{worst_run}** ({worst_val:.4f})."
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

_ID_COLS_FOR_ROW_LABEL = ("run_name", "alias", "model", "run", "run_dir", "baseline", "other")
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
    if kind in ("leaderboard", "compare_delta", "speed_quality", "format_metrics", "runs_summary"):
        return f"NARR_PREAMBLE_{kind.upper()}"
    return "NARR_PREAMBLE_GENERIC"


def _table_preamble_lines(
    rel: str,
    df: pd.DataFrame | None,
    kind: str,
    is_ru: bool,
    tpl: dict[str, str],
    *,
    table_no: int | None = None,
) -> list[str]:
    key = _preamble_template_key(kind)
    static = str(tpl.get(key) or "").strip()
    if not static and key != "NARR_PREAMBLE_GENERIC":
        static = str(tpl.get("NARR_PREAMBLE_GENERIC") or "").strip()
    if kind not in ("alias_legend", "eval_settings") and key == "NARR_PREAMBLE_GENERIC":
        # Skip generic boilerplate for routine tables; title + takeaways are enough.
        return []
    if not static:
        return []
    if table_no is not None:
        static = static.replace("{n}", str(table_no))
    static = re.sub(
        r"^\*\*(?:Таблица|Table)\s*(?:\{n\}|\d+)\*\*\s*[—–\-]\s*",
        "",
        static,
    )
    static = re.sub(
        r"^\*\*(?:Таблица|Table)\s*(?:\{n\}|\d+)\*\*\s+",
        "",
        static,
    )
    return [static, ""]


def _row_label_from_df(df: pd.DataFrame, idx: Any) -> str:
    for c in _ID_COLS_FOR_ROW_LABEL:
        if c in df.columns:
            v = df.loc[idx, c]
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                s = str(v).strip()
                if s:
                    return s
    return str(idx)


def _compare_delta_takeaways(df: pd.DataFrame, is_ru: bool, abbreviations: dict[str, str] | None = None) -> list[str]:
    abbreviations = abbreviations or {}
    lines: list[str] = []
    if df is None or len(df) == 0:
        return lines
    metric_cols = [c for c in df.columns if c not in {"baseline", "other", "run_name", "run_dir", "model"}]
    for col in metric_cols[:3]:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        idx = s.idxmax()
        row = df.loc[idx]
        other = _abbrev_value(row.get("other", row.get("run_name", "?")), abbreviations)
        delta = float(s.loc[idx])
        lab = _column_display_name(col, is_ru)
        if is_ru:
            lines.append(f"Наибольший выигрыш по **{lab}**: **{other}** ({delta:+.4f}).")
        else:
            lines.append(f"Largest gain on **{lab}**: **{other}** ({delta:+.4f}).")
    return lines[:MAX_NARRATIVE_BULLETS]


def _numeric_spread_takeaways(df: pd.DataFrame, is_ru: bool, abbreviations: dict[str, str] | None = None) -> list[str]:
    abbreviations = abbreviations or {}
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
        rmax = _abbrev_value(_row_label_from_df(df, imax), abbreviations)
        rmin = _abbrev_value(_row_label_from_df(df, imin), abbreviations)
        lab = label_fn(col)
        if is_ru:
            lines.append(f"{lab}: максимум у **{rmax}** ({vmax:.4g}), минимум у **{rmin}** ({vmin:.4g}).")
        else:
            lines.append(f"{lab}: max **{rmax}** ({vmax:.4g}), min **{rmin}** ({vmin:.4g}).")
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
            lines.append(f"Лучший **{row_lab}** по {_column_display_name(col, is_ru)}: **{float(s.loc[idx]):.4f}**.")
        else:
            lines.append(f"Best **{row_lab}** on {_column_display_name(col, is_ru)}: **{float(s.loc[idx]):.4f}**.")
        break
    if not lines and is_ru:
        lines.append("Качество по выбранным колонкам недоступно для сравнения (все значения пропущены).")
    elif not lines:
        lines.append("Quality comparison unavailable (all values missing).")
    return lines[:MAX_NARRATIVE_BULLETS]


def _perf_status_takeaways(df: pd.DataFrame, is_ru: bool) -> list[str]:
    lines: list[str] = []
    if df is None or "performance_status" not in df.columns:
        return lines
    st = df["performance_status"].astype(str).str.strip().str.lower()
    bad = int((st != "ok").sum())
    if bad > 0:
        if is_ru:
            lines.append(f"Строк со статусом производительности не `ok`: **{bad}** из {len(df)}.")
        else:
            lines.append(f"Rows with performance_status not `ok`: **{bad}** of {len(df)}.")
    return lines


def _speed_quality_takeaways(
    df: pd.DataFrame,
    is_ru: bool,
    abbreviations: dict[str, str] | None = None,
    *,
    scatter_x_metric: str = "avg_inference_ms_per_frame",
    scatter_y_metric: str = "mAP50-95",
) -> list[str]:
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
            y_name = _column_display_name(scatter_y_metric, is_ru)
            if is_ru:
                lines.append(f"Лучший компромисс по качеству (**{y_name}**): **{model_label}** ({float(sy.max()):.4f}).")
            else:
                lines.append(f"Best quality (**{y_name}**): **{model_label}** ({float(sy.max()):.4f}).")
    if mcol and x in df.columns:
        sx = pd.to_numeric(df[x], errors="coerce")
        if sx.notna().sum() > 0:
            fast = df.loc[sx.idxmin()]
            model_label = abbreviations.get(str(fast[mcol]), str(fast[mcol]))
            x_name = _column_display_name(scatter_x_metric, is_ru)
            if is_ru:
                lines.append(f"Наиболее быстрый по **{x_name}**: **{model_label}** ({float(sx.min()):.4f}).")
            else:
                lines.append(f"Fastest on **{x_name}**: **{model_label}** ({float(sx.min()):.4f}).")
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
                lines.append(f"Лидер по composite_score: **{m}** ({v:.4f}).")
            else:
                lines.append(f"Leader by composite_score: **{m}** ({v:.4f}).")
    lines.extend(_numeric_spread_takeaways(df, is_ru))
    return lines[:MAX_NARRATIVE_BULLETS]


def _pr_summary_takeaways(df: pd.DataFrame, is_ru: bool, abbreviations: dict[str, str] | None = None) -> list[str]:
    abbreviations = abbreviations or {}
    lines: list[str] = []
    if df is None or len(df) == 0:
        return lines
    if {"class_name", "model", "ap"}.issubset(df.columns):
        g = df.groupby(["class_name", "model"], as_index=False)["ap"].mean()
        if len(g) > 0:
            worst = g.sort_values("ap", ascending=True).iloc[0]
            cn = str(worst["class_name"])
            md = _abbrev_value(worst["model"], abbreviations)
            apv = float(worst["ap"])
            if is_ru:
                lines.append(f"Наименьший средний AP: класс **{cn}**, модель **{md}** ({apv:.4f}).")
            else:
                lines.append(f"Lowest mean AP: class **{cn}**, model **{md}** ({apv:.4f}).")
    lines.extend(_numeric_spread_takeaways(df, is_ru, abbreviations))
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
        return [f"Доля заполненных sys-полей в таблице профиля: **{ratio:.0%}** ({filled}/{total})."]
    return [f"Share of populated sys_* fields: **{ratio:.0%}** ({filled}/{total})."]


def _table_takeaway_lines(
    rel: str,
    df: pd.DataFrame | None,
    kind: str,
    is_ru: bool,
    *,
    manifest: dict[str, Any],
    report_root: str,
    tpl: dict[str, str],
    abbreviations: dict[str, str] | None = None,
) -> list[str]:
    lines: list[str] = []
    abbreviations = abbreviations or {}
    if isinstance(manifest.get("abbreviations"), dict):
        abbreviations = {**manifest["abbreviations"], **abbreviations}
    no_data = str(tpl.get("NARR_TAKEAWAY_NO_DATA") or "").strip()
    if df is None or len(df) == 0:
        if no_data:
            lines.append(no_data)
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind == "compare_delta":
        lines.extend(_compare_delta_takeaways(df, is_ru, abbreviations))
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind == "runs_summary_extra":
        lines.extend(_quality_metric_comments(df, is_ru))
        if len(lines) >= 2:
            return lines[:MAX_NARRATIVE_BULLETS]
        lines.extend(_numeric_spread_takeaways(df, is_ru, abbreviations))
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind in ("format_metrics", "format_metrics_pt_uni"):
        lines.extend(_format_metrics_takeaways(df, is_ru))
        lines.extend(_numeric_spread_takeaways(df, is_ru, abbreviations))
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind.startswith("perf_"):
        lines.extend(_perf_status_takeaways(df, is_ru))
        lines.extend(_numeric_spread_takeaways(df, is_ru, abbreviations))
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind == "speed_quality":
        sq = manifest.get("speed_quality") if isinstance(manifest.get("speed_quality"), dict) else {}
        lines.extend(
            _speed_quality_takeaways(
                df,
                is_ru,
                abbreviations,
                scatter_x_metric=str(sq.get("scatter_x") or "avg_inference_ms_per_frame"),
                scatter_y_metric=str(sq.get("scatter_y") or "mAP50-95"),
            )
        )
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind == "leaderboard":
        lines.extend(_leaderboard_takeaways(df, is_ru))
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind == "pr_per_class_summary":
        lines.extend(_pr_summary_takeaways(df, is_ru, abbreviations))
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind in ("system_profile_train_sparse", "system_profile_train_cards", "test_system_profile", "system_profile_train"):
        lines.extend(_system_sparse_takeaways_fixed(df, is_ru))
        return lines[:MAX_NARRATIVE_BULLETS]

    if kind == "alias_legend":
        if "alias" in df.columns:
            nu = int(df["alias"].astype(str).nunique(dropna=False))
            if is_ru:
                lines.append(f"Число уникальных алиасов: **{nu}**.")
            else:
                lines.append(f"Unique aliases: **{nu}**.")
        return lines[:MAX_NARRATIVE_BULLETS]

    lines.extend(_numeric_spread_takeaways(df, is_ru, abbreviations))
    if not lines and no_data:
        lines.append(no_data)
    return lines[:MAX_NARRATIVE_BULLETS]

