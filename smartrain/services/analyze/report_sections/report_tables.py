"""Report markdown section builders."""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from smartrain.services.analyze.report_markdown_formatting import (
    MAX_NARRATIVE_BULLETS,
    _abbrev_df,
    _build_test_metrics_summary,
    _center_close,
    _center_open,
    _column_display_name,
    _filter_generic_table_for_selection,
    _filter_runs_summary_for_selection,
    _justify_block,
    _md_table_from_df,
    _os_display_train_profile_row,
    _pr_summary_takeaways,
    _read_template,
    _row_label_from_df,
    _select_table_columns,
    _should_hide_system_profile_table,
    _speed_quality_takeaways,
    _subsection_intro_lines,
    _table_preamble_lines,
    _table_takeaway_lines,
)
from smartrain.core.runtime.logging_config import get_logger

logger = get_logger(__name__)

from smartrain.services.analyze.report_sections.report_common import append_table_footer, emit_centered_table_block

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
    if "compare_delta" in low:
        return "compare_delta"
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
    except Exception as exc:
        logger.warning("Failed to load filtered table %s: %s", rel, exc)
        return None

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
    except Exception as exc:
        logger.warning("Failed to render table section: %s", exc)
        return table_no
    if emit_before_table is not None:
        emit_before_table()
    emit_centered_table_block(
        lines,
        table_no=table_no,
        title=_table_title(rel, is_ru),
        preamble_lines=_table_preamble_lines(rel, df, "speed_quality", is_ru, tpl, table_no=table_no),
        table_body_lines=_md_table_from_df(df, abbreviations, limit=None, is_ru=is_ru),
        source_rel=rel,
        takeaways=_table_takeaway_lines(
            rel,
            df,
            "speed_quality",
            is_ru,
            manifest=manifest,
            report_root=str(report_root),
            tpl=tpl,
        ),
        is_ru=is_ru,
    )
    return table_no + 1
