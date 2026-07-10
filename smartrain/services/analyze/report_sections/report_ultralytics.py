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

from smartrain.services.analyze.report_sections.report_tables import _csv_source_label

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
    except Exception as exc:
        logger.warning("Failed to append speed/quality table %s: %s", csv_rel, exc)
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
