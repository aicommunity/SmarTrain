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


def _normalize_takeaway_line(line: str) -> str:
    s = str(line or "").strip()
    if s.startswith("- "):
        return s[2:].strip()
    return s


def _append_takeaway_bullets(lines: list[str], bullets: list[str]) -> None:
    if not bullets:
        return
    for b in bullets:
        lines.append(b)
    lines.append("")


def append_table_source(
    lines: list[str],
    *,
    source_rel: str | None,
    is_ru: bool,
) -> None:
    if source_rel:
        lines.append("")
        lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{source_rel}`"))


def append_table_takeaways(lines: list[str], takeaways: list[str]) -> None:
    if not takeaways:
        return
    for item in takeaways:
        norm = _normalize_takeaway_line(item)
        if norm:
            lines.extend(_justify_block(norm))


def finalize_centered_table(lines: list[str], *, takeaways: list[str]) -> None:
    lines.extend(_center_close())
    append_table_takeaways(lines, takeaways)


def append_table_footer(
    lines: list[str],
    *,
    source_rel: str | None,
    takeaways: list[str],
    is_ru: bool,
) -> None:
    """Append source inside centered block; call finalize_centered_table() to close and emit takeaways."""
    append_table_source(lines, source_rel=source_rel, is_ru=is_ru)
def append_numbered_table_title(lines: list[str], table_no: int, title: str, is_ru: bool) -> None:
    label = "Таблица" if is_ru else "Table"
    lines.append(f"**{label} {table_no}. {title}**")
    lines.append("")


def emit_centered_table_block(
    lines: list[str],
    *,
    table_no: int,
    title: str,
    preamble_lines: list[str],
    table_body_lines: list[str],
    source_rel: str | None,
    takeaways: list[str],
    is_ru: bool,
) -> None:
    lines.extend(_center_open())
    lines.append("")
    append_numbered_table_title(lines, table_no, title, is_ru)
    if preamble_lines:
        lines.extend(preamble_lines)
    lines.extend(table_body_lines)
    append_table_source(lines, source_rel=source_rel, is_ru=is_ru)
    finalize_centered_table(lines, takeaways=takeaways)
