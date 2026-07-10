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

from smartrain.services.analyze.report_sections.report_common import _append_takeaway_bullets
from smartrain.services.analyze.report_sections.report_tables import _table_title

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
                except Exception as exc:
                    logger.debug("Figure takeaway skipped for %s: %s", rel, exc)
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
                except Exception as exc:
                    logger.debug("Figure takeaway skipped for %s: %s", rel, exc)
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
                except Exception as exc:
                    logger.debug("Figure takeaway skipped for %s: %s", rel, exc)
    if not lines:
        t = str(tpl.get("NARR_TAKEAWAY_NO_DATA") or "").strip()
        if t:
            lines.append(f"- {t}")
    return lines[:MAX_NARRATIVE_BULLETS]

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
