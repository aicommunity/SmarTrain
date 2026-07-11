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
        except Exception as exc:
            logger.debug("Failed to resolve report path %s: %s", p, exc)
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


def _abbrev_label(value: Any, abbreviations: dict[str, str]) -> str:
    s = str(value or "").strip()
    if not s:
        return s
    if s in abbreviations:
        return abbreviations[s]
    base = os.path.basename(s.rstrip("/"))
    return abbreviations.get(base, abbreviations.get(s, s))


def _executive_insights_from_manifest(
    manifest: dict[str, Any],
    lang: str,
    abbreviations: dict[str, str],
) -> list[str]:
    lines: list[str] = []
    report_root = str(manifest.get("_report_root") or "")
    sq = manifest.get("speed_quality") if isinstance(manifest.get("speed_quality"), dict) else {}
    sq_meta = sq or {}
    x_metric = str(sq_meta.get("scatter_x") or "scatter_x_value")
    y_metric = str(sq_meta.get("scatter_y") or "scatter_y_value")
    sq_csv_rel = str(sq_meta.get("csv") or "")
    if report_root and sq_csv_rel:
        sq_csv = os.path.join(report_root, sq_csv_rel)
        if os.path.isfile(sq_csv):
            try:
                df = pd.read_csv(sq_csv)
                x_col, y_col = "scatter_x_value", "scatter_y_value"
                if {x_col, y_col, "model"}.issubset(df.columns):
                    best = df.sort_values(y_col, ascending=False).iloc[0]
                    fastest = df.sort_values(x_col, ascending=True).iloc[0]
                    best_label = _abbrev_label(best["model"], abbreviations)
                    fast_label = _abbrev_label(fastest["model"], abbreviations)
                    x_name = _column_display_name(x_metric, lang == "ru")
                    y_name = _column_display_name(y_metric, lang == "ru")
                    if lang == "ru":
                        lines.append(
                            f"- Лучший компромисс качество/скорость: **{best_label}** ({y_name}={float(best[y_col]):.4f})."
                        )
                        lines.append(
                            f"- Самый быстрый запуск: **{fast_label}** ({x_name}={float(fastest[x_col]):.2f})."
                        )
                    else:
                        lines.append(
                            f"- Best quality/speed trade-off: **{best_label}** ({y_name}={float(best[y_col]):.4f})."
                        )
                        lines.append(
                            f"- Fastest run: **{fast_label}** ({x_name}={float(fastest[x_col]):.2f})."
                        )
            except Exception as exc:
                logger.warning("Failed to render executive insight: %s", exc)
    pr = manifest.get("pr_per_class") if isinstance(manifest.get("pr_per_class"), dict) else {}
    pr_csv_rel = str((pr or {}).get("csv") or "")
    if report_root and pr_csv_rel:
        pr_csv = os.path.join(report_root, pr_csv_rel)
        if os.path.isfile(pr_csv):
            try:
                pdf = pd.read_csv(pr_csv)
                if {"model", "class_name", "ap"}.issubset(pdf.columns):
                    pdf = pdf.copy()
                    pdf["model"] = pdf["model"].astype(str).map(lambda x: _abbrev_label(x, abbreviations))
                    grp = pdf.groupby(["model", "class_name"], as_index=False)["ap"].mean()
                    if len(grp["model"].unique()) >= 2:
                        best_model = (
                            grp.groupby("model", as_index=False)["ap"]
                            .mean()
                            .sort_values("ap", ascending=False)
                            .iloc[0]["model"]
                        )
                        pivot = grp.pivot(index="class_name", columns="model", values="ap")
                        diff = pivot.sub(pivot[best_model], axis=0).drop(columns=[best_model], errors="ignore")
                        if len(diff.columns) > 0:
                            worst_class = diff.min(axis=1).idxmin()
                            if lang == "ru":
                                lines.append(
                                    f"- Наибольшая деградация по классу относительно **{best_model}**: **{worst_class}**."
                                )
                            else:
                                lines.append(
                                    f"- Largest per-class degradation vs **{best_model}**: **{worst_class}**."
                                )
            except Exception as exc:
                logger.warning("Failed to render executive insight: %s", exc)
    return lines[:MAX_NARRATIVE_BULLETS]


def _technical_insights_from_manifest(manifest: dict[str, Any], lang: str) -> list[str]:
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
            except Exception as exc:
                logger.warning("Failed to render report section: %s", exc)
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
    return lines


def _insights_from_manifest(
    manifest: dict[str, Any],
    lang: str,
    abbreviations: dict[str, str] | None = None,
) -> list[str]:
    abbreviations = abbreviations or {}
    if isinstance(manifest.get("abbreviations"), dict):
        abbreviations = {**manifest["abbreviations"], **abbreviations}
    return _executive_insights_from_manifest(manifest, lang, abbreviations)


def _render_run_legend_table_lines(
    manifest: dict[str, Any],
    *,
    is_ru: bool,
    workspace_root: str,
    table_no: int,
) -> tuple[list[str], int]:
    rows = manifest.get("run_legend") or []
    if not isinstance(rows, list) or not rows:
        return [], table_no
    lines: list[str] = []
    lines.extend(_center_open())
    lines.append("")
    title = "Легенда запусков" if is_ru else "Run legend"
    lines.append(f"**{'Таблица' if is_ru else 'Table'} {table_no}. {title}**")
    lines.append("")
    if is_ru:
        header = ["M", "Архитектура", "Датасет", "Эпохи", "Batch", "Путь run"]
    else:
        header = ["M", "Architecture", "Dataset", "Epochs", "Batch", "Run path"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        if not isinstance(row, dict):
            continue
        short_label = str(row.get("short_label") or f"M{row.get('index', '?')}")
        architecture = str(row.get("architecture") or "-")
        dataset_label = str(row.get("dataset_label") or row.get("dataset_name") or "-")
        epochs = str(row.get("epochs") or "-")
        batch = str(row.get("batch") or "-")
        run_name = str(row.get("run_name") or "")
        run_path = _path_for_report(str(row.get("run_dir") or run_name), workspace_root)
        if run_path == run_name or not run_path:
            run_display = f"`{run_name}`"
        else:
            run_display = f"`{run_path}`"
        role = str(row.get("role") or "")
        m_cell = f"**{short_label}**" + (f" ({'базовый' if is_ru else 'baseline'})" if role == "baseline" else "")
        lines.append(
            "| "
            + " | ".join([m_cell, architecture, dataset_label, epochs, batch, run_display])
            + " |"
        )
    lines.append("")
    lines.extend(_center_close())
    return lines, table_no + 1


def _load_artifact_csv(report_root: str, rel: str) -> pd.DataFrame | None:
    if not report_root or not rel:
        return None
    path = os.path.join(report_root, rel)
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to read artifact csv %s: %s", rel, exc)
        return None


def _find_table_rel(manifest: dict[str, Any], needle: str) -> str:
    for rel in manifest.get("tables") or []:
        if isinstance(rel, str) and needle in rel.lower():
            return rel
    return ""


def _render_executive_summary_section(
    manifest: dict[str, Any],
    *,
    is_ru: bool,
    tpl: dict[str, str],
    abbreviations: dict[str, str],
    table_no: int,
) -> tuple[list[str], int]:
    lines: list[str] = []
    report_root = str(manifest.get("_report_root") or "")
    lang = "ru" if is_ru else "en"
    if tpl.get("EXECUTIVE_SUMMARY"):
        lines.extend(_justify_block(tpl["EXECUTIVE_SUMMARY"]))
    leader_rel = _find_table_rel(manifest, "leaderboard")
    lb_df = _load_artifact_csv(report_root, leader_rel)
    if lb_df is not None and len(lb_df) > 0:
        lb_df = _filter_generic_table_for_selection(lb_df, manifest)
        keep = [c for c in ("model", "run_name", "composite_score", "quality_metric", "speed_metric") if c in lb_df.columns]
        if "model" in lb_df.columns and "run_name" in lb_df.columns:
            same = lb_df["model"].astype(str).equals(lb_df["run_name"].astype(str))
            if same:
                keep = [c for c in keep if c != "model"]
        if keep:
            lb_df = _abbrev_df(lb_df[keep], abbreviations)
        lines.extend(_table_preamble_lines(leader_rel, lb_df, "leaderboard", is_ru, tpl))
        lines.extend(_center_open())
        lines.append("")
        lines.append(
            f"**{'Таблица' if is_ru else 'Table'} {table_no}. "
            + ("Рейтинг моделей (сводка)" if is_ru else "Model leaderboard (summary)")
            + "**"
        )
        lines.append("")
        lines.extend(_md_table_from_df(lb_df, abbreviations, limit=5, is_ru=is_ru))
        lines.append("")
        if leader_rel:
            lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{leader_rel}`"))
        lines.append("")
        from smartrain.services.analyze.report_sections.report_common import _append_takeaway_bullets

        _append_takeaway_bullets(
            lines,
            _table_takeaway_lines(
                leader_rel,
                lb_df,
                "leaderboard",
                is_ru,
                manifest=manifest,
                report_root=report_root,
                tpl=tpl,
            ),
        )
        lines.extend(_center_close())
        table_no += 1
    delta_rel = _find_table_rel(manifest, "compare_delta")
    delta_df = _load_artifact_csv(report_root, delta_rel)
    if delta_df is not None and len(delta_df) > 0:
        delta_df = _filter_generic_table_for_selection(delta_df, manifest)
        delta_df = _select_table_columns(delta_rel, delta_df)
        delta_df = _abbrev_df(delta_df, abbreviations)
        lines.extend(_table_preamble_lines(delta_rel, delta_df, "compare_delta", is_ru, tpl))
        lines.extend(_center_open())
        lines.append("")
        lines.append(
            f"**{'Таблица' if is_ru else 'Table'} {table_no}. "
            + ("Ключевые дельты относительно baseline" if is_ru else "Key deltas vs baseline")
            + "**"
        )
        lines.append("")
        lines.extend(_md_table_from_df(delta_df, abbreviations, limit=5, is_ru=is_ru))
        lines.append("")
        if delta_rel:
            lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{delta_rel}`"))
        lines.append("")
        from smartrain.services.analyze.report_sections.report_common import _append_takeaway_bullets

        _append_takeaway_bullets(
            lines,
            _table_takeaway_lines(
                delta_rel,
                delta_df,
                "compare_delta",
                is_ru,
                manifest=manifest,
                report_root=report_root,
                tpl=tpl,
                abbreviations=abbreviations,
            ),
        )
        lines.extend(_center_close())
        table_no += 1
    sq_meta = manifest.get("speed_quality") if isinstance(manifest.get("speed_quality"), dict) else {}
    sq_rel = _find_table_rel(manifest, "speed_quality")
    if not sq_rel:
        sq_rel = str(sq_meta.get("csv") or "")
    sq_df = _load_artifact_csv(report_root, sq_rel)
    if sq_df is not None and len(sq_df) > 0:
        sq_df = _filter_generic_table_for_selection(sq_df, manifest)
        keep = [
            c
            for c in (
                "model",
                "scatter_x_metric",
                "scatter_x_value",
                "scatter_y_metric",
                "scatter_y_value",
                "quality_source",
            )
            if c in sq_df.columns
        ]
        if keep:
            sq_df = _abbrev_df(sq_df[keep], abbreviations)
        lines.extend(_table_preamble_lines(sq_rel, sq_df, "speed_quality", is_ru, tpl))
        lines.extend(_center_open())
        lines.append("")
        lines.append(
            f"**{'Таблица' if is_ru else 'Table'} {table_no}. "
            + ("Компромисс скорость–качество (сводка)" if is_ru else "Speed–quality trade-off (summary)")
            + "**"
        )
        lines.append("")
        lines.extend(_md_table_from_df(sq_df, abbreviations, limit=5, is_ru=is_ru))
        lines.append("")
        if sq_rel:
            lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{sq_rel}`"))
        lines.append("")
        from smartrain.services.analyze.report_sections.report_common import _append_takeaway_bullets

        _append_takeaway_bullets(
            lines,
            _table_takeaway_lines(
                sq_rel,
                sq_df,
                "speed_quality",
                is_ru,
                manifest=manifest,
                report_root=report_root,
                tpl=tpl,
                abbreviations=abbreviations,
            ),
        )
        lines.extend(_center_close())
        table_no += 1
    exec_insights = _executive_insights_from_manifest(manifest, lang, abbreviations)
    if exec_insights:
        lines.append("### " + ("Ключевые выводы" if is_ru else "Key findings"))
        lines.append("")
        lines.extend(exec_insights)
        lines.append("")
    if tpl.get("CONCLUSION"):
        lines.extend(_justify_block(tpl["CONCLUSION"]))
    detail_ref = (
        "Подробные таблицы и графики — в разделах «Анализ качества», «Сравнение форматов» и «Анализ по классам»."
        if is_ru
        else "Detailed tables and figures are in Quality, Format comparison, and Per-class sections."
    )
    lines.extend(_justify_block(detail_ref))
    lines.append("")
    return lines, table_no


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
            except Exception as exc:
                logger.warning("Failed to render report section: %s", exc)
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
        except Exception as exc:
            logger.debug("Skipping confidence recommendation row: %s", exc)
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
