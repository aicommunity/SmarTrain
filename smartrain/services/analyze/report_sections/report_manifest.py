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
            except Exception as exc:
                logger.warning("Failed to render report section: %s", exc)
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
            except Exception as exc:
                logger.warning("Failed to render report section: %s", exc)
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
