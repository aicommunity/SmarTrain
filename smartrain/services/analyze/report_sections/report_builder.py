"""Report markdown section builders."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from smartrain.services.analyze.report_markdown_formatting import (
    MAX_NARRATIVE_BULLETS,
    _abbrev_df,
    _abbrev_value,
    _build_pr_per_class_summary,
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
    _should_drop_split_column,
    _should_hide_system_profile_table,
    _speed_quality_takeaways,
    _subsection_intro_lines,
    _table_preamble_lines,
    _table_takeaway_lines,
)
from smartrain.core.runtime.logging_config import get_logger

logger = get_logger(__name__)


from smartrain.services.analyze.report_sections.report_common import _append_takeaway_bullets
from smartrain.services.analyze.report_sections.report_figures import (
    _discover_missing_pr_images,
    _figure_caption,
    _figure_preamble_lines,
    _figure_takeaway_lines,
    is_ultralytics_compact_main_image,
    ultralytics_report_mode,
)
from smartrain.services.analyze.report_sections.report_manifest import (
    _build_run_model_abbreviations,
    _missing_reasons_from_manifest,
    _path_for_report,
    _perf_not_collected_hint_lines,
    _render_executive_summary_section,
    _render_run_legend_table_lines,
    _technical_insights_from_manifest,
)
from smartrain.services.analyze.report_sections.report_tables import (
    _append_speed_quality_table,
    _infer_table_kind,
    _load_filtered_table_df,
    _table_title,
)
from smartrain.services.analyze.report_sections.report_ultralytics import (
    _ultralytics_completeness_lines,
    _ultralytics_per_class_ap_table_lines,
    _csv_source_label,
)

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
    section_order = ["exec", "context", "quality", "format_compare", "per_class", "ultra", "appendix", "conclusion"]
    section_index = {k: i + 1 for i, k in enumerate(section_order)}
    section_titles["appendix"] = "Приложение: полные иллюстрации Ultralytics" if is_ru else "Appendix: full Ultralytics illustrations"
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
    table_no = 1
    figure_no = 1
    appendix_images: list[tuple[str, dict[str, Any] | None]] = []
    early_figure1_inserted = False
    early_figure1_emitted_in_doc = False

    lines.append(_sec("exec"))
    lines.append("")
    exec_lines, table_no = _render_executive_summary_section(
        manifest,
        is_ru=is_ru,
        tpl=tpl,
        abbreviations=abbreviations,
        table_no=table_no,
    )
    lines.extend(exec_lines)
    legend_lines, table_no = _render_run_legend_table_lines(
        manifest,
        is_ru=is_ru,
        workspace_root=workspace_root,
        table_no=table_no,
    )
    if legend_lines:
        lines.append("### " + ("Справочник запусков" if is_ru else "Run reference"))
        lines.append("")
        lines.extend(legend_lines)
    lines.append("")
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
            _figure_takeaway_lines(
                rel_e,
                is_ru,
                manifest=manifest,
                report_root=str(report_root),
                tpl=tpl,
                abbreviations=abbreviations,
            ),
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
            except Exception as exc:
                logger.warning("Failed to render report section: %s", exc)
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
            except Exception as exc:
                logger.warning("Failed to render report section: %s", exc)
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
                takeaway_df = df
                if "runs_summary" in rel.lower():
                    test_summary = _build_test_metrics_summary(df, abbreviations)
                    if len(test_summary) > 0:
                        takeaway_df = test_summary
                        tk_kind = "runs_summary_extra"
                _append_takeaway_bullets(
                    lines,
                    _table_takeaway_lines(
                        rel,
                        takeaway_df,
                        tk_kind,
                        is_ru,
                        manifest=manifest,
                        report_root=str(report_root),
                        tpl=tpl,
                        abbreviations=abbreviations,
                    ),
                )
            except Exception as e:
                lines.append(f"- {('Ошибка чтения' if is_ru else 'Read error')}: {e}")
        else:
            lines.append(f"- {('Файл не найден' if is_ru else 'File not found')}")
        lines.extend(_center_close())
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
            except Exception as exc:
                logger.warning("Failed to read issues payload: %s", exc)
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
                    except Exception as exc:
                        logger.debug("Failed to resolve leaderboard row: %s", exc)
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
            ultra_mode = ultralytics_report_mode(manifest)
            for rel in item.get("images") or []:
                rel = str(rel)
                if not rel:
                    continue
                if ultra_mode == "compact" and not is_ultralytics_compact_main_image(rel):
                    appendix_images.append((rel, item))
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
                    _figure_takeaway_lines(
                        rel,
                        is_ru,
                        manifest=manifest,
                        report_root=str(report_root),
                        tpl=tpl,
                        abbreviations=abbreviations,
                    ),
                )
            if ultra_mode == "compact":
                deferred = [r for r, _ in appendix_images if r.startswith("artifacts/ultralytics-test/")]
                if deferred:
                    lines.append(
                        "- "
                        + (
                            "Дополнительные кривые и диаграммы — в приложении ниже."
                            if is_ru
                            else "Additional curves and plots are in the appendix below."
                        )
                    )
                    lines.append("")
    else:
        lines.append("- " + ("Артефакты Ultralytics test не обнаружены." if is_ru else "No Ultralytics test artifacts found."))
        lines.append("")
    eval_rows = manifest.get("eval_dataset_tests") or []
    lines.append("## " + ("Тесты на внешних датасетах" if is_ru else "Cross-dataset test runs"))
    lines.append("")
    if isinstance(eval_rows, list) and eval_rows:
        for row in eval_rows:
            if not isinstance(row, dict):
                continue
            run_code = str(row.get("run_code") or row.get("run_name") or "-")
            slot_key = str(row.get("slot_key") or "-")
            status = str(row.get("status") or "-")
            session_csv = str(row.get("session_metrics_csv") or "")
            dataset_yaml = str(row.get("dataset_yaml") or "")
            if is_ru:
                lines.append(f"- Запуск `{run_code}`; slot `{slot_key}`; статус: `{status}`; data: `{dataset_yaml or '-'}`")
                if session_csv:
                    lines.append(f"  - Метрики: `{session_csv}`")
            else:
                lines.append(f"- Run `{run_code}`; slot `{slot_key}`; status: `{status}`; data: `{dataset_yaml or '-'}`")
                if session_csv:
                    lines.append(f"  - Metrics: `{session_csv}`")
        lines.append("")
    else:
        lines.append("- " + ("Данных по cross-dataset тестам нет." if is_ru else "No cross-dataset test data found."))
        lines.append("")
    if appendix_images:
        lines.append(_sec("appendix"))
        lines.append("")
        lines.extend(
            _subsection_intro_lines(
                tpl,
                "SUB_ULTRA_APPENDIX",
            )
        )
        if not tpl.get("SUB_ULTRA_APPENDIX"):
            lines.extend(
                _justify_block(
                    "Полный набор Ultralytics test-иллюстраций для каждого запуска."
                    if is_ru
                    else "Full Ultralytics test illustration set for each run."
                )
            )
        for rel, item in appendix_images:
            run_code = ""
            if isinstance(item, dict):
                run_code = str(item.get("run_code") or item.get("run_name") or "")
            if run_code:
                lines.append(f"#### {run_code}")
                lines.append("")
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
                _figure_takeaway_lines(
                    rel,
                    is_ru,
                    manifest=manifest,
                    report_root=str(report_root),
                    tpl=tpl,
                    abbreviations=abbreviations,
                ),
            )
    lines.append(_sec("conclusion"))
    lines.append("")
    if leaderboard_rel:
        lines.append(
            "- "
            + (
                f"Итоговый рейтинг моделей см. в разделе «{section_titles['exec']}» (таблица leaderboard)."
                if is_ru
                else f"See model leaderboard in «{section_titles['exec']}»."
            )
        )
        lines.append("")
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
    tech_lines = _technical_insights_from_manifest(manifest, lang)
    if tech_lines:
        lines.append("")
        lines.append("### " + ("Служебная информация" if is_ru else "Technical metadata"))
        lines.append("")
        lines.extend(tech_lines)
    lines.append("")
    lines.append("")
    return lines
