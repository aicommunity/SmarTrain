"""Report figure captions and narrative helpers."""

from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd

from smartrain.services.analyze.report_markdown_formatting import (
    MAX_NARRATIVE_BULLETS,
    _abbrev_value,
    _column_display_name,
    _justify_block,
    _pr_summary_takeaways,
    _speed_quality_takeaways,
)
from smartrain.core.runtime.logging_config import get_logger

logger = get_logger(__name__)

from smartrain.services.analyze.report_sections.report_common import _append_takeaway_bullets

_ULTRALYTICS_DIAG_BASENAMES = frozenset(
    {
        "boxpr_curve.png",
        "pr_curve.png",
        "boxf1_curve.png",
        "f1_curve.png",
        "boxp_curve.png",
        "p_curve.png",
        "boxr_curve.png",
        "r_curve.png",
    }
)

_ULTRALYTICS_COMPACT_BASENAMES = frozenset(
    {
        "confusion_matrix_normalized.png",
        "val_batch0_pred.jpg",
    }
)


def ultralytics_report_mode(manifest: dict[str, Any]) -> str:
    explicit = str(manifest.get("ultralytics_report_mode") or "").strip().lower()
    if explicit in {"compact", "full"}:
        return explicit
    single = bool(manifest.get("single_run_mode")) or not (manifest.get("others") or [])
    return "full" if single else "compact"


def is_ultralytics_compact_main_image(rel: str) -> bool:
    return os.path.basename(rel.replace("\\", "/")).lower() in _ULTRALYTICS_COMPACT_BASENAMES


def is_ultralytics_diagnostic_image(rel: str) -> bool:
    return os.path.basename(rel.replace("\\", "/")).lower() in _ULTRALYTICS_DIAG_BASENAMES


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
    if "ultralytics-test/" in low:
        return "NARR_FIG_ULTRA"
    return "NARR_FIG_DEFAULT"


def _figure_preamble_lines(rel: str, is_ru: bool, tpl: dict[str, str]) -> list[str]:
    if "ultralytics-test/" in rel.replace("\\", "/").lower():
        return []
    key = _figure_narrative_key(rel)
    t = str(tpl.get(key) or tpl.get("NARR_FIG_DEFAULT") or "").strip()
    if not t:
        return []
    return _justify_block(t)


def _ultra_item_for_rel(manifest: dict[str, Any], rel: str) -> dict[str, Any] | None:
    norm = rel.replace("\\", "/").lower()
    m = re.search(r"ultralytics-test/([^/]+)/", norm)
    if not m:
        return None
    run_code = m.group(1)
    rows = manifest.get("ultralytics_test") or []
    if not isinstance(rows, list):
        return None
    for item in rows:
        if not isinstance(item, dict):
            continue
        code = str(item.get("run_code") or item.get("run_name") or "")
        if code == run_code or run_code.startswith(code) or code.startswith(run_code):
            return item
    return None


def _figure_context_suffix(
    rel: str,
    manifest: dict[str, Any],
    abbreviations: dict[str, str],
    is_ru: bool,
) -> str:
    norm = rel.replace("\\", "/")
    low = norm.lower()
    parts: list[str] = []

    if "compare_curves" in low:
        baseline = os.path.basename(str(manifest.get("baseline", "")).rstrip("/"))
        others = [os.path.basename(str(x).rstrip("/")) for x in (manifest.get("others") or [])]
        b = abbreviations.get(baseline, baseline)
        o = ",".join(abbreviations.get(x, x) for x in others)
        if o:
            parts.append(f"{'базовый' if is_ru else 'baseline'}: {b}; {'сравнение' if is_ru else 'others'}: {o}")
        return f" ({'; '.join(parts)})" if parts else ""

    if "benchmark" in low and low.endswith(".png"):
        runs = []
        for row in manifest.get("run_legend") or []:
            if isinstance(row, dict):
                runs.append(str(row.get("short_label") or ""))
        if runs:
            parts.append(f"PT CPU benchmark; runs: {', '.join(r for r in runs if r)}")
        else:
            parts.append("PT CPU benchmark")
        return f" ({'; '.join(parts)})" if parts else ""

    if "speed_vs_map" in low:
        sq = manifest.get("speed_quality") if isinstance(manifest.get("speed_quality"), dict) else {}
        x_name = _column_display_name(str(sq.get("scatter_x") or "avg_inference_ms_per_frame"), is_ru)
        y_name = _column_display_name(str(sq.get("scatter_y") or "mAP50-95"), is_ru)
        parts.append(f"{x_name} vs {y_name}")
        report_root = str(manifest.get("_report_root") or "")
        sq_rel = str((sq or {}).get("csv") or "")
        if report_root and sq_rel:
            p = os.path.join(report_root, sq_rel)
            if os.path.isfile(p):
                try:
                    pdf = pd.read_csv(p)
                    if "model" in pdf.columns:
                        labels = [
                            abbreviations.get(str(v), str(v))
                            for v in pdf["model"].astype(str).tolist()
                        ]
                        if labels:
                            parts.append(", ".join(labels))
                except Exception as exc:
                    logger.debug("Figure context skipped for %s: %s", rel, exc)
        return f" ({'; '.join(parts)})" if parts else ""

    if "pr_all_classes" in low or ("pr/" in low and "per_class" in low):
        runs = []
        for row in manifest.get("run_legend") or []:
            if isinstance(row, dict):
                runs.append(str(row.get("short_label") or ""))
        if runs:
            parts.append(f"{'все запуски' if is_ru else 'all runs'}: {', '.join(r for r in runs if r)}")
        return f" ({'; '.join(parts)})" if parts else ""

    if "pr_class_" in low:
        m = re.search(r"pr_class_\d+_(.+)\.png$", os.path.basename(rel), flags=re.IGNORECASE)
        if m:
            cls = m.group(1).replace("_", " ")
            parts.append(f"{'класс' if is_ru else 'class'}: {cls}")
        runs = []
        for row in manifest.get("run_legend") or []:
            if isinstance(row, dict):
                runs.append(str(row.get("short_label") or ""))
        if runs:
            parts.append(f"{'все запуски' if is_ru else 'all runs'}: {', '.join(r for r in runs if r)}")
        return f" ({'; '.join(parts)})" if parts else ""

    if "ultralytics-test/" in low:
        m = re.search(r"ultralytics-test/([^/]+)/", norm, flags=re.IGNORECASE)
        run_code = m.group(1) if m else ""
        run_label = _abbrev_value(run_code, abbreviations) if run_code else ""
        if run_label:
            parts.append(f"{'запуск' if is_ru else 'run'}: {run_label}")
        item = _ultra_item_for_rel(manifest, rel)
        if item:
            run_info = item.get("run_info") if isinstance(item.get("run_info"), dict) else {}
            model = str(run_info.get("model") or "").strip()
            dataset = str(run_info.get("dataset_name") or "").strip()
            if model:
                parts.append(f"{'модель' if is_ru else 'model'}: {model}")
            if dataset:
                parts.append(f"{'датасет' if is_ru else 'dataset'}: {_abbrev_value(dataset, abbreviations)}")
            epochs = run_info.get("epochs")
            batch = run_info.get("batch_size")
            val_img = run_info.get("val_imgsz") or run_info.get("train_image_size")
            cfg_bits: list[str] = []
            if epochs is not None:
                cfg_bits.append(f"epochs={epochs}")
            if batch is not None:
                cfg_bits.append(f"batch={batch}")
            if val_img is not None:
                cfg_bits.append(f"imgsz={val_img}")
            if cfg_bits:
                parts.append(", ".join(cfg_bits))
            comp = item.get("completeness") if isinstance(item.get("completeness"), dict) else {}
            split_hint = str(comp.get("primary_split") or comp.get("split") or "").strip()
            if not split_hint:
                sources = str(comp.get("sources_summary") or item.get("sources_summary") or "")
                if "test-split" in sources.lower():
                    split_hint = "test"
                elif "train-val" in sources.lower() or "train-ultralytics" in sources.lower():
                    split_hint = "val"
            if split_hint:
                parts.append(f"split: {split_hint}")
        return f" ({'; '.join(parts)})" if parts else ""

    return ""


def _figure_takeaway_lines(
    rel: str,
    is_ru: bool,
    *,
    manifest: dict[str, Any],
    report_root: str,
    tpl: dict[str, str],
    abbreviations: dict[str, str] | None = None,
) -> list[str]:
    abbreviations = abbreviations or {}
    if isinstance(manifest.get("abbreviations"), dict):
        abbreviations = {**manifest["abbreviations"], **abbreviations}
    lines: list[str] = []
    low = rel.replace("\\", "/").lower()
    if "ultralytics-test/" in low and is_ultralytics_diagnostic_image(rel):
        return lines
    rr = str(report_root or "")
    if "speed_vs_map" in low and rr:
        sq = manifest.get("speed_quality") if isinstance(manifest.get("speed_quality"), dict) else {}
        csv_rel = str((sq or {}).get("csv") or "").strip()
        if csv_rel:
            p = os.path.join(rr, csv_rel)
            if os.path.isfile(p):
                try:
                    df = pd.read_csv(p)
                    lines.extend(
                        _speed_quality_takeaways(
                            df,
                            is_ru,
                            abbreviations,
                            scatter_x_metric=str(sq.get("scatter_x") or "avg_inference_ms_per_frame"),
                            scatter_y_metric=str(sq.get("scatter_y") or "mAP50-95"),
                        )
                    )
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
                                lab = _abbrev_value(pdf.loc[i].get("run_name", pdf.loc[i].get("model", i)), abbreviations)
                                if is_ru:
                                    lines.append(
                                        f"- Максимум **{_column_display_name(col, is_ru)}**: **{lab}** ({float(s.max()):.4g})."
                                    )
                                else:
                                    lines.append(
                                        f"- Max **{_column_display_name(col, is_ru)}**: **{lab}** ({float(s.max()):.4g})."
                                    )
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
                    lines.extend(_pr_summary_takeaways(pdf, is_ru, abbreviations))
                except Exception as exc:
                    logger.debug("Figure takeaway skipped for %s: %s", rel, exc)
    if not lines:
        return []
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
    suffix = _figure_context_suffix(rel, manifest, abbreviations, is_ru)
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
