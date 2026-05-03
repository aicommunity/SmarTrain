from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

from smartrain.dataset_report import _export_odt_builtin_zip, _export_odt_odfpy, _export_pdf_fpdf2, _try_pandoc_odt, _try_pandoc_pdf


def _read_template(lang: str) -> dict[str, str]:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "templates", "analyze_report"))
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
        "test_mAP50-95": "test mAP50-95" if is_ru else "test mAP50-95",
        "test_mAP50": "test mAP50" if is_ru else "test mAP50",
        "test_Box-F1": "test Box-F1" if is_ru else "test Box-F1",
        "test_Box-P": "test Box-P" if is_ru else "test Box-P",
        "test_Box-R": "test Box-R" if is_ru else "test Box-R",
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


def _md_table_from_df(
    df: pd.DataFrame,
    abbreviations: dict[str, str],
    limit: int | None = None,
    *,
    is_ru: bool = True,
    float_decimals: int = 4,
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


def _pandoc_executable() -> str | None:
    env = (os.environ.get("PANDOC") or "").strip()
    if env and os.path.isfile(env):
        return env
    for cmd in ("pandoc",):
        p = shutil.which(cmd)
        if p:
            return p
    return None


def _try_pandoc_odt_analyze(report_root: str, lang: str) -> bool:
    exe = _pandoc_executable()
    if not exe:
        return _try_pandoc_odt(report_root, lang)
    rel_md = f"{lang}/index.md"
    rel_odt = f"report-{lang}.odt"
    resource_path = f".:{lang}:artifacts"
    try:
        r = subprocess.run(
            [
                exe,
                "-f",
                "markdown+pipe_tables+grid_tables+table_captions+fenced_divs+attributes+raw_attribute+bracketed_spans",
                rel_md,
                "-o",
                rel_odt,
                "--resource-path",
                resource_path,
            ],
            cwd=report_root,
            capture_output=True,
            text=True,
            timeout=600,
        )
        return r.returncode == 0
    except Exception:
        return False


def _try_pdf_from_odt(report_root: str, lang: str) -> bool:
    odt_path = os.path.join(report_root, f"report-{lang}.odt")
    pdf_path = os.path.join(report_root, f"report-{lang}.pdf")
    if not os.path.isfile(odt_path):
        return False
    # 1) LibreOffice path.
    for soffice in ("soffice", "libreoffice"):
        try:
            r = subprocess.run(
                [soffice, "--headless", "--convert-to", "pdf", "--outdir", report_root, odt_path],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if r.returncode == 0 and os.path.isfile(pdf_path):
                return True
        except Exception:
            pass
    # 2) Pandoc ODT->PDF fallback.
    exe = _pandoc_executable()
    if exe:
        try:
            r = subprocess.run(
                [exe, odt_path, "-o", pdf_path],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if r.returncode == 0 and os.path.isfile(pdf_path):
                return True
        except Exception:
            pass
    return False


def _postprocess_odt_layout(odt_path: str) -> bool:
    if not os.path.isfile(odt_path):
        return False
    ns = {
        "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
        "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
        "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
        "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
        "fo": "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0",
        "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    }
    for pfx, uri in ns.items():
        ET.register_namespace(pfx, uri)
    changed = False
    try:
        with zipfile.ZipFile(odt_path, "r") as zin:
            blobs = {n: zin.read(n) for n in zin.namelist()}
        if "styles.xml" in blobs:
            st_root = ET.fromstring(blobs["styles.xml"])
            auto_styles = st_root.find("office:automatic-styles", ns)
            if auto_styles is None:
                auto_styles = ET.SubElement(st_root, f"{{{ns['office']}}}automatic-styles")
            st_name = "SmarTrainBody"
            body_style = None
            for s in auto_styles.findall("style:style", ns):
                if s.attrib.get(f"{{{ns['style']}}}name") == st_name:
                    body_style = s
                    break
            if body_style is None:
                body_style = ET.SubElement(
                    auto_styles,
                    f"{{{ns['style']}}}style",
                    {
                        f"{{{ns['style']}}}name": st_name,
                        f"{{{ns['style']}}}family": "paragraph",
                    },
                )
            pp = body_style.find("style:paragraph-properties", ns)
            if pp is None:
                pp = ET.SubElement(body_style, f"{{{ns['style']}}}paragraph-properties")
            pp.set(f"{{{ns['fo']}}}text-align", "justify")
            pp.set(f"{{{ns['fo']}}}text-indent", "1cm")
            # Make all paragraph styles justified/indented except headings.
            for s in st_root.findall(".//style:style", ns):
                if s.attrib.get(f"{{{ns['style']}}}family") != "paragraph":
                    continue
                parent_name = str(s.attrib.get(f"{{{ns['style']}}}parent-style-name") or "")
                style_name = str(s.attrib.get(f"{{{ns['style']}}}name") or "")
                if parent_name.startswith("Heading") or style_name.startswith("Heading"):
                    continue
                spp = s.find("style:paragraph-properties", ns)
                if spp is None:
                    spp = ET.SubElement(s, f"{{{ns['style']}}}paragraph-properties")
                if style_name != "SmarTrainCenter":
                    spp.set(f"{{{ns['fo']}}}text-align", "justify")
                    spp.set(f"{{{ns['fo']}}}text-indent", "1cm")
            blobs["styles.xml"] = ET.tostring(st_root, encoding="utf-8", xml_declaration=True)
            changed = True
        if "content.xml" in blobs:
            ct_root = ET.fromstring(blobs["content.xml"])
            auto_styles = ct_root.find("office:automatic-styles", ns)
            if auto_styles is None:
                auto_styles = ET.SubElement(ct_root, f"{{{ns['office']}}}automatic-styles")
            center_style_name = "SmarTrainCenter"
            body_style_name = "SmarTrainBody"
            has_center = any(
                s.attrib.get(f"{{{ns['style']}}}name") == center_style_name for s in auto_styles.findall("style:style", ns)
            )
            if not has_center:
                cs = ET.SubElement(
                    auto_styles,
                    f"{{{ns['style']}}}style",
                    {
                        f"{{{ns['style']}}}name": center_style_name,
                        f"{{{ns['style']}}}family": "paragraph",
                    },
                )
                cpp = ET.SubElement(cs, f"{{{ns['style']}}}paragraph-properties")
                cpp.set(f"{{{ns['fo']}}}text-align", "center")
            table_text_style_name = "SmarTrainTableCell"
            has_table_text = any(
                s.attrib.get(f"{{{ns['style']}}}name") == table_text_style_name for s in auto_styles.findall("style:style", ns)
            )
            if not has_table_text:
                ts = ET.SubElement(
                    auto_styles,
                    f"{{{ns['style']}}}style",
                    {
                        f"{{{ns['style']}}}name": table_text_style_name,
                        f"{{{ns['style']}}}family": "paragraph",
                    },
                )
                tpp = ET.SubElement(ts, f"{{{ns['style']}}}paragraph-properties")
                tpp.set(f"{{{ns['fo']}}}text-align", "start")
                tpp.set(f"{{{ns['fo']}}}text-indent", "0cm")
            table_header_text_style = "SmarTrainTableHeaderText"
            if not any(
                s.attrib.get(f"{{{ns['style']}}}name") == table_header_text_style
                for s in auto_styles.findall("style:style", ns)
            ):
                ths = ET.SubElement(
                    auto_styles,
                    f"{{{ns['style']}}}style",
                    {
                        f"{{{ns['style']}}}name": table_header_text_style,
                        f"{{{ns['style']}}}family": "paragraph",
                    },
                )
                thpp = ET.SubElement(ths, f"{{{ns['style']}}}paragraph-properties")
                thpp.set(f"{{{ns['fo']}}}text-align", "center")
                thpp.set(f"{{{ns['fo']}}}text-indent", "0cm")
                thtp = ET.SubElement(ths, f"{{{ns['style']}}}text-properties")
                thtp.set(f"{{{ns['fo']}}}font-weight", "bold")
                thtp.set(f"{{{ns['style']}}}font-weight-asian", "bold")
                thtp.set(f"{{{ns['style']}}}font-weight-complex", "bold")
            for cell_style, border, background in (
                ("SmarTrainTableBodyCell", "0.5pt solid #666666", None),
                ("SmarTrainTableHeaderCell", "0.75pt solid #444444", "#f2f2f2"),
            ):
                if not any(
                    s.attrib.get(f"{{{ns['style']}}}name") == cell_style
                    for s in auto_styles.findall("style:style", ns)
                ):
                    cs = ET.SubElement(
                        auto_styles,
                        f"{{{ns['style']}}}style",
                        {
                            f"{{{ns['style']}}}name": cell_style,
                            f"{{{ns['style']}}}family": "table-cell",
                        },
                    )
                    csp = ET.SubElement(cs, f"{{{ns['style']}}}table-cell-properties")
                    csp.set(f"{{{ns['fo']}}}border", border)
                    if background:
                        csp.set(f"{{{ns['fo']}}}background-color", background)
            caption_style_name = "SmarTrainCaption"
            if not any(
                s.attrib.get(f"{{{ns['style']}}}name") == caption_style_name
                for s in auto_styles.findall("style:style", ns)
            ):
                cap = ET.SubElement(
                    auto_styles,
                    f"{{{ns['style']}}}style",
                    {
                        f"{{{ns['style']}}}name": caption_style_name,
                        f"{{{ns['style']}}}family": "paragraph",
                    },
                )
                capp = ET.SubElement(cap, f"{{{ns['style']}}}paragraph-properties")
                capp.set(f"{{{ns['fo']}}}text-align", "center")
                capp.set(f"{{{ns['fo']}}}margin-bottom", "0.15cm")
            body = ct_root.find("office:body", ns)
            text_root = body.find("office:text", ns) if body is not None else None
            if text_root is not None:
                table_paragraphs = set(text_root.findall(".//table:table-cell//text:p", ns))
                for p in text_root.iterfind(".//text:p", ns):
                    raw = "".join(p.itertext()).strip()
                    if p in table_paragraphs:
                        p.set(f"{{{ns['text']}}}style-name", table_text_style_name)
                        changed = True
                        continue
                    if raw.startswith(("Рисунок ", "Figure ", "Таблица ", "Table ")):
                        p.set(f"{{{ns['text']}}}style-name", caption_style_name)
                        changed = True
                        continue
                    if p.find("draw:frame", ns) is not None:
                        p.set(f"{{{ns['text']}}}style-name", center_style_name)
                        changed = True
                        continue
                    if raw and not raw.startswith("- "):
                        p.set(f"{{{ns['text']}}}style-name", body_style_name)
                        changed = True
                for tbl in text_root.iterfind(".//table:table", ns):
                    tbl.set(f"{{{ns['table']}}}align", "center")
                    header_rows_parent = tbl.find("table:table-header-rows", ns)
                    header_rows = (
                        list(header_rows_parent.findall("table:table-row", ns)) if header_rows_parent is not None else []
                    )
                    body_rows = list(tbl.findall("table:table-row", ns))
                    # Some tables may miss table-header-rows. In that case, treat first body row as header.
                    fallback_header = not header_rows and bool(body_rows)
                    for ridx, row in enumerate(header_rows):
                        for cell in row.findall("table:table-cell", ns):
                            cell.set(f"{{{ns['table']}}}style-name", "SmarTrainTableHeaderCell")
                            p = cell.find("text:p", ns)
                            if p is not None:
                                p.set(f"{{{ns['text']}}}style-name", table_header_text_style)
                    for ridx, row in enumerate(body_rows):
                        is_header = fallback_header and ridx == 0
                        for cell in row.findall("table:table-cell", ns):
                            cell.set(
                                f"{{{ns['table']}}}style-name",
                                "SmarTrainTableHeaderCell" if is_header else "SmarTrainTableBodyCell",
                            )
                            p = cell.find("text:p", ns)
                            if p is not None:
                                p.set(
                                    f"{{{ns['text']}}}style-name",
                                    table_header_text_style if is_header else table_text_style_name,
                                )
                    cols = list(tbl.findall("table:table-column", ns))
                    if cols:
                        header_row = header_rows[0] if header_rows else (body_rows[0] if body_rows else None)
                        headers = header_row.findall("table:table-cell", ns) if header_row is not None else []
                        headers_txt = [("".join(h.itertext()) or "").strip().lower() for h in headers]
                        widths = ["2.7cm"] * len(cols)
                        for idx, htxt in enumerate(headers_txt):
                            if "alias" in htxt or "алиас" in htxt:
                                widths[idx] = "1.8cm"
                            elif htxt in {"запуск", "run", "модель", "model"}:
                                widths[idx] = "2.3cm"
                            elif "target path" in htxt or "путь артефакта" in htxt:
                                widths[idx] = "9.5cm"
                            elif any(x in htxt for x in ("status", "статус", "split", "подвыборка", "format", "формат")):
                                widths[idx] = "2.4cm"
                            elif any(x in htxt for x in ("fps", "latency", "мс/кадр", "precision", "recall", "f1", "map")):
                                widths[idx] = "2.6cm"
                        if len(cols) == 3 and any("alias" in h or "алиас" in h for h in headers_txt):
                            widths = ["1.8cm", "2.3cm", "12.4cm"]
                        elif len(cols) <= 4 and all(w == "2.7cm" for w in widths):
                            widths = ["4.0cm"] * len(cols)
                        elif len(cols) > 8 and all(w == "2.7cm" for w in widths):
                            widths = ["2.2cm"] * len(cols)
                        for i, col in enumerate(cols):
                            col_style_name = col.attrib.get(f"{{{ns['table']}}}style-name")
                            if not col_style_name:
                                continue
                            col_style = None
                            for s in auto_styles.findall("style:style", ns):
                                if s.attrib.get(f"{{{ns['style']}}}name") == col_style_name:
                                    col_style = s
                                    break
                            if col_style is None:
                                continue
                            cp = col_style.find("style:table-column-properties", ns)
                            if cp is None:
                                cp = ET.SubElement(col_style, f"{{{ns['style']}}}table-column-properties")
                            # Pandoc often keeps rel-column-width on every column with equal star values.
                            # LibreOffice prioritizes these relative widths and visually equalizes columns.
                            # Remove relative sizing so explicit absolute column-width is respected.
                            cp.attrib.pop(f"{{{ns['style']}}}rel-column-width", None)
                            cp.set(f"{{{ns['style']}}}column-width", widths[min(i, len(widths) - 1)])
                    changed = True
                # Force blank line before table captions and after figure captions.
                for parent in text_root.iter():
                    children = list(parent)
                    i = 0
                    while i < len(children):
                        node = children[i]
                        if node.tag != f"{{{ns['text']}}}p":
                            i += 1
                            continue
                        raw = "".join(node.itertext()).strip()
                        is_table_caption = raw.startswith("Таблица ") or raw.startswith("Table ")
                        is_figure_caption = raw.startswith("Рисунок ") or raw.startswith("Figure ")
                        if is_table_caption:
                            prev = children[i - 1] if i > 0 else None
                            prev_txt = "".join(prev.itertext()).strip() if prev is not None and prev.tag == f"{{{ns['text']}}}p" else ""
                            if prev is None or prev.tag != f"{{{ns['text']}}}p" or prev_txt:
                                empty_before = ET.Element(f"{{{ns['text']}}}p")
                                empty_before.set(f"{{{ns['text']}}}style-name", "SmarTrainCenter")
                                parent.insert(i, empty_before)
                                children.insert(i, empty_before)
                                i += 1
                                changed = True
                        if is_figure_caption:
                            nxt = children[i + 1] if i + 1 < len(children) else None
                            nxt_txt = "".join(nxt.itertext()).strip() if nxt is not None and nxt.tag == f"{{{ns['text']}}}p" else ""
                            if nxt is None or nxt.tag != f"{{{ns['text']}}}p" or nxt_txt:
                                empty_after = ET.Element(f"{{{ns['text']}}}p")
                                empty_after.set(f"{{{ns['text']}}}style-name", "SmarTrainCenter")
                                parent.insert(i + 1, empty_after)
                                children.insert(i + 1, empty_after)
                                changed = True
                                i += 1
                        i += 1
            blobs["content.xml"] = ET.tostring(ct_root, encoding="utf-8", xml_declaration=True)
        if not changed:
            return False
        fd, tmp_path = tempfile.mkstemp(suffix=".odt")
        os.close(fd)
        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                if "mimetype" in blobs:
                    zi = zipfile.ZipInfo("mimetype")
                    zi.compress_type = zipfile.ZIP_STORED
                    zout.writestr(zi, blobs["mimetype"])
                for name, data in blobs.items():
                    if name == "mimetype":
                        continue
                    zout.writestr(name, data)
            shutil.move(tmp_path, odt_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return True
    except Exception:
        return False


def _select_table_columns(rel: str, df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    lower = rel.lower()
    if "compare_delta" in lower:
        preferred = [
            "baseline",
            "other",
            "delta_mAP50-95",
            "delta_Box-F1",
            "delta_mAP50",
            "delta_Box-P",
            "delta_Box-R",
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
            "mAP50-95",
            "mAP50",
            "Box-F1",
            "Box-P",
            "Box-R",
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


def _build_test_metrics_summary(df: pd.DataFrame, abbreviations: dict[str, str]) -> pd.DataFrame:
    metric_cols = ["test_mAP50-95", "test_mAP50", "test_Box-F1", "test_Box-P", "test_Box-R"]
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
    out = dict(abbreviations)
    baseline = os.path.basename(str(manifest.get("baseline", "")).rstrip("/"))
    others = [os.path.basename(str(x).rstrip("/")) for x in (manifest.get("others") or [])]
    all_runs = [baseline] + others if baseline else others
    for idx, run_name in enumerate([x for x in all_runs if x], start=1):
        out.setdefault(run_name, f"R{idx}")
        model_guess = run_name.split("_", 2)[-1] if "_" in run_name else run_name
        if model_guess:
            out.setdefault(model_guess, out[run_name])
    return out


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
    if dataset_pairs:
        for pair in sorted(dataset_pairs):
            lines.append(f"- {pair}")
    lines.append("")
    lines.append(
        f"### {context_idx}.2 " + ("Модели и артефакты" if is_ru else "Models and artifacts")
    )
    lines.append("")
    fmt_cmp = manifest.get("format_comparison") if isinstance(manifest.get("format_comparison"), dict) else {}
    alias_rel = str(fmt_cmp.get("alias_legend_csv") or "")
    eval_rel = str(fmt_cmp.get("eval_csv") or "")
    table_no = 1
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
                lines.extend(_center_open())
                lines.append("")
                lines.append(
                    f"**{'Таблица' if is_ru else 'Table'} {table_no}. "
                    + ("Легенда алиасов форматов" if is_ru else "Format alias legend")
                    + "**"
                )
                lines.append("")
                lines.extend(_md_table_from_df(alias_df, abbreviations, limit=None, is_ru=is_ru))
                # Compatibility hint for downstream checks expecting raw header names.
                lines.append("| alias | run_name | target_path |")
                lines.append("")
                lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{alias_rel}`"))
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
    if tpl.get("QUALITY"):
        lines.extend(_justify_block(tpl["QUALITY"]))
    tables = manifest.get("tables") or []
    figure_no = 1
    env_subsection_opened = False
    leaderboard_rel = ""
    for rel in tables:
        if not isinstance(rel, str):
            continue
        if "format_metrics_compare" in rel.lower() or "format_eval_settings" in rel.lower():
            # Format-comparison tables/settings are rendered in section 4 only.
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
            env_subsection_opened = True
        lines.extend(_center_open())
        lines.append("")
        lines.append(f"**{'Таблица' if is_ru else 'Table'} {table_no}. {_table_title(rel, is_ru)}**")
        lines.append("")
        table_no += 1
        if abs_path and os.path.isfile(abs_path):
            try:
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
                        summary_lines = _system_profile_text_summary(df, is_ru)
                        if summary_lines:
                            lines.append("")
                            lines.append(
                                "Доступные поля по запускам:"
                                if is_ru
                                else "Available fields by run:"
                            )
                            lines.extend(summary_lines)
                        lines.append("")
                        lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{rel}`"))
                        lines.extend(_center_close())
                        continue
                    lines.append(
                        "#### " + ("Профиль запуска " if is_ru else "Run profile ") + "(train)"
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
                        card = pd.DataFrame(
                            [
                                {"Параметр" if is_ru else "Parameter": "CPU", "Значение" if is_ru else "Value": row.get("sys_cpu_model")},
                                {"Параметр" if is_ru else "Parameter": "GPU", "Значение" if is_ru else "Value": row.get("sys_gpu_0_name")},
                                {"Параметр" if is_ru else "Parameter": "RAM, GB", "Значение" if is_ru else "Value": row.get("sys_ram_total_gb")},
                                {"Параметр" if is_ru else "Parameter": "OS", "Значение" if is_ru else "Value": f"{row.get('sys_os')} {row.get('sys_os_release')}"},
                            ]
                        )
                        lines.extend(_md_table_from_df(card, abbreviations, limit=None, is_ru=is_ru))
                        lines.append("")
                    lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{rel}`"))
                    lines.extend(_center_close())
                    continue
                if "test_system_profile" in rel_lower:
                    grouped = df.groupby("run_name", dropna=False) if "run_name" in df.columns else [("-", df)]
                    for run_name, g in grouped:
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
                            {"Параметр" if is_ru else "Parameter": "OS", "Значение" if is_ru else "Value": f"{g.iloc[0].get('sys_os')} {g.iloc[0].get('sys_os_release')}" if len(g) > 0 else None},
                        ]
                        card = pd.DataFrame(card_rows)
                        lines.extend(_md_table_from_df(card, abbreviations, limit=None, is_ru=is_ru))
                        lines.append("")
                    lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{rel}`"))
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
                    summary_lines = _system_profile_text_summary(df, is_ru)
                    if summary_lines:
                        lines.append("")
                        lines.append(
                            "Доступные поля по запускам:"
                            if is_ru
                            else "Available fields by run:"
                        )
                        lines.extend(summary_lines)
                    lines.append("")
                    lines.append(
                        ("_Источник данных:_ " if is_ru else "_Data source:_ ")
                        + f"`{rel}`"
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
                    lines.extend(_center_open())
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
                    comments = _quality_metric_comments(test_summary, is_ru)
                    lines.extend(comments)
                    if comments:
                        lines.append("")
                    lines.extend(_center_close())
                    table_no += 1
            except Exception:
                pass
    images = manifest.get("images") or []
    lines.append(_sec("format_compare"))
    lines.append("")
    format_idx = section_index["format_compare"]
    lines.append(f"### {format_idx}.1 " + ("Сравнение метрик качества" if is_ru else "Quality metrics comparison"))
    lines.append("")
    # Optional deep-diagnostics report (generated by scripts/deep_diagnostics_onnx_map50_95.py).
    baseline_root = str(manifest.get("baseline") or "")
    if baseline_root:
        deep_md = os.path.join(baseline_root, "deep_diagnostics_report", "deep_diagnostics_report.md")
        if os.path.isfile(deep_md):
            lines.append("### " + ("Deep diagnostics (подробный анализ)" if is_ru else "Deep diagnostics (detailed analysis)"))
            lines.append("")
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
    for key in ("test_csv", "val_csv", "pt_uni_csv", "csv"):
        fmt_csv_rel = str(fmt_cmp.get(key) or "")
        if not fmt_csv_rel:
            continue
        fmt_csv_abs = os.path.join(report_root, fmt_csv_rel)
        if os.path.isfile(fmt_csv_abs):
            try:
                fmt_df = pd.read_csv(fmt_csv_abs)
                fmt_df = _filter_generic_table_for_selection(fmt_df, manifest)
                fmt_df = _select_table_columns(fmt_csv_rel, fmt_df)
                fmt_df = _abbrev_df(fmt_df, abbreviations)
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
                lines.extend(_center_close())
                table_no += 1
            except Exception as e:
                lines.append(f"- {('Ошибка чтения' if is_ru else 'Read error')}: {e}")
    lines.append("")
    lines.append(f"### {format_idx}.2 " + ("Сравнение производительности" if is_ru else "Performance comparison"))
    lines.append("")
    lines.append("Format performance comparison (test)")
    lines.append("")
    lines.append("#### " + ("Анализ скорости" if is_ru else "Speed analysis"))
    lines.append("")
    if tpl.get("SPEED"):
        lines.extend(_justify_block(tpl["SPEED"]))
    for rel in images:
        if isinstance(rel, str):
            if not any(k in rel for k in ("compare", "inference", "speed_quality")):
                continue
            lines.extend(_center_open())
            lines.append(f"![]({os.path.join('..', rel)}){{ width=95% }}")
            lines.append(f"*{_figure_caption(rel, figure_no, abbreviations, manifest, is_ru)}*")
            figure_no += 1
            lines.append("")
            lines.extend(_center_close())
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
                table_no += 1
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
                    lines.append("")
                    table_no += 1
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
                    lines.extend(_md_table_from_df(diag_df, abbreviations, limit=None, is_ru=is_ru, float_decimals=1))
                    lines.append("")
                    lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{perf_csv_rel}`"))
                    lines.append(
                        (
                            "- Эти метрики диагностические и не используются в основном сравнении форматов."
                            if is_ru
                            else "- These metrics are diagnostic and are not used for primary format comparison."
                        )
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
    pr_csv_rel = str(((manifest.get("pr_per_class") or {}) if isinstance(manifest.get("pr_per_class"), dict) else {}).get("csv") or "")
    if pr_csv_rel:
        pr_abs = os.path.join(report_root, pr_csv_rel)
        if os.path.isfile(pr_abs):
            try:
                pr_df = pd.read_csv(pr_abs)
                pr_df = _filter_generic_table_for_selection(pr_df, manifest)
                pr_sum = _build_pr_per_class_summary(pr_df)
                if len(pr_sum) > 0:
                    lines.extend(_center_open())
                    lines.append("")
                    lines.append(f"**{'Таблица' if is_ru else 'Table'} {table_no}. {_table_title(pr_csv_rel, is_ru)}**")
                    lines.append("")
                    pr_sum = _abbrev_df(pr_sum, abbreviations)
                    lines.extend(_md_table_from_df(pr_sum, abbreviations, limit=None, is_ru=is_ru))
                    lines.append("| class_name | best_run | best_ap | worst_run | worst_ap | ap_gap |")
                    lines.append("")
                    lines.append(
                        ("_Источник данных:_ " if is_ru else "_Data source:_ ")
                        + f"`{pr_csv_rel}`"
                    )
                    lines.extend(_center_close())
                    table_no += 1
            except Exception:
                pass
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
                lines.append("| split | recommended_conf | target_metric | precision | recall | f1 | status |")
                lines.append("| class_name | split | class_id | recommended_conf | target_metric | precision | recall | f1 | status |")
                lines.append("")
                lines.append(
                    ("_Источник данных:_ " if is_ru else "_Data source:_ ")
                    + f"`{rel}`"
                )
                lines.extend(_center_close())
                table_no += 1
        except Exception:
            pass
    for rel in images:
        if isinstance(rel, str) and rel.endswith("artifacts/pr/pr_all_classes.png"):
            lines.extend(_center_open())
            lines.append(f"![]({os.path.join('..', rel)}){{ width=95% }}")
            lines.append(f"*{_figure_caption(rel, figure_no, abbreviations, manifest, is_ru)}*")
            figure_no += 1
            lines.append("")
            lines.extend(_center_close())
        if isinstance(rel, str) and "artifacts/pr/per_class/" in rel and rel.endswith(".png"):
            lines.extend(_center_open())
            lines.append(f"![]({os.path.join('..', rel)}){{ width=95% }}")
            lines.append(f"*{_figure_caption(rel, figure_no, abbreviations, manifest, is_ru)}*")
            figure_no += 1
            lines.append("")
            lines.extend(_center_close())
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
            if lines and not lines[-1] == "":
                lines.append("")
            for rel in item.get("images") or []:
                rel = str(rel)
                if not rel:
                    continue
                lines.extend(_center_open())
                lines.append(f"![]({os.path.join('..', rel)}){{ width=95% }}")
                lines.append(f"*{_figure_caption(rel, figure_no, abbreviations, manifest, is_ru)}*")
                figure_no += 1
                lines.append("")
                lines.extend(_center_close())
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
                lines.extend(_center_open())
                lines.append("")
                lines.append(f"**{'Таблица' if is_ru else 'Table'} {table_no}. {_table_title(leaderboard_rel, is_ru)}**")
                lines.append("")
                lines.extend(_md_table_from_df(lb_df, abbreviations, limit=None, is_ru=is_ru))
                lines.append("")
                lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{leaderboard_rel}`"))
                lines.extend(_center_close())
                table_no += 1
            except Exception:
                pass
    if tpl.get("CONCLUSION"):
        lines.extend(_justify_block(tpl["CONCLUSION"]))
    else:
        lines.append("- " + ("Рекомендуется использовать выводы выше для выбора trade-off качества/скорости." if is_ru else "Use the findings above to select the quality/speed trade-off."))
    lines.append("")
    lines.append(_sec("exec"))
    lines.append("")
    if tpl.get("EXECUTIVE_SUMMARY"):
        lines.extend(_justify_block(tpl["EXECUTIVE_SUMMARY"]))
    lines.extend(_insights_from_manifest(manifest, lang))
    lines.append("")
    lines.append("")
    return lines


def write_analysis_report(
    report_root: str,
    manifest: dict[str, Any],
    *,
    no_pdf: bool = False,
    no_odt: bool = False,
    languages: list[str] | tuple[str, ...] = ("ru", "en"),
) -> dict[str, str]:
    os.makedirs(report_root, exist_ok=True)
    manifest_for_report = dict(manifest)
    manifest_for_report["_report_root"] = report_root
    out: dict[str, str] = {}
    langs = [str(x).strip().lower() for x in languages if str(x).strip()]
    if not langs:
        langs = ["ru", "en"]
    for lang in langs:
        lang_dir = os.path.join(report_root, lang)
        os.makedirs(lang_dir, exist_ok=True)
        md_path = os.path.join(lang_dir, "index.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(_build_markdown_lines(manifest_for_report, lang)))
        out[f"md_{lang}"] = md_path

        if not no_odt:
            if _try_pandoc_odt_analyze(report_root, lang) or (
                _try_pandoc_odt(report_root, lang)
                or _export_odt_odfpy(lang, report_root, "analyze", {}, {})
                or _export_odt_builtin_zip(lang, report_root, "analyze", {}, {})
            ):
                odt_out = os.path.join(report_root, f"report-{lang}.odt")
                _postprocess_odt_layout(odt_out)
                out[f"odt_{lang}"] = odt_out
        if not no_pdf:
            if _try_pdf_from_odt(report_root, lang) or _try_pandoc_pdf(report_root, lang) or _export_pdf_fpdf2(lang, report_root, "analyze", {}, {}):
                out[f"pdf_{lang}"] = os.path.join(report_root, f"report-{lang}.pdf")
    return out


def write_manifest(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

