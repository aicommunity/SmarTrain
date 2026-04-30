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


def _md_table_from_df(df: pd.DataFrame, abbreviations: dict[str, str], limit: int | None = None) -> list[str]:
    if len(df) == 0:
        return ["_No data._"]
    preview = df.head(limit).copy() if isinstance(limit, int) and limit > 0 else df.copy()
    cols = [abbreviations.get(str(c), str(c)) for c in preview.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in preview.iterrows():
        vals = []
        for real_col in preview.columns:
            v = row.get(real_col)
            if pd.isna(v):
                vals.append("-")
                continue
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                sv = str(v)
                vals.append(abbreviations.get(sv, sv))
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
                        p.set(f"{{{ns['text']}}}style-name", center_style_name)
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
            "backend_status",
            "throughput_img_s",
            "latency_p50_ms",
            "latency_p95_ms",
        ]
    elif "format_eval_settings" in lower:
        preferred = [
            "alias",
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
            "dataset_name",
            "format",
            "test_backend",
            "test_provider",
            "test_device",
            "sys_cpu_model",
            "sys_cpu_logical_cores",
            "sys_ram_total_gb",
            "sys_gpu_0_name",
            "sys_gpu_0_vram_gb",
            "sys_gpu_total_vram_gb",
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
    scoped = df[sys_cols].copy()
    scoped = scoped.replace("", np.nan)
    total = int(scoped.size)
    if total <= 0:
        return False
    empty = int(scoped.isna().sum().sum())
    return (empty / float(total)) > 0.70


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
        best_idx = int(s.idxmax())
        worst_idx = int(s.idxmin())
        best_run = str(df.loc[best_idx, "run"])
        worst_run = str(df.loc[worst_idx, "run"])
        best_val = float(s.loc[best_idx])
        worst_val = float(s.loc[worst_idx])
        if is_ru:
            lines.append(f"- {metric}: лучший запуск **{best_run}** ({best_val:.4f}), худший **{worst_run}** ({worst_val:.4f}).")
        else:
            lines.append(f"- {metric}: best run **{best_run}** ({best_val:.4f}), worst **{worst_run}** ({worst_val:.4f}).")
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
    if isinstance(sources, dict):
        for by_metric in sources.values():
            if not isinstance(by_metric, dict):
                continue
            recomputed += sum(1 for v in by_metric.values() if v == "recomputed")
            missing += sum(1 for v in by_metric.values() if v == "missing")
    cache = manifest.get("cache") or {}
    hits = int(cache.get("hits", 0)) if isinstance(cache, dict) else 0
    misses = int(cache.get("misses", 0)) if isinstance(cache, dict) else 0
    if lang == "ru":
        lines.append(f"- Переоценённых метрик: **{recomputed}**, отсутствующих: **{missing}**.")
        lines.append(f"- Кэш single-run: **hit={hits}**, **miss={misses}**.")
    else:
        lines.append(f"- Recomputed metrics: **{recomputed}**, missing metrics: **{missing}**.")
        lines.append(f"- Single-run cache: **hit={hits}**, **miss={misses}**.")
    report_root = str(manifest.get("_report_root") or "")
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
    lines.append(_sec("quality"))
    lines.append("")
    if tpl.get("QUALITY"):
        lines.extend(_justify_block(tpl["QUALITY"]))
    report_root = manifest.get("_report_root") or ""
    tables = manifest.get("tables") or []
    table_no = 1
    figure_no = 1
    for rel in tables:
        if not isinstance(rel, str):
            continue
        if "format_metrics_compare" in rel.lower() or "format_eval_settings" in rel.lower():
            # Format-comparison tables/settings are rendered in section 4 only.
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
                    lines.append(
                        ("_Источник данных:_ " if is_ru else "_Data source:_ ")
                        + f"`{rel}`"
                    )
                    lines.extend(_center_close())
                    continue
                df = _abbrev_df(df, abbreviations)
                lines.extend(_md_table_from_df(df, abbreviations, limit=None))
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
                    lines.extend(_md_table_from_df(test_summary, abbreviations, limit=None))
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
    lines.append(_sec("speed"))
    lines.append("")
    if tpl.get("SPEED"):
        lines.extend(_justify_block(tpl["SPEED"]))
    images = manifest.get("images") or []
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
    lines.append(_sec("format_compare"))
    lines.append("")
    lines.append("### 4.1 " + ("Сравнение метрик качества" if is_ru else "Quality metrics comparison"))
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
                        "- Детализация по run/split/format:"
                        if is_ru
                        else "- Detailed list by run/split/format:"
                    )
                )
                for item in issues_payload:
                    if not isinstance(item, dict):
                        continue
                    run_name = str(item.get("run_name") or "-")
                    split_name = str(item.get("split") or "-")
                    fmt = str(item.get("format") or "-")
                    status = str(item.get("status") or "-")
                    reason_code = str(item.get("reason_code") or "unknown").strip() or "unknown"
                    reason = str(item.get("reason") or "-").replace("\n", " ").strip()
                    lines.append(f"- `{run_name}` / `{split_name}` / `{fmt}`: `{status}` / `{reason_code}` - {reason}")
                lines.append("")
    eval_rel = str(fmt_cmp.get("eval_csv") or "")
    alias_rel = str(fmt_cmp.get("alias_legend_csv") or "")
    if alias_rel:
        alias_abs = os.path.join(report_root, alias_rel)
        if os.path.isfile(alias_abs):
            try:
                alias_df = pd.read_csv(alias_abs)
                alias_df = _filter_generic_table_for_selection(alias_df, manifest)
                preferred_alias = [c for c in ("alias", "format", "run_name", "target_path") if c in alias_df.columns]
                if preferred_alias:
                    alias_df = alias_df[preferred_alias]
                alias_df = _abbrev_df(alias_df, abbreviations)
                lines.extend(_center_open())
                lines.append("")
                lines.append(
                    f"**{'Таблица' if is_ru else 'Table'} {table_no}. "
                    + ("Легенда алиасов форматов" if is_ru else "Format alias legend")
                    + "**"
                )
                lines.append("")
                lines.extend(_md_table_from_df(alias_df, abbreviations, limit=None))
                lines.append("")
                lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{alias_rel}`"))
                lines.extend(_center_close())
                table_no += 1
            except Exception as e:
                lines.append(f"- {('Ошибка чтения' if is_ru else 'Read error')}: {e}")
    if eval_rel:
        lines.append(
            (
                "- Параметры расчета (`imgsz/conf/iou`) вынесены в отдельную справочную таблицу ниже."
                if is_ru
                else "- Calculation parameters (`imgsz/conf/iou`) are provided in a separate reference table below."
            )
        )
        lines.append("")
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
                lines.extend(_md_table_from_df(fmt_df, abbreviations, limit=None))
                lines.append("")
                lines.append(
                    ("_Источник данных:_ " if is_ru else "_Data source:_ ")
                    + f"`{fmt_csv_rel}`"
                )
                lines.extend(_center_close())
                table_no += 1
            except Exception as e:
                lines.append(f"- {('Ошибка чтения' if is_ru else 'Read error')}: {e}")
    lines.append("")
    lines.append("### 4.2 " + ("Сравнение производительности" if is_ru else "Performance comparison"))
    lines.append("")
    for key in ("perf_test_csv",):
        perf_csv_rel = str(fmt_cmp.get(key) or "")
        if not perf_csv_rel:
            continue
        perf_csv_abs = os.path.join(report_root, perf_csv_rel)
        if os.path.isfile(perf_csv_abs):
            try:
                perf_df = pd.read_csv(perf_csv_abs)
                perf_df = _filter_generic_table_for_selection(perf_df, manifest)
                perf_df = _select_table_columns(perf_csv_rel, perf_df)
                perf_df = _abbrev_df(perf_df, abbreviations)
                lines.extend(_center_open())
                lines.append("")
                lines.append(f"**{'Таблица' if is_ru else 'Table'} {table_no}. {_table_title(perf_csv_rel, is_ru)}**")
                lines.append("")
                lines.extend(_md_table_from_df(perf_df, abbreviations, limit=None))
                lines.append("")
                lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{perf_csv_rel}`"))
                lines.extend(_center_close())
                table_no += 1
            except Exception as e:
                lines.append(f"- {('Ошибка чтения' if is_ru else 'Read error')}: {e}")
    if eval_rel:
        eval_abs = os.path.join(report_root, eval_rel)
        if os.path.isfile(eval_abs):
            try:
                eval_df = pd.read_csv(eval_abs)
                eval_df = _filter_generic_table_for_selection(eval_df, manifest)
                eval_df = _select_table_columns(eval_rel, eval_df)
                eval_df = _abbrev_df(eval_df, abbreviations)
                lines.extend(_center_open())
                lines.append("")
                lines.append(f"**{'Таблица' if is_ru else 'Table'} {table_no}. {_table_title(eval_rel, is_ru)}**")
                lines.append("")
                lines.extend(_md_table_from_df(eval_df, abbreviations, limit=None))
                lines.append("")
                if all(c in eval_df.columns for c in ("eval_imgsz", "eval_conf", "eval_iou")):
                    eval_clean = eval_df.dropna(subset=["eval_imgsz", "eval_conf", "eval_iou"], how="any")
                    if len(eval_clean) == 0:
                        lines.append(
                            (
                                "- Часть параметров расчета не найдена; сопоставимость проверяется по доступным данным."
                                if is_ru
                                else "- Some calculation parameters are missing; comparability is evaluated using available data."
                            )
                        )
                    else:
                        uniq = eval_clean[["eval_imgsz", "eval_conf", "eval_iou"]].drop_duplicates()
                        if len(uniq) > 1:
                            lines.append(
                                (
                                    "- **Обнаружены расхождения параметров расчета метрик (`imgsz/conf/iou`) между форматами. "
                                    "Сравнение значений может быть методически несопоставимым.**"
                                    if is_ru
                                    else "- **Metric calculation parameter mismatches (`imgsz/conf/iou`) were detected across formats. "
                                         "Comparisons may be methodologically inconsistent.**"
                                )
                            )
                lines.append("")
                lines.append((("_Источник данных:_ " if is_ru else "_Data source:_ ") + f"`{eval_rel}`"))
                lines.extend(_center_close())
                table_no += 1
            except Exception as e:
                lines.append(f"- {('Ошибка чтения' if is_ru else 'Read error')}: {e}")
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
                    lines.extend(_md_table_from_df(pr_sum, abbreviations, limit=None))
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
                lines.extend(_md_table_from_df(local_df, abbreviations, limit=None))
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

