"""Shared Markdown/ODT/PDF export helpers for analyze and dataset reports."""

from __future__ import annotations

import html
import os
import shutil
import subprocess
import sys
import zipfile
from typing import Any

def _log(msg: str) -> None:
    print(msg, flush=True)


def _xml_attr(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


UI: dict[str, dict[str, str]] = {
    "en": {
        "doc_title": "Dataset sample report",
        "dataset": "Dataset",
        "generated": "Generated",
        "class_heading": "Class",
        "examples_heading": "Examples",
        "example_caption": "Example",
        "no_instances": "No labeled instances found for this class.",
        "footer_classes": "Classes in data.yaml",
        "stats_heading": "Dataset statistics",
        "stats_classes": "Classes (with objects)",
        "stats_images": "Images (total)",
        "stats_labeled": "Labeled images (≥1 object)",
        "stats_empty": "Empty images (no objects)",
        "stats_empty_pct": "Empty share",
        "stats_instances": "Instances (total)",
        "stats_per_split": "Per split",
        "stats_split_images": "images",
        "stats_split_objects": "objects",
        "stats_imbalance": "Imbalance (max/min per class)",
        "stats_gini": "Gini (class distribution)",
        "stats_quality": "Quality (basic checks)",
        "stats_quality_ok": "OK",
        "stats_quality_warn": "WARN",
        "stats_issues": "Issues",
        "stats_broken_lines": "broken label lines",
        "stats_unknown_ids": "unknown class ids",
        "stats_orphan_images": "images without label file",
        "stats_orphan_labels": "label files without image",
        "stats_no_splits": "No per-split counts (unusual layout).",
    },
    "ru": {
        "doc_title": "Отчёт по примерам датасета",
        "dataset": "Датасет",
        "generated": "Сформировано",
        "class_heading": "Класс",
        "examples_heading": "Примеры",
        "example_caption": "Пример",
        "no_instances": "Для этого класса нет размеченных экземпляров.",
        "footer_classes": "Классы в data.yaml",
        "stats_heading": "Статистика датасета",
        "stats_classes": "Классов (с объектами)",
        "stats_images": "Изображений (всего)",
        "stats_labeled": "Размеченных изображений (≥1 объект)",
        "stats_empty": "Пустых изображений (без объектов)",
        "stats_empty_pct": "Доля пустых",
        "stats_instances": "Экземпляров (всего)",
        "stats_per_split": "По сплитам",
        "stats_split_images": "изображений",
        "stats_split_objects": "объектов",
        "stats_imbalance": "Дисбаланс (макс/мин по классу)",
        "stats_gini": "Джини (по классам)",
        "stats_quality": "Качество (базовые проверки)",
        "stats_quality_ok": "OK",
        "stats_quality_warn": "ВНИМАНИЕ",
        "stats_issues": "Замечания",
        "stats_broken_lines": "битых строк разметки",
        "stats_unknown_ids": "неизвестных id класса",
        "stats_orphan_images": "изображений без файла разметки",
        "stats_orphan_labels": "файлов разметки без изображения",
        "stats_no_splits": "Нет счётчиков по сплитам (нестандартная структура).",
    },
}


def _t(lang: str, key: str) -> str:
    block = UI.get(lang) or UI["en"]
    return block.get(key) or UI["en"][key]

def _resolve_pandoc_path(candidate: str) -> str | None:
    """
    ``pypandoc.get_pandoc_path()`` often returns ``.../pandoc`` on Windows while the real binary is
    ``pandoc.exe`` next to it (pypandoc-binary wheel). Accept either form.
    """
    c = (candidate or "").strip()
    if not c:
        return None
    if os.path.isfile(c):
        return c
    if os.name == "nt":
        if c.lower().endswith(".exe"):
            return None
        with_exe = f"{c}.exe"
        if os.path.isfile(with_exe):
            return with_exe
        parent, base = os.path.split(c)
        if base == "pandoc":
            alt = os.path.join(parent, "pandoc.exe")
            if os.path.isfile(alt):
                return alt
    return None


def resolve_pandoc_executable(*, quiet: bool = False) -> str | None:
    """PANDOC env, then PATH, then bundled pandoc from ``pypandoc-binary`` (base dependency)."""
    raw = (os.environ.get("PANDOC") or "").strip()
    if raw:
        if os.path.isfile(raw):
            return raw
        resolved = _resolve_pandoc_path(raw)
        if resolved:
            return resolved
        w = shutil.which(raw)
        if w:
            return w
    w = shutil.which("pandoc")
    if w:
        return w
    try:
        import pypandoc

        p = pypandoc.get_pandoc_path()
        resolved = _resolve_pandoc_path(p) if p else None
        if resolved:
            if not quiet:
                _log(f"[INFO] Using bundled pandoc: {resolved}")
            return resolved
    except Exception as e:
        if not quiet:
            _log(f"[INFO] pypandoc/bundled pandoc unavailable ({e}); reinstall smartrain or set PANDOC.")
    return None


def _pandoc_executable() -> str | None:
    return resolve_pandoc_executable(quiet=False)


def check_weasyprint_ready() -> tuple[bool, str]:
    if _weasyprint_import_ok():
        return True, "import ok"
    return False, "not installed or missing OS libraries (Cairo/Pango)"


def check_pandoc_ready(*, quiet: bool = True) -> tuple[bool, str]:
    exe = resolve_pandoc_executable(quiet=quiet)
    if exe:
        return True, exe
    return False, "not found (reinstall smartrain or set PANDOC)"


def _pandoc_resource_path(out_dir: str, lang: str) -> str:
    """Search dirs for images etc. (pandoc resolves relative URIs from the .md location)."""
    parts = [
        out_dir,
        os.path.join(out_dir, lang),
        os.path.join(out_dir, "assets"),
    ]
    return os.pathsep.join(parts)


def _weasyprint_import_ok() -> bool:
    """WeasyPrint is often installed but unusable on Windows without GTK/Pango; skip noisy pandoc attempts."""
    try:
        import weasyprint  # noqa: F401

        return True
    except Exception:
        return False


def _pandoc_pdf_engine_variants() -> list[list[str]]:
    """
    Try engines that avoid a full TeX stack first (typst, weasyprint, wkhtmltopdf).
    Pandoc's default is often pdflatex — often missing on minimal hosts.
    """
    variants: list[list[str]] = []
    if shutil.which("typst"):
        variants.append(["--pdf-engine=typst"])
    if shutil.which("weasyprint") and _weasyprint_import_ok():
        variants.append(["--pdf-engine=weasyprint"])
    if shutil.which("wkhtmltopdf"):
        variants.append(["--pdf-engine=wkhtmltopdf"])
    variants.append([])  # pandoc default (often pdflatex)
    if shutil.which("xelatex"):
        variants.append(["--pdf-engine=xelatex"])
    if shutil.which("lualatex"):
        variants.append(["--pdf-engine=lualatex"])
    if shutil.which("pdflatex"):
        variants.append(["--pdf-engine=pdflatex"])
    # de-dupe while preserving order
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for v in variants:
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _try_pandoc_pdf(out_dir: str, lang: str) -> bool:
    exe = _pandoc_executable()
    if not exe:
        _log("[WARNING] pandoc: executable not found (set PANDOC=/full/path if it is outside PATH).")
        return False
    rel_md = f"{lang}/index.md"
    rel_pdf = f"report-{lang}.pdf"
    rp = _pandoc_resource_path(out_dir, lang)
    last_err = ""
    for extra in _pandoc_pdf_engine_variants():
        cmd = [exe, rel_md, "-o", rel_pdf, "--resource-path", rp, *extra]
        try:
            r = subprocess.run(
                cmd,
                cwd=out_dir,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if r.returncode == 0:
                if extra:
                    _log(f"[INFO] pandoc PDF for [{lang}] using {' '.join(extra)}")
                return True
            tail = (r.stderr or r.stdout or "").strip()
            last_err = tail[-4000:] if tail else f"exit {r.returncode}"
            _log(f"[INFO] pandoc PDF attempt failed ({' '.join(extra) or 'default engine'}): {last_err[:800]}")
        except (OSError, subprocess.TimeoutExpired) as e:
            last_err = str(e)
            _log(f"[INFO] pandoc PDF attempt error ({extra}): {e}")
    _log(f"[WARNING] pandoc could not build PDF for [{lang}] after all engines. Last error (truncated):\n{last_err[:2000]}")
    return False


def _try_pandoc_odt(out_dir: str, lang: str) -> bool:
    exe = _pandoc_executable()
    if not exe:
        _log("[WARNING] pandoc: executable not found (set PANDOC=/full/path if it is outside PATH).")
        return False
    rel_md = f"{lang}/index.md"
    rel_odt = f"report-{lang}.odt"
    rp = _pandoc_resource_path(out_dir, lang)
    try:
        r = subprocess.run(
            [exe, rel_md, "-o", rel_odt, "--resource-path", rp],
            cwd=out_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r.returncode == 0:
            return True
        err = (r.stderr or r.stdout or "").strip()
        _log(f"[WARNING] pandoc ODT for [{lang}] failed (exit {r.returncode}): {err[:2000]}")
    except (OSError, subprocess.TimeoutExpired) as e:
        _log(f"[WARNING] pandoc ODT for [{lang}]: {e}")
    return False


def _fpdf_dejavu_paths() -> tuple[str | None, str | None]:
    """(regular_ttf, bold_ttf). Bold may equal regular if no Bold file found."""
    reg: str | None = None
    try:
        import fpdf

        base = Path(fpdf.__file__).resolve().parent
        for sub in ("font", "fonts"):
            d = base / sub
            r = d / "DejaVuSans.ttf"
            b = d / "DejaVuSans-Bold.ttf"
            if r.is_file():
                reg = str(r)
                if b.is_file():
                    return reg, str(b)
                return reg, reg
    except Exception:
        pass
    try:
        import matplotlib.font_manager as fm

        reg_p = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
        if reg_p and os.path.isfile(reg_p) and "dejavu" in os.path.basename(reg_p).lower():
            bold_p = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
            reg = reg_p
            if (
                bold_p
                and os.path.isfile(bold_p)
                and os.path.normcase(bold_p) != os.path.normcase(reg_p)
                and "dejavu" in os.path.basename(bold_p).lower()
            ):
                return reg, bold_p
            return reg, reg
    except Exception:
        pass
    windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or r"C:\Windows"
    if os.name == "nt":
        fonts = os.path.join(windir, "Fonts")
        for reg_name, bold_name in (
            ("arial.ttf", "arialbd.ttf"),
            ("calibri.ttf", "calibrib.ttf"),
            ("segoeui.ttf", "segoeuib.ttf"),
        ):
            rp = os.path.join(fonts, reg_name)
            bp = os.path.join(fonts, bold_name)
            if os.path.isfile(rp):
                return rp, bp if os.path.isfile(bp) else rp
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    ):
        if os.path.isfile(p):
            reg = p
            break
    if not reg:
        return None, None
    bold_cand = os.path.join(os.path.dirname(reg), "DejaVuSans-Bold.ttf")
    if os.path.isfile(bold_cand):
        return reg, bold_cand
    return reg, reg


def _export_pdf_fpdf2(lang: str, out_dir: str, dataset_name: str, picks: dict[int, list[tuple[Any, str]]], classes_by_id: dict[int, str]) -> bool:
    try:
        from fpdf import FPDF
    except ImportError:
        return False

    font_path, bold_path = _fpdf_dejavu_paths()
    if not font_path:
        _log(
            "[WARNING] fpdf2 PDF: no Unicode TTF found (DejaVu/matplotlib/WINDOWS\\Fonts). "
            "Install fonts-dejavu (Linux) or ensure matplotlib or system fonts are available."
        )
        return False

    pdf_path = os.path.join(out_dir, f"report-{lang}.pdf")
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()
        try:
            pdf.add_font("ReportFont", "", font_path)
            pdf.add_font("ReportFont", "B", bold_path or font_path)
        except Exception as e:
            _log(f"[WARNING] fpdf2: add_font failed: {e}")
            return False
        font = "ReportFont"
        pdf.set_font(font, size=14)
        pdf.multi_cell(0, 8, f"{_t(lang, 'doc_title')}: {dataset_name}")
        pdf.ln(4)

        for cid in sorted(classes_by_id.keys()):
            cname = classes_by_id[cid]
            pdf.set_font(font, style="B", size=12)
            pdf.multi_cell(0, 7, f"{_t(lang, 'class_heading')}: {cname}")
            pdf.set_font(font, size=10)
            items = picks.get(cid) or []
            if not items:
                pdf.multi_cell(0, 6, _t(lang, "no_instances"))
                pdf.ln(2)
                continue
            for _inst, rel in items:
                img_abs = os.path.join(out_dir, rel.replace("/", os.sep))
                if os.path.isfile(img_abs):
                    try:
                        pdf.image(img_abs, w=min(160, pdf.w - 24))
                    except Exception as ie:
                        _log(f"[WARNING] fpdf2: skip image {img_abs!r}: {ie}")
                    pdf.ln(4)
        pdf.output(pdf_path)
        return True
    except Exception as e:
        _log(f"[WARNING] fpdf2 PDF failed for [{lang}]: {e}")
        try:
            if os.path.isfile(pdf_path):
                os.remove(pdf_path)
        except OSError:
            pass
        return False


def _export_odt_builtin_zip(
    lang: str,
    out_dir: str,
    dataset_name: str,
    picks: dict[int, list[tuple[Any, str]]],
    classes_by_id: dict[int, str],
) -> bool:
    """
    Minimal valid ODT (OpenDocument Text) via zip + XML — no odfpy required.
    """
    odt_path = os.path.join(out_dir, f"report-{lang}.odt")
    title = html.escape(f"{_t(lang, 'doc_title')}: {dataset_name}")
    body_parts: list[str] = [f'<text:p>{title}</text:p>', "<text:p/>"]

    manifest_entries: list[tuple[str, str]] = [
        ("/", "application/vnd.oasis.opendocument.text"),
        ("content.xml", "text/xml"),
        ("styles.xml", "text/xml"),
        ("meta.xml", "text/xml"),
    ]
    pic_index = 0
    picture_internal: list[str] = []

    for cid in sorted(classes_by_id.keys()):
        cname = classes_by_id[cid]
        body_parts.append(f'<text:p>{html.escape(_t(lang, "class_heading"))}: {html.escape(cname)} (id={cid})</text:p>')
        items = picks.get(cid) or []
        if not items:
            body_parts.append(f'<text:p>{html.escape(_t(lang, "no_instances"))}</text:p>')
            continue
        for _inst, rel in items:
            img_abs = os.path.join(out_dir, rel.replace("/", os.sep))
            if not os.path.isfile(img_abs):
                continue
            ext = os.path.splitext(img_abs)[1].lower() or ".png"
            if ext not in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                ext = ".png"
            mt = "image/png" if ext == ".png" else "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            internal = f"Pictures/img{pic_index}{ext}"
            picture_internal.append((internal, img_abs, mt))
            manifest_entries.append((internal, mt))
            href_esc = _xml_attr(internal)
            body_parts.append(
                "<text:p>"
                '<draw:frame text:anchor-type="paragraph" draw:z-index="0" '
                'svg:width="15cm" svg:height="11cm">'
                f'<draw:image xlink:href="{href_esc}" xlink:type="simple" xlink:show="embed" xlink:actuate="onLoad"/>'
                "</draw:frame>"
                "</text:p>"
            )
            pic_index += 1
        body_parts.append("<text:p/>")

    manifest_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">',
    ]
    for full_path, media in manifest_entries:
        manifest_lines.append(
            f'<manifest:file-entry manifest:full-path="{_xml_attr(full_path)}" '
            f'manifest:media-type="{_xml_attr(media)}"/>'
        )
    manifest_lines.append("</manifest:manifest>")
    manifest_xml = "\n".join(manifest_lines)

    content_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
        'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" '
        'office:version="1.2">'
        "<office:body><office:text>"
        + "".join(body_parts)
        + "</office:text></office:body></office:document-content>"
    )

    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<office:document-styles xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'office:version="1.2"><office:styles/></office:document-styles>'
    )
    meta_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0" office:version="1.2">'
        "<office:meta><meta:generator>smartrain dataset_report</meta:generator></office:meta>"
        "</office:document-meta>"
    )

    try:
        with zipfile.ZipFile(odt_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zi = zipfile.ZipInfo("mimetype")
            zi.compress_type = zipfile.ZIP_STORED
            zf.writestr(zi, "application/vnd.oasis.opendocument.text")
            zf.writestr("META-INF/manifest.xml", manifest_xml.encode("utf-8"))
            zf.writestr("content.xml", content_xml.encode("utf-8"))
            zf.writestr("styles.xml", styles_xml.encode("utf-8"))
            zf.writestr("meta.xml", meta_xml.encode("utf-8"))
            for internal, abs_path, _mt in picture_internal:
                with open(abs_path, "rb") as f:
                    zf.writestr(internal, f.read())
        return True
    except Exception as e:
        _log(f"[WARNING] built-in ODT failed for [{lang}]: {e}")
        try:
            if os.path.isfile(odt_path):
                os.remove(odt_path)
        except OSError:
            pass
        return False


def _export_odt_odfpy(lang: str, out_dir: str, dataset_name: str, picks: dict[int, list[tuple[Any, str]]], classes_by_id: dict[int, str]) -> bool:
    try:
        from odf.draw import Frame, Image
        from odf.opendocument import OpenDocumentText
        from odf.text import H, P
    except ImportError:
        _log("[INFO] odfpy import failed; using built-in ODT zip writer.")
        return False

    doc = OpenDocumentText()
    doc.text.addElement(H(outlinelevel=1, text=f"{_t(lang, 'doc_title')}: {dataset_name}"))
    doc.text.addElement(P(text=""))

    for cid in sorted(classes_by_id.keys()):
        cname = classes_by_id[cid]
        doc.text.addElement(H(outlinelevel=2, text=f"{_t(lang, 'class_heading')}: {cname}"))
        items = picks.get(cid) or []
        if not items:
            doc.text.addElement(P(text=_t(lang, "no_instances")))
            continue
        for _inst, rel in items:
            img_abs = os.path.join(out_dir, rel.replace("/", os.sep))
            if os.path.isfile(img_abs):
                try:
                    href = doc.addPicture(img_abs)
                    frame = Frame(width="16cm", height="12cm", zindex="0")
                    frame.addElement(Image(href=href, type="simple"))
                    doc.text.addElement(frame)
                except Exception:
                    doc.text.addElement(P(text=img_abs))
            doc.text.addElement(P(text=""))

    odt_path = os.path.join(out_dir, f"report-{lang}.odt")
    try:
        doc.save(odt_path)
        return True
    except Exception as e:
        _log(f"[WARNING] odfpy ODT save failed for [{lang}]: {e}")
        return False
