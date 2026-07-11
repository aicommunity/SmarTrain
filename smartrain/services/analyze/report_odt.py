"""ODT/PDF post-processing for analyze reports."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET

from smartrain.services.reporting.document_export import _pandoc_executable, _try_pandoc_odt

_FIGURE_CAPTION_TAIL_RE = re.compile(r"((?:Рисунок|Figure) \d+\..*)$")
_FIGURE_CAPTION_START_RE = re.compile(r"^(?:Рисунок|Figure) \d+\.")
_LEGEND_LINE_SPLIT_RE = re.compile(r"(?=M\d+ ·)")


def _figure_caption_tail(raw: str) -> str | None:
    m = _FIGURE_CAPTION_TAIL_RE.search(str(raw or "").strip())
    return m.group(1).strip() if m else None


def _is_standalone_figure_caption(raw: str) -> bool:
    return bool(_FIGURE_CAPTION_START_RE.match(str(raw or "").strip()))


def _split_merged_figure_paragraphs(
    text_root: ET.Element,
    ns: dict[str, str],
    *,
    center_style_name: str,
    caption_style_name: str,
) -> bool:
    """Split pandoc paragraphs that combine image, run legend, and figure caption."""
    text_tag = f"{{{ns['text']}}}p"
    changed = False
    for parent in text_root.iter():
        children = list(parent)
        idx = 0
        while idx < len(children):
            p = children[idx]
            if p.tag != text_tag:
                idx += 1
                continue
            frame = p.find("draw:frame", ns)
            if frame is None:
                idx += 1
                continue
            raw = "".join(p.itertext()).strip()
            caption_text = _figure_caption_tail(raw)
            if not caption_text:
                idx += 1
                continue
            legend_blob = raw[: raw.rfind(caption_text)].strip()
            legend_lines = [ln.strip() for ln in _LEGEND_LINE_SPLIT_RE.split(legend_blob) if ln.strip()]
            parent.remove(p)
            insert_at = idx
            img_p = ET.Element(text_tag)
            img_p.set(f"{{{ns['text']}}}style-name", center_style_name)
            img_p.append(frame)
            parent.insert(insert_at, img_p)
            insert_at += 1
            for line in legend_lines:
                leg_p = ET.Element(text_tag)
                leg_p.set(f"{{{ns['text']}}}style-name", center_style_name)
                leg_p.text = line
                parent.insert(insert_at, leg_p)
                insert_at += 1
            cap_p = ET.Element(text_tag)
            cap_p.set(f"{{{ns['text']}}}style-name", caption_style_name)
            cap_p.text = caption_text
            parent.insert(insert_at, cap_p)
            children = list(parent)
            idx = insert_at + 1
            changed = True
    return changed

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
                ttp = ET.SubElement(ts, f"{{{ns['style']}}}text-properties")
                ttp.set(f"{{{ns['fo']}}}font-size", "9pt")
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
                if _split_merged_figure_paragraphs(
                    text_root,
                    ns,
                    center_style_name=center_style_name,
                    caption_style_name=caption_style_name,
                ):
                    changed = True
                table_paragraphs = set(text_root.findall(".//table:table-cell//text:p", ns))
                for p in text_root.iterfind(".//text:p", ns):
                    raw = "".join(p.itertext()).strip()
                    existing_style = p.attrib.get(f"{{{ns['text']}}}style-name", "")
                    if existing_style in (
                        center_style_name,
                        caption_style_name,
                        table_text_style_name,
                        table_header_text_style,
                    ):
                        continue
                    if p in table_paragraphs:
                        p.set(f"{{{ns['text']}}}style-name", table_text_style_name)
                        changed = True
                        continue
                    if _is_standalone_figure_caption(raw) or raw.startswith(("Таблица ", "Table ")):
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
                                widths[idx] = "2.7cm"
                            elif htxt in {"запуск", "run", "модель", "model"}:
                                widths[idx] = "2.3cm"
                            elif "target path" in htxt or "путь артефакта" in htxt:
                                widths[idx] = "9.5cm"
                            elif any(x in htxt for x in ("status", "статус", "split", "подвыборка", "format", "формат")):
                                widths[idx] = "2.4cm"
                            elif any(x in htxt for x in ("fps", "latency", "мс/кадр", "precision", "recall", "f1", "map")):
                                widths[idx] = "2.6cm"
                        if len(cols) == 3 and any("alias" in h or "алиас" in h for h in headers_txt):
                            widths = ["2.7cm", "2.3cm", "11.5cm"]
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
                        is_figure_caption = _is_standalone_figure_caption(raw)
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


