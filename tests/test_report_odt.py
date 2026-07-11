from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from smartrain.services.analyze import report_odt


def test_try_pandoc_odt_analyze_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(report_odt, "_pandoc_executable", lambda: None)
    monkeypatch.setattr(report_odt, "_try_pandoc_odt", lambda root, lang: True)
    assert report_odt._try_pandoc_odt_analyze(str(tmp_path), "en") is True


def test_postprocess_odt_layout_updates_styles(tmp_path: Path) -> None:
    odt = tmp_path / "report-en.odt"
    styles = """<?xml version='1.0' encoding='UTF-8'?>
<office:document-styles xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0">
<office:automatic-styles></office:automatic-styles>
</office:document-styles>"""
    content = """<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"></office:document-content>"""
    with zipfile.ZipFile(odt, "w") as zf:
        zf.writestr("styles.xml", styles)
        zf.writestr("content.xml", content)
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")

    assert report_odt._postprocess_odt_layout(str(odt)) is True


def test_split_merged_figure_paragraphs() -> None:
    ns = {
        "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
        "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
        "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    }
    for pfx, uri in ns.items():
        ET.register_namespace(pfx, uri)
    content = f"""<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office="{ns['office']}"
 xmlns:text="{ns['text']}"
 xmlns:draw="{ns['draw']}">
<office:body><office:text>
<text:p text:style-name="SmarTrainCenter">
<draw:frame draw:name="img1"><draw:image /></draw:frame>
M1 · yolo11m · D1 — базовый M2 · yolo11m · D2 M3 · yolo11m · D3 Рисунок 1. Caption text
</text:p>
</office:text></office:body>
</office:document-content>"""
    root = ET.fromstring(content)
    text_root = root.find("office:body/office:text", ns)
    assert text_root is not None
    assert report_odt._split_merged_figure_paragraphs(
        text_root,
        ns,
        center_style_name="SmarTrainCenter",
        caption_style_name="SmarTrainCaption",
    )
    paragraphs = text_root.findall("text:p", ns)
    assert len(paragraphs) == 5
    assert paragraphs[0].find("draw:frame", ns) is not None
    assert (paragraphs[1].text or "").startswith("M1 ·")
    assert (paragraphs[2].text or "").startswith("M2 ·")
    assert (paragraphs[3].text or "").startswith("M3 ·")
    assert (paragraphs[4].text or "").startswith("Рисунок 1.")
    assert paragraphs[4].attrib.get(f"{{{ns['text']}}}style-name") == "SmarTrainCaption"
