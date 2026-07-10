from __future__ import annotations

import zipfile
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
