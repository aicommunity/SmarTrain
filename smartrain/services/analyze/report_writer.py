"""Write analyze session manifest and multilingual report artifacts."""

from __future__ import annotations

import json
import os
from typing import Any

from smartrain.services.analyze.report_markdown import _build_markdown_lines
from smartrain.services.analyze.report_odt import (
    _postprocess_odt_layout,
    _try_pandoc_odt_analyze,
    _try_pdf_from_odt,
)
from smartrain.services.analyze.schema_contracts import ensure_analyze_session_manifest
from smartrain.services.reporting.document_export import (
    _export_odt_builtin_zip,
    _export_odt_odfpy,
    _export_pdf_fpdf2,
    _try_pandoc_odt,
    _try_pandoc_pdf,
)

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


def write_manifest(path: str, payload: dict[str, Any], *, session_type: str = "analyze_all") -> None:
    normalized = ensure_analyze_session_manifest(
        payload,
        session_type="compare" if str(session_type).strip().lower() == "compare" else "analyze_all",
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

