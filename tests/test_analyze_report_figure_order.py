"""Ordering checks for analyze_report markdown."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_analyze_report_figure_one_before_table_eleven(tmp_path: Path) -> None:
    from smartrain.workflows.analyze.analyze_report import write_analysis_report

    (tmp_path / "artifacts" / "compare").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "metrics").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "compare" / "compare_curves.png").write_bytes(b"x")

    tables: list[str] = []
    for i in range(11):
        rel = f"artifacts/metrics/stub_{i:02d}.csv"
        pd.DataFrame([{"n": i}]).to_csv(tmp_path / rel, index=False)
        tables.append(rel)

    manifest: dict = {
        "session_name": "s_fig_order",
        "profile": "full",
        "baseline": "run_a",
        "others": [],
        "tables": tables,
        "images": ["artifacts/compare/compare_curves.png"],
        "artifacts": [],
        "format_comparison": {},
        "abbreviations": {},
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True)
    ru_md = (tmp_path / "ru" / "index.md").read_text(encoding="utf-8")
    pos_fig = ru_md.find("Рисунок 1.")
    pos_tbl11 = ru_md.find("Таблица 11.")
    assert pos_fig != -1 and pos_tbl11 != -1
    assert pos_fig < pos_tbl11
