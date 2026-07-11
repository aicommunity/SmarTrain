"""Narrative blocks (subsection intros, table preambles, takeaways) in analyze_report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_subsection_intro_after_context_dataset_heading(tmp_path: Path) -> None:
    from smartrain.services.analyze.report_writer import write_analysis_report

    (tmp_path / "artifacts" / "metrics").mkdir(parents=True, exist_ok=True)
    rs = tmp_path / "artifacts" / "metrics" / "runs_summary.csv"
    pd.DataFrame(
        [
            {"run_name": "a", "test_mAP50-95": 0.5},
            {"run_name": "b", "test_mAP50-95": 0.9},
        ]
    ).to_csv(rs, index=False)

    manifest: dict = {
        "session_name": "s_narr",
        "profile": "full",
        "baseline": "",
        "others": [],
        "tables": ["artifacts/metrics/runs_summary.csv"],
        "images": [],
        "artifacts": [],
        "format_comparison": {},
        "abbreviations": {},
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True)
    ru_md = (tmp_path / "ru" / "index.md").read_text(encoding="utf-8")
    idx = ru_md.find("### 2.1")
    assert idx != -1
    tail = ru_md[idx : idx + 800]
    assert "Датасет" in tail
    assert "::: {style=" in tail
    assert "Здесь зафиксированы датасеты" in tail


def test_table_preamble_before_first_table_and_runs_summary_takeaways(tmp_path: Path) -> None:
    from smartrain.services.analyze.report_writer import write_analysis_report

    (tmp_path / "artifacts" / "metrics").mkdir(parents=True, exist_ok=True)
    rs = tmp_path / "artifacts" / "metrics" / "runs_summary.csv"
    pd.DataFrame(
        [
            {"run_name": "run_a", "test_mAP50-95": 0.91, "test_mAP50": 0.8},
            {"run_name": "run_b", "test_mAP50-95": 0.42, "test_mAP50": 0.4},
        ]
    ).to_csv(rs, index=False)

    manifest: dict = {
        "session_name": "s_rs",
        "profile": "full",
        "baseline": "",
        "others": [],
        "tables": ["artifacts/metrics/runs_summary.csv"],
        "images": [],
        "artifacts": [],
        "format_comparison": {},
        "abbreviations": {},
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True)
    ru_md = (tmp_path / "ru" / "index.md").read_text(encoding="utf-8")
    pos_tbl = ru_md.find("**Таблица")
    assert pos_tbl != -1
    assert "**Таблица 1.**" in ru_md or "**Таблица 1." in ru_md
    assert "Сводка параметров" in ru_md or "runs_summary" in ru_md
    assert "лучший" in ru_md or "худший" in ru_md
