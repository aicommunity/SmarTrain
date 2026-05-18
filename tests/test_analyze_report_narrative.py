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
    idx = ru_md.find("### 1.1")
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
    pos_tbl = ru_md.find("**Таблица 1.")
    assert pos_tbl != -1
    head = ru_md[max(0, pos_tbl - 1200) : pos_tbl]
    assert "::: {style=" in head

    assert "Сводка test-метрик" in ru_md
    pos_extra = ru_md.find("Сводка test-метрик")
    assert pos_extra != -1
    after = ru_md[pos_extra : pos_extra + 2500]
    assert "лучший" in after or "худший" in after
