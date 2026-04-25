from __future__ import annotations

from pathlib import Path

from smartrain import cli


def test_analyze_help_examples_use_current_compare_flags() -> None:
    text = cli.HELP_ANALYZE_GROUP
    assert "--baseline" in text
    assert "--others" in text
    assert "--run-dir" not in text


def test_analyze_docs_examples_use_current_compare_flags() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    doc = (repo_root / "docs" / "cli" / "analyze.md").read_text(encoding="utf-8")
    assert "analyze compare --baseline" in doc
    assert "--run-dir" not in doc
    assert "leaderboard" in doc
    assert "report-ru.pdf" in doc
    assert "--pr-per-class" in doc
    assert "--report-languages" in doc
    assert "--scatter-x" in doc

