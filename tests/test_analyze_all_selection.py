"""Tests for analyze all run selection (baseline-only / single-run mode)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from smartrain.services.analyze.all_finalize import finalize_all_session
from smartrain.services.analyze.all_selection import prepare_all_selection


def test_prepare_all_selection_non_interactive_baseline_only() -> None:
    args = SimpleNamespace(
        baseline="/w/runs/ds/run_a",
        others=None,
        profile="full",
        models_root="/w/runs",
    )
    baseline, others, profile, used = prepare_all_selection(
        args,
        filtered_run_records_cb=MagicMock(),
        prompt_int_cb=MagicMock(side_effect=AssertionError("prompt_int must not be called")),
        prompt_text_cb=MagicMock(side_effect=AssertionError("prompt_text must not be called")),
        prompt_choice_cb=MagicMock(side_effect=AssertionError("prompt_choice must not be called")),
    )
    assert baseline == "/w/runs/ds/run_a"
    assert others == []
    assert profile == "full"
    assert used is False


def test_prepare_all_selection_interactive_single_run_auto_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    rec = SimpleNamespace(model="m1", dataset_name="ds")
    indexed = [("/w/runs/ds/run_only", rec)]
    prompt_int = MagicMock(side_effect=AssertionError("prompt_int must not be called"))
    prompt_text = MagicMock(side_effect=AssertionError("prompt_text must not be called"))
    prompt_choice = MagicMock(return_value="quality")

    args = SimpleNamespace(baseline=None, others=None, profile=None, models_root="/w/runs")
    baseline, others, profile, used = prepare_all_selection(
        args,
        filtered_run_records_cb=lambda _a: indexed,
        prompt_int_cb=prompt_int,
        prompt_text_cb=prompt_text,
        prompt_choice_cb=prompt_choice,
    )
    assert baseline == "/w/runs/ds/run_only"
    assert others == []
    assert profile == "quality"
    assert used is True
    prompt_int.assert_not_called()
    prompt_text.assert_not_called()
    prompt_choice.assert_called_once()
    assert args.baseline == baseline
    assert args.others == []
    assert args.profile == "quality"


def test_prepare_all_selection_interactive_no_runs_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    args = SimpleNamespace(baseline=None, others=None, profile=None, models_root="/w/runs")

    with pytest.raises(SystemExit) as exc:
        prepare_all_selection(
            args,
            filtered_run_records_cb=lambda _a: [],
            prompt_int_cb=MagicMock(),
            prompt_text_cb=MagicMock(),
            prompt_choice_cb=MagicMock(),
        )
    assert exc.value.code == 1


def test_finalize_all_session_sets_single_run_mode_in_manifest(tmp_path) -> None:
    captured: dict = {}

    def write_manifest(path: str, payload: dict) -> None:
        captured["path"] = path
        captured["payload"] = payload

    args = SimpleNamespace(
        no_pdf=True,
        no_odt=True,
        strict_diagnostics=False,
        scatter_x="avg_inference_ms_per_frame",
        scatter_y="mAP50-95",
        report_languages="en",
        recompute_missing_metrics_choice="no",
    )
    session_root = str(tmp_path / "session")
    finalize_all_session(
        args=args,
        session_root=session_root,
        profile="quality",
        baseline="/w/runs/ds/run_a",
        others=[],
        data_yaml="/w/datasets/ds/data.yaml",
        report_languages=["en"],
        run_data_yaml_map={"/w/runs/ds/run_a": "/w/datasets/ds/data.yaml"},
        unresolved_data_yaml_runs=[],
        artifacts=[],
        cache_events=[],
        artifact_failures=[],
        metric_sources_payload={"sources": {}},
        recompute_missing_metrics=False,
        build_abbreviations_for_report_cb=lambda runs: {"run_a": "R1"},
        collect_ultralytics_test_artifacts_cb=lambda *a, **k: ([], []),
        write_format_compare_artifacts_cb=lambda *a, **k: None,
        collect_confidence_recommendation_tables_cb=lambda *a, **k: {},
        write_manifest_cb=write_manifest,
        write_analysis_report_cb=lambda *a, **k: {"en": str(tmp_path / "en" / "index.md")},
        record_failure_cb=MagicMock(),
        replay_parser=None,
    )
    assert captured["payload"]["single_run_mode"] is True
    assert captured["payload"]["others"] == []


def test_finalize_all_session_stores_baseline_workspace_relative(tmp_path: Path) -> None:
    captured: dict = {}

    def write_manifest(path: str, payload: dict) -> None:
        captured["payload"] = payload

    run_a = tmp_path / "runs" / "ds" / "run_a"
    run_a.mkdir(parents=True)
    data_yaml = tmp_path / "datasets" / "ds" / "data.yaml"
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text("train: images\n", encoding="utf-8")

    args = SimpleNamespace(
        no_pdf=True,
        no_odt=True,
        strict_diagnostics=False,
        scatter_x="avg_inference_ms_per_frame",
        scatter_y="mAP50-95",
        report_languages="en",
        recompute_missing_metrics_choice="no",
        workspace=str(tmp_path),
    )
    finalize_all_session(
        args=args,
        session_root=str(tmp_path / "analytics" / "session"),
        profile="quality",
        baseline=str(run_a),
        others=[],
        data_yaml=str(data_yaml),
        report_languages=["en"],
        run_data_yaml_map={str(run_a): str(data_yaml)},
        unresolved_data_yaml_runs=[],
        artifacts=[],
        cache_events=[],
        artifact_failures=[],
        metric_sources_payload={"sources": {}},
        recompute_missing_metrics=False,
        build_abbreviations_for_report_cb=lambda runs: {"run_a": "R1"},
        collect_ultralytics_test_artifacts_cb=lambda *a, **k: ([], []),
        write_format_compare_artifacts_cb=lambda *a, **k: None,
        collect_confidence_recommendation_tables_cb=lambda *a, **k: {},
        write_manifest_cb=write_manifest,
        write_analysis_report_cb=lambda *a, **k: {"en": str(tmp_path / "en" / "index.md")},
        record_failure_cb=MagicMock(),
        replay_parser=None,
    )
    assert captured["payload"]["baseline"] == "runs/ds/run_a"
    assert "\\" not in captured["payload"]["baseline"]
    assert captured["payload"]["run_data_yaml_map"] == {"runs/ds/run_a": "datasets/ds/data.yaml"}


def test_report_context_mentions_single_run_mode(tmp_path: Path) -> None:
    from smartrain.services.analyze.report_writer import write_analysis_report

    (tmp_path / "artifacts" / "metrics").mkdir(parents=True, exist_ok=True)
    rs = tmp_path / "artifacts" / "metrics" / "runs_summary.csv"
    pd.DataFrame([{"run_name": "run_a", "test_mAP50-95": 0.5}]).to_csv(rs, index=False)

    manifest: dict = {
        "session_name": "s_single",
        "profile": "quality",
        "baseline": "/w/runs/ds/run_a",
        "others": [],
        "single_run_mode": True,
        "tables": ["artifacts/metrics/runs_summary.csv"],
        "images": [],
        "artifacts": [],
        "format_comparison": {},
        "abbreviations": {"run_a": "R1"},
    }
    write_analysis_report(str(tmp_path), manifest, no_pdf=True, no_odt=True, languages=["en"])
    en_md = (tmp_path / "en" / "index.md").read_text(encoding="utf-8")
    assert "single-run report" in en_md.lower()
