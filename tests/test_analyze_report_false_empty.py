"""Regression for false «нет данных» placeholders in analyze reports."""

from __future__ import annotations

import pandas as pd

from smartrain.services.analyze.report_markdown_formatting import (
    _abbrev_df,
    _filter_runs_summary_for_selection,
)
from smartrain.services.analyze.report_sections.report_manifest import _exec_runs_metrics_dataframe


def test_exec_metrics_omits_release_comment_when_all_nan() -> None:
    df = pd.DataFrame(
        {
            "run_name": ["detect_a", "detect_b"],
            "release_comment": [float("nan"), float("nan")],
            "test_metrics/mAP50-95(B)": [0.8, 0.9],
            "train_image_size": [640, 640],
        }
    )
    out = _exec_runs_metrics_dataframe(df, abbreviations={})
    assert "release_comment" not in out.columns
    assert "run_name" in out.columns


def test_exec_metrics_keeps_release_comment_when_present() -> None:
    df = pd.DataFrame(
        {
            "run_name": ["detect_a", "detect_b"],
            "release_comment": ["note", ""],
            "test_metrics/mAP50-95(B)": [0.8, 0.9],
        }
    )
    out = _exec_runs_metrics_dataframe(df, abbreviations={})
    assert "release_comment" in out.columns


def test_abbrev_df_keeps_distinct_train_best_last_map_columns() -> None:
    df = pd.DataFrame(
        {
            "run_name": ["r1"],
            "train_best_metrics/mAP50-95(B)": [0.1],
            "train_last_metrics/mAP50-95(B)": [0.2],
        }
    )
    out = _abbrev_df(df, abbreviations={})
    assert len(out.columns) == 3
    assert len(set(out.columns)) == 3
    # Must not raise on dtype access
    for c in out.columns:
        _ = out[c].dtype


def test_filter_runs_summary_dedupes_models_over_runs() -> None:
    df = pd.DataFrame(
        {
            "run_name": ["same_stem", "same_stem", "other"],
            "run_dir": [
                "/ws/models/ds/same_stem",
                "/ws/runs/ds/same_stem",
                "/ws/models/ds/other",
            ],
        }
    )
    manifest = {
        "baseline": "/ws/models/ds/same_stem",
        "others": ["/ws/models/ds/other"],
    }
    out = _filter_runs_summary_for_selection(df, manifest)
    assert len(out) == 2
    assert set(out["run_name"].astype(str)) == {"same_stem", "other"}
    assert all("/models/" in p for p in out["run_dir"].astype(str))
