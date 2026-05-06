from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from smartrain.core.training.confidence_recommendation import (
    compute_confidence_recommendations,
    recommendation_file_path,
    recommendations_complete,
    write_not_available_recommendations,
    write_recommendation_file,
)
from smartrain.results_analyzer import _collect_confidence_recommendation_tables


class _FakeMetrics:
    def __init__(self) -> None:
        self.names = {0: "cat", 1: "dog"}
        conf = [0.1, 0.3, 0.6]
        p = [
            [0.7, 0.8, 0.9],
            [0.6, 0.7, 0.85],
        ]
        r = [
            [0.9, 0.8, 0.6],
            [0.85, 0.75, 0.5],
        ]
        f1 = [
            [0.7875, 0.8, 0.72],
            [0.7058, 0.7241, 0.6296],
        ]
        self.curves_results = [
            (conf, p, "Confidence", "Precision", "P-curve"),
            (conf, r, "Confidence", "Recall", "R-curve"),
            (conf, f1, "Confidence", "F1", "F1-curve"),
        ]


def test_compute_confidence_recommendations_builds_a_b_c() -> None:
    payload = compute_confidence_recommendations(
        _FakeMetrics(),
        split="test",
        beta_recall=2.0,
        beta_precision=0.5,
        fallback_confidence=0.25,
    )
    assert payload["split"] == "test"
    assert set(payload["objectives"].keys()) == {"A", "B", "C"}
    assert payload["objectives"]["A"]["global"]["threshold"] == 0.3
    assert len(payload["objectives"]["A"]["per_class"]) == 2
    assert recommendations_complete(payload) is True


def test_write_not_available_recommendations_creates_contract(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = write_not_available_recommendations(
        model_dir=str(run_dir),
        split="val",
        reason="missing_split",
    )
    payload = json.loads(Path(out).read_text(encoding="utf-8"))
    assert payload["status"] == "not_available"
    assert payload["objectives"]["A"]["global"]["threshold"] == 0.25
    assert recommendations_complete(payload) is True


def test_collect_confidence_recommendation_tables_exports_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run_a"
    run_dir.mkdir(parents=True, exist_ok=True)
    md = {"training_info": {"model": "yolo11n", "dataset": {"name": "ds"}}}
    (run_dir / "training_metadata.json").write_text(json.dumps(md), encoding="utf-8")

    payload = compute_confidence_recommendations(_FakeMetrics(), split="test")
    write_recommendation_file(recommendation_file_path(str(run_dir), "test"), payload)
    write_recommendation_file(recommendation_file_path(str(run_dir), "val"), payload | {"split": "val"})

    out = _collect_confidence_recommendation_tables([str(run_dir)], str(tmp_path / "out"))
    assert set(out.keys()) == {"A", "B", "C"}

    df = pd.read_csv(out["A"])
    assert {"run_dir", "level", "recommended_conf", "objective"}.issubset(df.columns)
    assert set(df["level"].astype(str).unique()) >= {"global", "class"}
