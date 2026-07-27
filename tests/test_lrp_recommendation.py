"""Unit tests for Optimal LRP recommendations."""

from __future__ import annotations

from pathlib import Path

from smartrain.core.training.confidence_recommendation import (
    compute_confidence_recommendations,
    recommendations_complete,
)
from smartrain.core.training.lrp_recommendation import (
    compute_lrp_at_score,
    compute_lrp_recommendations,
    lrp_error,
    maybe_write_lrp_recommendations,
    optimal_lrp_threshold,
    read_lrp_recommendation_file,
)


def test_lrp_error_perfect_and_all_fp() -> None:
    assert lrp_error(n_tp=2, n_fp=0, n_fn=0, sum_one_minus_iou=0.0, iou_thr=0.5) == 0.0
    err = lrp_error(n_tp=0, n_fp=3, n_fn=2, sum_one_minus_iou=0.0, iou_thr=0.5)
    assert err == 1.0


def test_optimal_lrp_picks_threshold_on_synthetic_tp_fp() -> None:
    # 2 GT. High-score TP (iou=0.9), mid-score FP, low-score TP (iou=0.8).
    dets = [
        {"score": 0.9, "class_id": 0, "iou": 0.9},
        {"score": 0.6, "class_id": 0, "iou": 0.0},  # FP
        {"score": 0.4, "class_id": 0, "iou": 0.8},
    ]
    gt = {0: 2}
    # At 0.7: 1 TP + 0 FP + 1 FN
    e_hi = compute_lrp_at_score(dets, gt, score_thr=0.7, iou_thr=0.5, class_id=0)
    # At 0.5: 1 TP + 1 FP + 1 FN
    e_mid = compute_lrp_at_score(dets, gt, score_thr=0.5, iou_thr=0.5, class_id=0)
    # At 0.3: 2 TP + 1 FP + 0 FN
    e_lo = compute_lrp_at_score(dets, gt, score_thr=0.3, iou_thr=0.5, class_id=0)
    assert e_hi is not None and e_mid is not None and e_lo is not None
    assert e_mid > e_hi  # FP hurts
    opt = optimal_lrp_threshold(
        dets,
        gt,
        iou_thr=0.5,
        class_id=0,
        conf_grid=[0.3, 0.5, 0.7, 0.95],
    )
    assert opt["status"] == "ok"
    assert opt["threshold"] in {0.3, 0.7}  # mid FP threshold should lose
    assert opt["threshold"] != 0.5


def test_compute_lrp_recommendations_payload() -> None:
    dets = [
        {"score": 0.8, "class_id": 0, "iou": 0.9},
        {"score": 0.2, "class_id": 1, "iou": 0.0},
    ]
    payload = compute_lrp_recommendations(
        dets,
        {0: 1, 1: 1},
        split="test",
        class_names={0: "a", 1: "b"},
        conf_grid=[0.1, 0.5, 0.9],
    )
    assert payload["objective"] == "D"
    assert payload["status"] == "ok"
    assert "global" in payload and payload["global"]["threshold"] is not None
    assert len(payload["per_class"]) == 2


def test_maybe_write_noop_without_flag(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    assert maybe_write_lrp_recommendations(model_dir=str(run), split="test", compute_lrp=False) is None
    assert list(run.rglob("lrp_recommendations_*.json")) == []


def test_maybe_write_skip_without_matches(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    path = maybe_write_lrp_recommendations(model_dir=str(run), split="test", compute_lrp=True)
    assert path is not None
    payload = read_lrp_recommendation_file(path)
    assert payload is not None
    assert payload["status"] == "skipped"
    assert "prediction_gt" in str(payload.get("reason") or "")


def test_abc_unchanged_when_lrp_not_requested() -> None:
    class _Fake:
        names = {0: "a"}
        curves_results = None
        box = None

    payload = compute_confidence_recommendations(_Fake(), split="test")
    assert set(payload["objectives"].keys()) == {"A", "B", "C"}
    assert "D" not in payload["objectives"]
    assert recommendations_complete(payload) is True
