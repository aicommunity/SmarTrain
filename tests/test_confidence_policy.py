"""Tests for confidence threshold policy and micro/macro aggregations."""

from __future__ import annotations

from smartrain.core.training.confidence_policy import (
    DEFAULT_OBJECTIVE,
    resolve_inference_confidence,
)
from smartrain.core.training.confidence_recommendation import (
    DEFAULT_FALLBACK_CONFIDENCE,
    compute_confidence_recommendations,
)


class _FakeMetrics:
    def __init__(self, *, with_support: bool = False) -> None:
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
        if with_support:
            self.nt_per_class = [10, 100]


def test_payload_includes_macro_and_micro_aggregations() -> None:
    payload = compute_confidence_recommendations(_FakeMetrics(), split="test")
    obj = payload["objectives"]["A"]
    assert obj["aggregation"] == "macro"
    assert "macro" in obj["aggregations"] and "micro" in obj["aggregations"]
    micro = obj["aggregations"]["micro"]["global"]
    assert micro["aggregation"] == "micro"
    assert micro["status"] == "fallback"
    assert "support_unavailable" in str(micro.get("reason") or "")


def test_micro_uses_support_when_available() -> None:
    payload = compute_confidence_recommendations(_FakeMetrics(with_support=True), split="test")
    micro = payload["objectives"]["A"]["aggregations"]["micro"]["global"]
    assert micro["status"] == "ok"
    assert micro["threshold"] is not None


def test_policy_defaults_to_a_macro_fallback() -> None:
    assert DEFAULT_OBJECTIVE == "A"
    assert resolve_inference_confidence(None) == DEFAULT_FALLBACK_CONFIDENCE
    payload = compute_confidence_recommendations(_FakeMetrics(), split="test")
    assert resolve_inference_confidence(payload) == payload["objectives"]["A"]["global"]["threshold"]
    assert resolve_inference_confidence(payload, objective="B") == payload["objectives"]["B"]["global"]["threshold"]
