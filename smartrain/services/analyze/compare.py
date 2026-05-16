from __future__ import annotations

from typing import Any

import pandas as pd

from smartrain.services.analyze.models import RunRecord


def build_delta_rows(
    baseline: str,
    baseline_metrics: dict[str, Any],
    others: list[str],
    other_metrics_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for other, om in zip(others, other_metrics_rows):
        row: dict[str, Any] = {"baseline": baseline, "other": other}
        keys = set(baseline_metrics) | set(om)
        for k in keys:
            if k is None or str(k).strip() == "":
                continue
            try:
                bv = float(baseline_metrics[k]) if k in baseline_metrics and pd.notna(baseline_metrics.get(k)) else None
                ov = float(om[k]) if k in om and pd.notna(om.get(k)) else None
            except (TypeError, ValueError):
                continue
            if bv is not None and ov is not None:
                row[f"delta_{k}"] = ov - bv
        rows.append(row)
    return rows


def compute_composite_score(
    rec: RunRecord,
    *,
    weight_quality: float,
    weight_speed: float,
    weight_stability: float,
    quality_metric: str,
    speed_metric: str,
) -> float | None:
    def _safe_float(v: Any) -> float | None:
        try:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    q = _safe_float(rec.test_metrics.get(quality_metric))
    s = _safe_float(rec.test_metrics.get(speed_metric))
    stable = 1.0 if rec.training_ok and rec.testing_ok else 0.0
    if q is None and s is None:
        return None
    speed_component = None
    if s is not None:
        speed_component = s if "fps" in speed_metric.lower() else (1.0 / s if s > 0 else 0.0)
    parts: list[tuple[float, float]] = []
    if q is not None:
        parts.append((weight_quality, q))
    if speed_component is not None:
        parts.append((weight_speed, speed_component))
    parts.append((weight_stability, stable))
    denom = sum(w for w, _ in parts)
    if denom <= 0:
        return None
    return sum(w * v for w, v in parts) / denom


def generate_compare_insights(
    baseline: str,
    others: list[str],
    delta_rows: list[dict[str, Any]],
) -> list[str]:
    lines = [f"Baseline: {baseline}", f"Candidates: {len(others)}", ""]
    for row in delta_rows:
        other = str(row.get("other", "unknown"))
        positive = []
        negative = []
        for k, v in row.items():
            if not str(k).startswith("delta_"):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            metric = str(k)[len("delta_") :]
            if fv > 0:
                positive.append((metric, fv))
            elif fv < 0:
                negative.append((metric, fv))
        positive.sort(key=lambda x: x[1], reverse=True)
        negative.sort(key=lambda x: x[1])
        lines.append(f"[{other}]")
        if positive:
            top = ", ".join(f"{m}: {v:+.4f}" for m, v in positive[:3])
            lines.append(f"  better: {top}")
        if negative:
            worst = ", ".join(f"{m}: {v:+.4f}" for m, v in negative[:3])
            lines.append(f"  worse: {worst}")
        if not positive and not negative:
            lines.append("  no numeric deltas available.")
        lines.append("")
    return lines

