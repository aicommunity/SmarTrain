from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pandas as pd


def build_leaderboard_records(
    *,
    runs: list[str],
    speed_metric: str,
    quality_metric: str,
    weight_quality: float,
    weight_speed: float,
    weight_stability: float,
    load_run_record: Callable[[str], Any],
    read_test_performance_by_format_artifacts: Callable[[str], dict[str, list[dict[str, Any]]]],
    compute_composite_score: Callable[..., float | None],
) -> list[dict[str, Any]]:
    def _resolve_speed_metric_from_performance(run_dir: str, metric_name: str) -> float | None:
        metric = str(metric_name or "").strip().lower()
        if not metric:
            return None
        perf_by_fmt = read_test_performance_by_format_artifacts(run_dir)
        candidates: list[float] = []
        for rows in perf_by_fmt.values():
            for row in rows:
                perf = row.get("performance") if isinstance(row, dict) else None
                if not isinstance(perf, dict):
                    continue
                value: Any = None
                if metric in {"avg_inference_fps", "throughput_img_s"}:
                    value = perf.get("throughput_img_s")
                elif metric in {"avg_inference_ms_per_frame", "latency_p50_ms"}:
                    latency_ms = perf.get("latency_ms")
                    if isinstance(latency_ms, dict):
                        steady = latency_ms.get("steady")
                        all_stats = latency_ms.get("all")
                        if isinstance(steady, dict) and steady.get("p50") is not None:
                            value = steady.get("p50")
                        elif isinstance(all_stats, dict):
                            value = all_stats.get("p50")
                elif metric == "latency_p95_ms":
                    latency_ms = perf.get("latency_ms")
                    if isinstance(latency_ms, dict):
                        steady = latency_ms.get("steady")
                        all_stats = latency_ms.get("all")
                        if isinstance(steady, dict) and steady.get("p95") is not None:
                            value = steady.get("p95")
                        elif isinstance(all_stats, dict):
                            value = all_stats.get("p95")
                try:
                    if value is None:
                        continue
                    fv = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fv):
                    candidates.append(fv)
        if not candidates:
            return None
        if "fps" in metric or "throughput" in metric:
            return float(max(candidates))
        return float(min(candidates))

    records: list[dict[str, Any]] = []
    for run_dir in runs:
        try:
            rec = load_run_record(run_dir)
        except Exception as exc:
            print(f"[WARN] {run_dir}: failed to load run ({exc})")
            continue
        speed_value = rec.test_metrics.get(speed_metric)
        if speed_value is None or (isinstance(speed_value, float) and pd.isna(speed_value)):
            fallback_speed = _resolve_speed_metric_from_performance(run_dir, speed_metric)
            if fallback_speed is not None:
                rec.test_metrics[speed_metric] = fallback_speed
        score = compute_composite_score(
            rec,
            weight_quality=weight_quality,
            weight_speed=weight_speed,
            weight_stability=weight_stability,
            quality_metric=quality_metric,
            speed_metric=speed_metric,
        )
        records.append(
            {
                "run_dir": rec.run_dir,
                "model": rec.model,
                "dataset_name": rec.dataset_name,
                "training_ok": rec.training_ok,
                "testing_ok": rec.testing_ok,
                "quality_metric": rec.test_metrics.get(quality_metric),
                "speed_metric": rec.test_metrics.get(speed_metric),
                "composite_score": score,
            }
        )
    return records


def write_leaderboard_csv(*, records: list[dict[str, Any]], out_csv: str) -> int:
    if not records:
        return 1
    df = pd.DataFrame(records)
    df = df.dropna(subset=["composite_score"]).sort_values("composite_score", ascending=False)
    if len(df) == 0:
        return 1
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Leaderboard CSV: {out_csv}")
    return 0
