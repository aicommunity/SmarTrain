from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _stats_ns(values_ns: list[int]) -> dict[str, float | int | None]:
    if not values_ns:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "std": None,
        }
    arr = np.asarray(values_ns, dtype=np.float64) / 1_000_000.0
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


@dataclass
class DualPerfProfiler:
    warmup_images: int = 5
    _started_ns: int = field(default_factory=time.perf_counter_ns)
    _ended_ns: int | None = None
    _end_to_end_ns: list[int] = field(default_factory=list)
    _infer_only_ns: list[int] = field(default_factory=list)
    _stages_ns: dict[str, list[int]] = field(default_factory=dict)

    def record_end_to_end(self, dt_ns: int) -> None:
        self._end_to_end_ns.append(int(max(0, dt_ns)))

    def record_infer_only(self, dt_ns: int) -> None:
        self._infer_only_ns.append(int(max(0, dt_ns)))

    def record_stage(self, stage: str, dt_ns: int) -> None:
        key = str(stage or "").strip()
        if not key:
            return
        self._stages_ns.setdefault(key, []).append(int(max(0, dt_ns)))

    def finish(self) -> None:
        if self._ended_ns is None:
            self._ended_ns = time.perf_counter_ns()

    def _profile_payload(self, values_ns: list[int], duration_s: float) -> dict[str, Any]:
        warm = int(max(0, self.warmup_images))
        steady_values = values_ns[warm:]
        throughput = float(len(steady_values) / duration_s) if duration_s > 0 and steady_values else 0.0
        return {
            "images_total": int(len(values_ns)),
            "warmup_images": warm,
            "duration_s": duration_s,
            "throughput_img_s": throughput,
            "latency_ms": {
                "all": _stats_ns(values_ns),
                "steady": _stats_ns(steady_values),
            },
        }

    def to_payload(self, *, methodology: dict[str, Any] | None = None) -> dict[str, Any]:
        self.finish()
        ended = self._ended_ns if self._ended_ns is not None else time.perf_counter_ns()
        duration_s = max(0.0, float(ended - self._started_ns) / 1_000_000_000.0)
        return {
            "end_to_end": self._profile_payload(self._end_to_end_ns, duration_s),
            "infer_only": self._profile_payload(self._infer_only_ns, duration_s),
            "stage_breakdown_ms": {k: _stats_ns(v) for k, v in self._stages_ns.items()},
            "methodology": methodology or {},
        }
