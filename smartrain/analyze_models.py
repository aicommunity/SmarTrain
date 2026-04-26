from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RunRecord:
    run_dir: str
    model: str | None = None
    dataset_name: str | None = None
    training_ok: bool | None = None
    testing_ok: bool | None = None
    training_duration_s: float | None = None
    test_metrics: dict[str, Any] = field(default_factory=dict)
    train_last_metrics: dict[str, Any] = field(default_factory=dict)

