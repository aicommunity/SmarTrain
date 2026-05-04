from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from smartrain.domain.canonical.validators import validate_payload
from smartrain.orchestrators.canonical_gateway import (
    load_metrics,
    load_predictions,
    load_target,
    resolve_task_context,
)


def test_resolve_task_context_and_load_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run1"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"task_type": "detection", "dataset": {"name": "ds"}}}),
        encoding="utf-8",
    )
    (run_dir / "test_metrics.csv").write_text("mAP50-95,Box-F1\n0.5,0.6\n", encoding="utf-8")

    ctx = resolve_task_context(str(run_dir), source_kind="run")
    assert ctx.task_type == "detection"
    assert ctx.model_id == "run1"
    assert ctx.dataset_ref == "ds"

    metrics = load_metrics(str(run_dir), source_kind="run")
    assert len(metrics) >= 1
    pt_rows = [m for m in metrics if m.namespace == "detection/test_pt"]
    assert pt_rows
    assert any(str(m.raw_path).endswith("test_metrics.csv") for m in pt_rows)

    base = load_target(str(run_dir), source_kind="run")
    merged = replace(base, metrics=metrics)
    validate_payload(merged)


def test_load_predictions_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run2"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")
    assert load_predictions(str(run_dir), source_kind="run") == []
