from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from smartrain.unified.domain.validators import validate_unified_payload
from smartrain.orchestrators.unified_gateway import (
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
    validate_unified_payload(merged)


def test_load_predictions_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run2"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")
    assert load_predictions(str(run_dir), source_kind="run") == []


def test_load_predictions_discovers_debug_jsonl(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run3"
    dd = run_dir / "tests" / "test-pt" / "deep_diagnostics"
    dd.mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")
    (dd / "debug_test.jsonl").write_text("{\"a\":1}\n{\"a\":2}\n", encoding="utf-8")

    preds = load_predictions(str(run_dir), source_kind="run", split="test")
    assert len(preds) == 1
    assert preds[0].count == 2
    assert preds[0].items_path.endswith("debug_test.jsonl")


def test_load_metrics_uses_task_specific_metrics_adapter(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run_cls"
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "models" / "best-cls.onnx").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"task_type": "classification", "dataset": {"name": "ds"}}}),
        encoding="utf-8",
    )
    (run_dir / "test_metrics_onnx.csv").write_text("top1,top5,mAP50-95\n0.8,0.95,0.3\n", encoding="utf-8")

    metrics = load_metrics(str(run_dir), source_kind="run", format_name="onnx")
    assert metrics
    assert metrics[0].namespace == "classification/test_onnx"
    assert metrics[0].primary_metrics.get("top1") == 0.8
    assert metrics[0].primary_metrics.get("top5") == 0.95
