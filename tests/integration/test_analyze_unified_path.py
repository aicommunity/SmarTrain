from __future__ import annotations

import json
from pathlib import Path

import smartrain.workflows.analyze.results_analyzer as results_analyzer
from smartrain.core.runtime.workspace_paths import deploy_workspace


def test_analyze_unified_path_uses_gateway_metrics_and_predictions(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))

    run_dir = tmp_path / "runs" / "ds_int" / "run_an"
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "models" / "run_an.pt").write_bytes(b"pt")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"task_type": "detection", "dataset": {"name": "ds_int"}}}),
        encoding="utf-8",
    )
    pred_dir = run_dir / "tests" / "test-onnx"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / "predictions_onnx.jsonl").write_text("{\"x\":1}\n{\"x\":2}\n", encoding="utf-8")

    rec = results_analyzer._build_run_record_unified(str(run_dir))
    # No test metrics artifact yet; canonical record should still resolve entity metadata.
    assert rec.model == "run_an"
    assert rec.dataset_name == "ds_int"

    from smartrain.run_model_contract.gateway import load_predictions

    preds = load_predictions(str(run_dir), source_kind="run", format_name="onnx")
    assert len(preds) == 1
    assert preds[0].count == 2

