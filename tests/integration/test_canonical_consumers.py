from __future__ import annotations

import json
from pathlib import Path

from smartrain.inference_cli import _resolve_model
from smartrain.workflows.testing.model_test_cli import _infer_task_from_training_metadata
from smartrain.workspace_paths import WorkspaceLayout, deploy_workspace


def test_canonical_consumers_infer_task_and_model_from_run(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv("SMARTTRAIN_CANONICAL_READ", "1")

    run_dir = tmp_path / "runs" / "ds_int" / "run_int"
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    w = run_dir / "models" / "run_int.onnx"
    w.write_bytes(b"onnx")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"task_type": "segmentation", "dataset": {"name": "ds_int"}}}),
        encoding="utf-8",
    )

    task = _infer_task_from_training_metadata(str(run_dir))
    assert task == "segment"

    import argparse

    args = argparse.Namespace(model_name=None, run=str(run_dir), weights=None)
    model_path, source_id, source_kind = _resolve_model(args, WorkspaceLayout(str(tmp_path)))
    assert source_kind == "runs"
    assert source_id == "run_int"
    assert model_path == w.resolve()

