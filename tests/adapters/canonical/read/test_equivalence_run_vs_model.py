from __future__ import annotations

import json
from pathlib import Path

from smartrain.adapters.canonical.read.model_adapter import ModelAdapter
from smartrain.adapters.canonical.read.run_adapter import RunAdapter


def test_equivalence_run_vs_model_for_same_pt_artifact(tmp_path: Path) -> None:
    pt_bytes = b"fake-weights"

    run_dir = tmp_path / "runs" / "ds_a" / "run_1"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(pt_bytes)
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"task_type": "detection"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    model_dir = tmp_path / "models" / "demo_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "demo_model.pt").write_bytes(pt_bytes)

    run_payload = RunAdapter().read(str(run_dir))
    model_payload = ModelAdapter().read(str(model_dir))
    assert run_payload.models[0].format == model_payload.models[0].format == "pt"
    assert run_payload.models[0].task_type == model_payload.models[0].task_type == "detection"

