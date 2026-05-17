from __future__ import annotations

import json
from pathlib import Path

from smartrain.run_model_contract.io.read.run_adapter import RunAdapter


def test_run_adapter_reads_canonical_payload(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds_a" / "run_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")
    (run_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_info": {
                    "task_type": "detection",
                    "dataset": {"name": "ds_a"},
                    "provider": {"id": "ultralytics"},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = RunAdapter().read(str(run_dir))
    assert payload.models
    assert payload.models[0].format == "pt"
    assert payload.runs[0].run_id == "run_1"


def test_run_adapter_infers_dataset_from_run_path_without_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds_path" / "run_2"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"fake")

    payload = RunAdapter().read(str(run_dir))
    assert payload.runs
    assert payload.runs[0].dataset_ref == "ds_path"


def test_run_adapter_infers_task_and_backend_from_model_artifact_without_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds_fmt" / "run_onnx_cls"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "models" / "best-cls.onnx").write_bytes(b"fake")

    payload = RunAdapter().read(str(run_dir))
    assert payload.models
    assert payload.models[0].format == "onnx"
    assert payload.models[0].task_type == "classification"
    assert payload.models[0].backend_type == "onnxruntime"

