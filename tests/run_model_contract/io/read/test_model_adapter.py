from __future__ import annotations

from pathlib import Path
import json

import pytest

from smartrain.unified.io.read.model_adapter import ModelAdapter


def test_model_adapter_reads_canonical_payload(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "demo_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "demo_model.pt").write_bytes(b"fake")
    (model_dir / "model_manifest.json").write_text(
        json.dumps({"task_type": "detection", "backend_type": "ultralytics"}),
        encoding="utf-8",
    )

    payload = ModelAdapter().read(str(model_dir))
    assert payload.models
    assert payload.models[0].model_id == "demo_model"
    assert payload.models[0].format == "pt"
    assert payload.models[0].task_type == "detection"
    assert payload.models[0].backend_type == "ultralytics"


def test_model_adapter_prefers_metadata_task_and_backend_when_available(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "seg_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "seg_model.onnx").write_bytes(b"fake")
    (model_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"task_type": "segmentation", "provider": {"id": "onnxruntime"}}}),
        encoding="utf-8",
    )

    payload = ModelAdapter().read(str(model_dir))
    model = payload.models[0]
    assert model.task_type == "segmentation"
    assert model.backend_type == "onnxruntime"
    assert model.provenance.get("task_resolution") == "metadata"
    assert model.provenance.get("backend_resolution") == "metadata"


def test_model_adapter_reads_task_from_ultralytics_train_block(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "plain_export"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "weights.pt").write_bytes(b"fake")
    (model_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"ultralytics_train": {"task": "segment"}}}),
        encoding="utf-8",
    )
    payload = ModelAdapter().read(str(model_dir))
    assert payload.models[0].task_type == "segmentation"
    assert payload.models[0].provenance.get("task_resolution") == "metadata"


def test_model_adapter_uses_name_and_format_hints_without_metadata(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "my-cls"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "my-cls.onnx").write_bytes(b"fake")

    payload = ModelAdapter().read(str(model_dir))
    model = payload.models[0]
    assert model.task_type == "classification"
    assert model.backend_type == "onnxruntime"
    assert model.provenance.get("task_resolution") == "name_hint"
    assert model.provenance.get("backend_resolution") == "format_hint"


def test_model_adapter_raises_without_task_provenance(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "plain"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "weights.pt").write_bytes(b"fake")

    with pytest.raises(ValueError, match="Cannot resolve task_type"):
        ModelAdapter().read(str(model_dir))

