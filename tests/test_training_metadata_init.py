from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from smartrain import model_training_module as mtm


def test_ensure_initial_training_metadata_creates_skeleton(tmp_path: Path) -> None:
    model_dir = tmp_path / "runs" / "ds_a" / "run1"
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = tmp_path / "datasets" / "ds_a"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    mtm._ensure_initial_training_metadata(
        model_dir=str(model_dir),
        dataset_path=str(dataset_dir),
        model_version="yolo11x",
        epochs=300,
        batch=4,
        img_size=1280,
        training_start_time=datetime(2026, 1, 1, 12, 0, 0),
        dataset_hash="abcd1234",
        workspace_root=str(tmp_path),
        task_type="detection",
    )

    payload = json.loads((model_dir / "training_metadata.json").read_text(encoding="utf-8"))
    assert payload["training_info"]["model"] == "yolo11x"
    assert payload["training_info"]["dataset"]["name"] == "ds_a"
    assert payload["status"]["training"]["success"] is None
    assert payload["timestamps"]["training"]["start"] == "2026-01-01T12:00:00"


def test_ensure_initial_training_metadata_does_not_overwrite_existing_core_fields(tmp_path: Path) -> None:
    model_dir = tmp_path / "runs" / "ds_b" / "run2"
    model_dir.mkdir(parents=True, exist_ok=True)
    meta_path = model_dir / "training_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "training_info": {
                    "model": "yolo26x",
                    "dataset": {"name": "existing_ds"},
                },
                "status": {"training": {"success": False}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    mtm._ensure_initial_training_metadata(
        model_dir=str(model_dir),
        dataset_path=str(tmp_path / "datasets" / "ds_b"),
        model_version="yolo11x",
        epochs=100,
        batch=8,
        img_size=640,
        training_start_time=datetime(2026, 1, 1, 12, 0, 0),
        dataset_hash=None,
        workspace_root=str(tmp_path),
        task_type="detection",
    )

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["training_info"]["model"] == "yolo26x"
    assert payload["training_info"]["dataset"]["name"] == "existing_ds"
