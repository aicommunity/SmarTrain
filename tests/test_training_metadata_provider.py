from __future__ import annotations

import json
from pathlib import Path

from smartrain.model_training_module import save_training_metadata


def test_training_metadata_contains_provider_block(tmp_path: Path) -> None:
    model_dir = tmp_path / "run"
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = tmp_path / "datasets" / "ds_a"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_training_metadata(
        model_dir=str(model_dir),
        dataset_path=str(dataset_dir),
        model_version="yolov8n",
        training_success=True,
        test_success=True,
        training_provider="ultralytics",
    )
    payload = json.loads((model_dir / "training_metadata.json").read_text(encoding="utf-8"))
    assert payload["training_info"]["provider"]["type"] == "builtin"
    assert payload["training_info"]["provider"]["id"] == "ultralytics"
