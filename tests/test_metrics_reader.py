from __future__ import annotations

import json
from pathlib import Path

from smartrain.metrics_reader import flatten_metadata


def test_flatten_metadata_falls_back_to_args_yaml_model(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "dataset_a" / "run1"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "args.yaml").write_text(
        "model: /old/machine/runs/weights/yolo26x.pt\n", encoding="utf-8"
    )
    md = {"status": {"training": {"success": True}}}

    row = flatten_metadata(md, str(run_dir))
    assert row["model"] == "yolo26x"


def test_flatten_metadata_falls_back_to_parent_dir_dataset_name(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "dataset_b" / "run2"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"status": {"training": {"success": True}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    md = {"training_info": {"model": "yolo11x"}, "status": {"training": {"success": True}}}

    row = flatten_metadata(md, str(run_dir))
    assert row["dataset_name"] == "dataset_b"
