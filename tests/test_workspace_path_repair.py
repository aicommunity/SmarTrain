from __future__ import annotations

import json
from pathlib import Path

from smartrain.workspace_path_repair import repair_workspace_paths


def test_repair_rewrites_absolute_data_path_in_datasets_info(tmp_path: Path) -> None:
    ws = tmp_path.resolve()
    ds = ws / "datasets"
    ds.mkdir(parents=True)
    d1 = ds / "d1"
    d1.mkdir()
    info = {"d1": {"data_path": str(d1.resolve()), "names": ["a"]}}
    (ds / "datasets_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    rep = repair_workspace_paths(str(ws), dry_run=False)
    assert rep.datasets_info_updated == 1
    loaded = json.loads((ds / "datasets_info.json").read_text(encoding="utf-8"))
    assert loaded["d1"]["data_path"] == "datasets/d1"


def test_repair_training_metadata_workspace_root_nameerror_fixed(tmp_path: Path) -> None:
    """Regression: _repair_training_metadata_file must define wr before comparing workspace.root."""
    ws = tmp_path.resolve()
    (ws / "datasets").mkdir(parents=True)
    (ws / "datasets" / "datasets_info.json").write_text("{}", encoding="utf-8")
    runs = ws / "runs" / "myds" / "exp1"
    runs.mkdir(parents=True)
    ds_abs = str((ws / "datasets" / "myds").resolve())
    meta = {
        "workspace": {
            "root": str(ws),
            "dataset_path_relative": "datasets/myds",
            "run_directory_relative": "runs/myds/exp1",
        },
        "training_info": {
            "dataset": {
                "name": "myds",
                "path_absolute": ds_abs,
                "path_relative": ".",
                "hash": None,
            },
            "hyperparameters": {},
        },
    }
    meta_path = runs / "training_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    rep = repair_workspace_paths(str(ws), dry_run=False)
    assert rep.training_metadata_updated == 1
    loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    assert loaded["workspace"]["root"] == "."
    assert loaded["training_info"]["dataset"]["path_under_workspace"] == "datasets/myds"
    assert "path_absolute" not in loaded["training_info"]["dataset"]


def test_repair_dry_run_does_not_write_datasets_info(tmp_path: Path) -> None:
    ws = tmp_path.resolve()
    ds = ws / "datasets"
    ds.mkdir(parents=True)
    d1 = ds / "d2"
    d1.mkdir()
    raw = str(d1.resolve())
    info = {"d2": {"data_path": raw}}
    text = json.dumps(info, indent=2)
    (ds / "datasets_info.json").write_text(text, encoding="utf-8")
    repair_workspace_paths(str(ws), dry_run=True)
    assert (ds / "datasets_info.json").read_text(encoding="utf-8") == text
