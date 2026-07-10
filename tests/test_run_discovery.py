from __future__ import annotations

import json
from pathlib import Path

from smartrain.core.runtime.run_discovery import discover_analysis_targets
from smartrain.core.runtime.workspace_paths import deploy_workspace


def _write_run(workspace: Path, dataset: str, run_name: str) -> Path:
    run_dir = workspace / "runs" / dataset / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"model": "yolov8n", "dataset": {"name": dataset}}}),
        encoding="utf-8",
    )
    return run_dir


def _write_released_model(workspace: Path, dataset: str, stem: str) -> Path:
    models_dir = workspace / "models" / dataset
    models_dir.mkdir(parents=True, exist_ok=True)
    release_dir = models_dir / stem
    release_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{stem}.pt").write_bytes(b"pt")
    (release_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"model": "yolov8n", "dataset": {"name": dataset}}}),
        encoding="utf-8",
    )
    (release_dir / "test_metrics.csv").write_text("mAP50-95,Box-F1\n0.5,0.6\n", encoding="utf-8")
    return release_dir


def test_discover_analysis_targets_includes_runs_and_models(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = _write_run(tmp_path, "ds_a", "run_a")
    release_dir = _write_released_model(tmp_path, "ds_b", "detect_yolo_20260115")

    targets = discover_analysis_targets(workspace_cli=str(tmp_path), models_root_cli=None)
    assert str(run_dir.resolve()) in targets
    assert str(release_dir.resolve()) in targets


def test_discover_analysis_targets_respects_custom_models_root(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    _write_run(tmp_path, "ds_a", "run_a")
    release_dir = _write_released_model(tmp_path, "ds_b", "detect_yolo_20260115")
    custom_runs = tmp_path / "custom_runs" / "ds_a"
    custom_runs.mkdir(parents=True)
    custom_run = _write_run(tmp_path, "ds_a", "custom_run")
    custom_run.rename(custom_runs / "custom_run")

    targets = discover_analysis_targets(
        workspace_cli=str(tmp_path),
        models_root_cli=str(tmp_path / "custom_runs"),
    )
    assert str((custom_runs / "custom_run").resolve()) in targets
    assert str(release_dir.resolve()) not in targets
