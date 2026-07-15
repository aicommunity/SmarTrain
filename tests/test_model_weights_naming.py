"""Tests for detect_* model weight file stem naming."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from smartrain.core.runtime.run_artifacts import (
    materialize_preferred_run_model,
    preferred_run_model_path,
    resolve_run_model,
)
from smartrain.core.runtime.workspace_paths import deploy_workspace
from smartrain.services.models.release_model_naming import (
    build_model_weights_stem,
    build_model_weights_stem_from_metadata,
)
from smartrain.services.models.release_models_manifest import (
    find_release_pt_in_dir,
    is_nested_release_layout,
    is_workspace_release_bundle,
)


def test_build_model_weights_stem_matches_expected_example() -> None:
    stem = build_model_weights_stem(
        "detection",
        "yolo11s.pt",
        400,
        16,
        640,
        timestamp=datetime(2026, 7, 6, 3, 2, 58, tzinfo=timezone.utc),
    )
    assert stem == "detect_yolo11s_20260706_030258_640px_400epochs_b16"


def test_preferred_run_model_path_uses_weights_stem_from_metadata(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    folder = "2026-07-14_20-42_ultralytics_yolo11s_640px_400epochs_b16-b1ef93cc"
    run_dir = tmp_path / "runs" / "my_ds" / folder
    models = run_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    weights_stem = "detect_yolo11s_20260706_030258_640px_400epochs_b16"
    (models / f"{weights_stem}.pt").write_bytes(b"pt")
    (run_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "paths": {"best_model": f"{weights_stem}.pt"},
                "training_info": {
                    "model": "yolo11s",
                    "task_type": "detect",
                    "hyperparameters": {"epochs": 400, "batch_size": 16, "image_size": 640},
                },
                "timestamps": {"training": {"start": "2026-07-06T03:02:58+00:00"}},
            }
        ),
        encoding="utf-8",
    )
    preferred = Path(preferred_run_model_path(str(run_dir), ".pt"))
    assert preferred.name == f"{weights_stem}.pt"
    assert preferred.parent == models
    assert resolve_run_model(str(run_dir), ".pt") == preferred


def test_resolve_run_model_falls_back_to_legacy_folder_name(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    folder = "2026-07-14_20-42_ultralytics_yolo11s_640px_400epochs_b16-b1ef93cc"
    run_dir = tmp_path / "runs" / "my_ds" / folder
    models = run_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    legacy = models / f"{folder}.pt"
    legacy.write_bytes(b"legacy")
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")
    assert resolve_run_model(str(run_dir), ".pt") == legacy
    assert Path(preferred_run_model_path(str(run_dir), ".pt")).name == f"{folder}.pt"


def test_materialize_moves_legacy_folder_pt_to_weights_stem(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    folder = "run-folder"
    run_dir = tmp_path / "runs" / "ds1" / folder
    models = run_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    legacy = models / f"{folder}.pt"
    legacy.write_bytes(b"weights")
    weights_stem = "detect_yolo11s_20260706_030258_640px_400epochs_b16"
    (run_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "paths": {"best_model": f"{weights_stem}.pt"},
                "training_info": {
                    "model": "yolo11s",
                    "task_type": "detect",
                    "hyperparameters": {"epochs": 400, "batch_size": 16, "image_size": 640},
                },
                "timestamps": {"training": {"start": "2026-07-06T03:02:58+00:00"}},
            }
        ),
        encoding="utf-8",
    )
    materialized = materialize_preferred_run_model(str(run_dir), ext=".pt", move=True)
    assert materialized is not None
    assert Path(materialized).name == f"{weights_stem}.pt"
    assert Path(materialized).is_file()
    assert not legacy.exists()


def test_nested_release_helpers_when_stem_differs_from_folder(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    folder = "2026-07-14_20-42_ultralytics_yolo11s_640px_400epochs_b16-hash"
    stem = "detect_yolo11s_20260706_030258_640px_400epochs_b16"
    release_dir = tmp_path / "models" / "ds1" / folder
    release_dir.mkdir(parents=True, exist_ok=True)
    pt = release_dir / f"{stem}.pt"
    pt.write_bytes(b"pt")
    (release_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "source": {"source_run": "/x"},
                "artifacts": {"release_dir": str(release_dir), "model_path": str(pt)},
            }
        ),
        encoding="utf-8",
    )
    assert is_nested_release_layout(pt)
    assert find_release_pt_in_dir(release_dir) == pt
    assert is_workspace_release_bundle(pt)
    assert is_workspace_release_bundle(release_dir)


def test_build_model_weights_stem_from_metadata() -> None:
    md = {
        "training_info": {
            "model": "yolo11s",
            "task_type": "detect",
            "hyperparameters": {"epochs": 400, "batch_size": 16, "image_size": 640},
        },
        "timestamps": {"training": {"start": "2026-07-06T03:02:58"}},
    }
    assert (
        build_model_weights_stem_from_metadata(md)
        == "detect_yolo11s_20260706_030258_640px_400epochs_b16"
    )
