from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from smartrain.services.train_runtime_helpers import (
    build_run_name,
    finalize_run_dir_naming,
    format_batch_token,
    read_effective_ultralytics_train_hyperparams,
)


def test_build_run_name_includes_img_size_and_batch_token() -> None:
    fixed = datetime(2026, 7, 4, 10, 34)
    name = build_run_name("ultralytics", "yolo11m.pt", 200, 16, "c78211ca", img_size=640, timestamp=fixed)
    assert name == "2026-07-04_10-34_ultralytics_yolo11m_640px_200epochs_b16-c78211ca"


def test_format_batch_token_fractional() -> None:
    assert format_batch_token(16) == "b16"
    assert format_batch_token(0.5) == "b0p5"


def test_read_effective_ultralytics_train_hyperparams(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    train_backend = run_dir / "train-ultralytics"
    train_backend.mkdir(parents=True)
    (train_backend / "args.yaml").write_text(
        "epochs: 50\nbatch: 8\nimgsz: [1024, 1024]\n",
        encoding="utf-8",
    )
    effective = read_effective_ultralytics_train_hyperparams(str(run_dir))
    assert effective == {"epochs": 50, "batch": 8, "img_size": 1024}


def test_finalize_run_dir_naming_renames_dir_models_and_metadata(tmp_path: Path) -> None:
    from smartrain.services.train_runtime_helpers import build_model_weights_stem

    fixed = datetime(2026, 7, 4, 10, 34)
    old_name = build_run_name("ultralytics", "yolo11m.pt", 200, 16, "c78211ca", img_size=640, timestamp=fixed)
    new_name = build_run_name("ultralytics", "yolo11m.pt", 200, 8, "c78211ca", img_size=640, timestamp=fixed)
    weights_stem = build_model_weights_stem(
        "detect", "yolo11m.pt", 200, 8, 640, timestamp=fixed
    )
    assert old_name != new_name

    run_dir = tmp_path / "ds_a" / old_name
    models_dir = run_dir / "models"
    train_backend = run_dir / "train-ultralytics"
    train_backend.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    (train_backend / "args.yaml").write_text(
        "epochs: 200\nbatch: 8\nimgsz: 640\n",
        encoding="utf-8",
    )
    (train_backend / "weights").mkdir(parents=True)
    (train_backend / "weights" / "best.pt").write_bytes(b"weights")
    (models_dir / f"{old_name}.pt").write_bytes(b"weights")
    (run_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "paths": {"best_model": f"{old_name}.pt"},
                "training_info": {
                    "task_type": "detect",
                    "model": "yolo11m",
                    "hyperparameters": {"epochs": 200, "batch_size": 16, "image_size": 640},
                },
            }
        ),
        encoding="utf-8",
    )

    new_dir, effective = finalize_run_dir_naming(
        str(run_dir),
        provider_id="ultralytics",
        model_version="yolo11m.pt",
        dataset_hash="c78211ca",
        training_start_time=fixed,
        task_type="detect",
    )

    assert effective["batch"] == 8
    assert Path(new_dir).name == new_name
    assert (tmp_path / "ds_a" / new_name / "models" / f"{weights_stem}.pt").is_file()
    assert not (tmp_path / "ds_a" / new_name / "models" / f"{new_name}.pt").exists()
    meta = json.loads((tmp_path / "ds_a" / new_name / "training_metadata.json").read_text(encoding="utf-8"))
    assert meta["paths"]["best_model"] == f"{weights_stem}.pt"
    assert meta["training_info"]["hyperparameters"]["batch_size"] == 8
