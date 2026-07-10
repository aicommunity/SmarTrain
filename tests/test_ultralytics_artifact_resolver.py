"""Tests for Ultralytics artifact resolver and analyze collector."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from smartrain.core.testing.ultralytics_artifact_resolver import (
    PROVENANCE_TRAIN_VAL,
    resolve_ultralytics_artifacts,
)
from smartrain.services.analyze.ultralytics_test_artifacts import (
    build_ultralytics_run_info,
    collect_ultralytics_test_artifacts,
)


def _write_train_val_plots(run_dir: Path) -> None:
    train = run_dir / "train-ultralytics"
    train.mkdir(parents=True, exist_ok=True)
    (train / "BoxPR_curve.png").write_bytes(b"png")
    (train / "BoxF1_curve.png").write_bytes(b"png")
    (train / "confusion_matrix.png").write_bytes(b"png")
    (train / "args.yaml").write_text("epochs: 400\nbatch: 16\nimgsz: 640\n", encoding="utf-8")


def test_resolver_finds_train_val_fallback_when_test_dir_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run_a"
    run_dir.mkdir(parents=True)
    _write_train_val_plots(run_dir)
    resolved = resolve_ultralytics_artifacts(str(run_dir))
    assert resolved.completeness == "train_val_fallback"
    assert "BoxPR_curve.png" in resolved.resolved
    assert resolved.resolved["BoxPR_curve.png"][1] == PROVENANCE_TRAIN_VAL


def test_resolver_prefers_test_dir_over_train_val(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run_b"
    test_dir = run_dir / "tests" / "test-ultralytics"
    test_dir.mkdir(parents=True)
    _write_train_val_plots(run_dir)
    (test_dir / "BoxPR_curve.png").write_bytes(b"test-png")
    (test_dir / "pr.csv").write_text("recall,precision\n0,0\n", encoding="utf-8")
    (test_dir / "pr_per_class.csv").write_text("class_name,ap\na,0.5\n", encoding="utf-8")
    (test_dir / "args.yaml").write_text("epochs: 1\n", encoding="utf-8")
    for name in ("BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png", "confusion_matrix.png", "confusion_matrix_normalized.png"):
        (test_dir / name).write_bytes(b"x")
    resolved = resolve_ultralytics_artifacts(str(run_dir))
    assert resolved.resolved["BoxPR_curve.png"][0].endswith("tests/test-ultralytics/BoxPR_curve.png")


def test_build_ultralytics_run_info_from_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_info": {
                    "model": "yolo11n",
                    "dataset": {"name": "ds1"},
                    "hyperparameters": {"epochs": 400, "batch_size": 16, "image_size": 640},
                },
                "system_profile": {
                    "cpu": {"model": "CPU-X", "logical_cores": 8},
                    "platform": {"os": "Linux", "os_release": "6.8"},
                },
            }
        ),
        encoding="utf-8",
    )
    info = build_ultralytics_run_info(str(run_dir))
    assert info["epochs"] == 400
    assert info["batch_size"] == 16
    assert info["train_image_size"] == 640


def test_collect_copies_train_val_images_to_session(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run_c"
    run_dir.mkdir(parents=True)
    _write_train_val_plots(run_dir)
    test_dir = run_dir / "tests" / "test-ultralytics"
    test_dir.mkdir(parents=True)
    pd.DataFrame({"recall": [0.0], "precision": [0.0]}).to_csv(test_dir / "pr.csv", index=False)
    pd.DataFrame({"class_name": ["a"], "ap": [0.9]}).to_csv(test_dir / "pr_per_class.csv", index=False)

    class _Rec:
        model = "run_c"
        dataset_name = "ds"

    rows, arts = collect_ultralytics_test_artifacts(
        str(tmp_path / "session"),
        [str(run_dir)],
        {"run_c": "R1"},
        run_test_backend_dir_cb=lambda rd, _b: str(Path(rd) / "tests" / "test-ultralytics"),
        build_run_record_unified_cb=lambda _rd: _Rec(),
    )
    assert len(rows) == 1
    assert rows[0]["completeness"] == "train_val_fallback"
    assert rows[0]["run_info"]["epochs"] == 400
    assert rows[0]["images"]
    assert any(a["path"].endswith("BoxPR_curve.png") for a in arts)


def test_uralkaliy_layout_fixture(tmp_path: Path) -> None:
    """Regression: partial test-ultralytics + rich train-ultralytics (Uralkaliy layout)."""
    run_dir = tmp_path / "models" / "ds" / "detect_yolo_20260704"
    run_dir.mkdir(parents=True)
    _write_train_val_plots(run_dir)
    partial = run_dir / "tests" / "test-ultralytics"
    partial.mkdir(parents=True)
    pd.DataFrame({"recall": [0.1, 0.2], "precision": [0.9, 0.8]}).to_csv(partial / "pr.csv", index=False)
    pd.DataFrame(
        [{"class_name": "construct", "ap": 0.5}, {"class_name": "digits", "ap": 0.7}]
    ).to_csv(partial / "pr_per_class.csv", index=False)

    resolved = resolve_ultralytics_artifacts(str(run_dir))
    assert resolved.completeness in {"train_val_fallback", "partial_csv_only"}
    assert "BoxPR_curve.png" in resolved.resolved
