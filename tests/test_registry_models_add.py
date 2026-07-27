"""Tests for registry models-add full bundle promotion."""

from __future__ import annotations

import json
from pathlib import Path

from smartrain.workflows.registry.registry_cli import RegistryCliContext, _cmd_models_add
from smartrain.core.runtime.run_bundle_copy import copy_run_bundle, normalize_training_metadata_paths_for_bundle
from smartrain.core.runtime.workspace_paths import WorkspaceLayout, deploy_workspace


def test_normalize_training_metadata_paths_for_bundle_updates_paths(tmp_path: Path) -> None:
    meta = tmp_path / "training_metadata.json"
    meta.write_text(
        json.dumps(
            {
                "paths": {"best_model": "train/weights/best.pt"},
                "source": {"source_weights": "old.pt"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert normalize_training_metadata_paths_for_bundle(meta, "models/run1.pt") is True
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["paths"]["best_model"] == "models/run1.pt"
    assert payload["source"]["source_weights"] == "models/run1.pt"


def test_copy_run_bundle_respects_flags(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    dest = tmp_path / "dest"
    (run_root / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_root / "train" / "weights" / "huge.pt").write_bytes(b"w")
    (run_root / "train" / "args.yaml").write_text("x: 1", encoding="utf-8")
    (run_root / "models").mkdir(parents=True, exist_ok=True)
    (run_root / "models" / "m.pt").write_bytes(b"m")
    (run_root / "tests" / "t").mkdir(parents=True, exist_ok=True)
    (run_root / "tests" / "t" / "f.txt").write_text("z", encoding="utf-8")
    (run_root / "training_metadata.json").write_text("{}", encoding="utf-8")

    copy_run_bundle(run_root, dest, include_tests=False, copy_run_models=False)
    assert (dest / "train" / "args.yaml").is_file()
    assert not (dest / "train" / "weights").exists()
    assert not (dest / "models").exists()
    assert not (dest / "tests").exists()

    copy_run_bundle(run_root, tmp_path / "dest2", include_tests=True, copy_run_models=True)
    d2 = tmp_path / "dest2"
    assert (d2 / "models" / "m.pt").read_bytes() == b"m"
    assert (d2 / "tests" / "t" / "f.txt").read_text(encoding="utf-8") == "z"


def test_registry_models_add_copies_bundle_and_manifest(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "promo_run"
    run_dir.mkdir(parents=True)
    (run_dir / "models").mkdir(parents=True)
    (run_dir / "models" / "promo_run.pt").write_bytes(b"canonical-pt")
    (run_dir / "models" / "export.onnx").write_bytes(b"onnx-bytes")
    (run_dir / "train" / "weights").mkdir(parents=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"legacy")
    (run_dir / "train" / "results.csv").write_text("epoch,map\n", encoding="utf-8")
    (run_dir / "test").mkdir()
    (run_dir / "test" / "batch.json").write_text("{}", encoding="utf-8")
    (run_dir / "tests" / "probe").mkdir(parents=True)
    (run_dir / "tests" / "probe" / "x.txt").write_text("ok", encoding="utf-8")
    (run_dir / "test_metrics_extra.csv").write_text("m,a\n1,2\n", encoding="utf-8")

    md = {
        "training_info": {
            "model": "yolov8n",
            "task_type": "detect",
            "dataset": {"name": "ds1", "hash": "abcdef12"},
            "hyperparameters": {"image_size": 640},
        },
        "timestamps": {"training": {"end": "2026-01-15T12:00:00+00:00", "start": None}},
        "paths": {"best_model": "models/promo_run.pt"},
        "source": {"source_weights": "models/promo_run.pt"},
    }
    (run_dir / "training_metadata.json").write_text(json.dumps(md, ensure_ascii=False), encoding="utf-8")

    ctx = RegistryCliContext(WorkspaceLayout(str(tmp_path)))
    _cmd_models_add(ctx, str(run_dir))

    models_root = tmp_path / "models"
    promoted_dirs = [p for p in models_root.iterdir() if p.is_dir() and (p / "model_manifest.json").is_file()]
    assert len(promoted_dirs) == 1
    promo = promoted_dirs[0]
    man = json.loads((promo / "model_manifest.json").read_text(encoding="utf-8"))
    assert man["weights_file"] == "models/promo_run.pt"
    assert man.get("bundle_layout_version") == 2
    assert man.get("task_type") == "detection"
    assert man.get("workspace_root") == "."
    assert man.get("source_run_relative") == man.get("source_run")
    assert man.get("source_run") == "runs/promo_run"
    assert "\\" not in str(man.get("source_run") or "")
    assert not str(man.get("source_run") or "").startswith("/")
    assert ":" not in str(man.get("source_run") or "")
    assert (promo / "models" / "promo_run.pt").read_bytes() == b"canonical-pt"
    assert (promo / "models" / "export.onnx").read_bytes() == b"onnx-bytes"
    # Source run may migrate ``train/`` → ``train-ultralytics/`` when canonical path is resolved.
    assert (promo / "train-ultralytics" / "results.csv").is_file() or (promo / "train" / "results.csv").is_file()
    assert not (promo / "train-ultralytics" / "weights").exists()
    assert not (promo / "train" / "weights").exists()
    assert (promo / "tests" / "test-ultralytics" / "batch.json").is_file() or (promo / "test" / "batch.json").is_file()
    assert (promo / "tests" / "probe" / "x.txt").read_text(encoding="utf-8") == "ok"
    tm = promo / "test_metrics_extra.csv"
    if not tm.is_file():
        tm = promo / "tests" / "test_metrics_extra.csv"
    assert tm.is_file()
    assert tm.read_text(encoding="utf-8").startswith("m,a")

    meta_out = json.loads((promo / "training_metadata.json").read_text(encoding="utf-8"))
    assert meta_out["paths"]["best_model"] == "models/promo_run.pt"
