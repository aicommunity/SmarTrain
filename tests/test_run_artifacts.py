from __future__ import annotations

import json
from pathlib import Path

import pytest

from smartrain.core.runtime.run_artifacts import (
    preferred_run_model_path,
    normalize_ultralytics_run_layout,
    consolidate_train_backend_dir,
    ensure_run_layout,
    materialize_preferred_run_model,
    reject_documentation_placeholder_path,
    relocate_or_remove_legacy_val_recs_at_run_root,
    resolve_run_model,
    run_train_backend_dir,
    run_tests_dir,
)
from smartrain.core.runtime.ultralytics_ephemeral import prune_empty_sidecar_dirs


def test_reject_documentation_placeholder_path_ellipsis() -> None:
    with pytest.raises(ValueError, match="ellipsis placeholder"):
        reject_documentation_placeholder_path("...", kind="run_dir")
    with pytest.raises(ValueError, match="ellipsis placeholder"):
        reject_documentation_placeholder_path("/tmp/runs/.../run-1", kind="run_dir")


def test_ensure_run_layout_rejects_placeholder_run_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ellipsis placeholder"):
        ensure_run_layout(str(tmp_path / "..."))


def test_resolve_run_model_finds_sibling_pt_for_release_bundle(tmp_path: Path) -> None:
    release_dir = tmp_path / "models" / "ds1" / "detect_yolo_20260115"
    release_dir.mkdir(parents=True, exist_ok=True)
    sibling = release_dir.parent / "detect_yolo_20260115.pt"
    sibling.write_bytes(b"released")

    resolved = resolve_run_model(str(release_dir), ".pt")
    assert resolved == sibling


def test_resolve_run_model_prefers_canonical(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds1" / "run-1"
    (run_dir / "train-ultralytics" / "weights").mkdir(parents=True, exist_ok=True)
    legacy = run_dir / "train-ultralytics" / "weights" / "best.pt"
    legacy.write_bytes(b"legacy")
    canonical = Path(preferred_run_model_path(str(run_dir), ".pt"))
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"canonical")

    resolved = resolve_run_model(str(run_dir), ".pt")
    assert resolved == canonical


def test_materialize_preferred_run_model_moves_legacy_and_normalizes_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds1" / "run-1"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    legacy = run_dir / "train" / "weights" / "best.pt"
    legacy.write_bytes(b"legacy")
    meta_path = run_dir / "training_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "paths": {"best_model": "train/weights/best.pt"},
                "source": {"source_weights": "train/weights/best.pt"},
            }
        ),
        encoding="utf-8",
    )

    canonical = materialize_preferred_run_model(str(run_dir), ext=".pt", move=True, normalize_metadata=True)
    assert canonical == Path(preferred_run_model_path(str(run_dir), ".pt"))
    assert canonical.is_file()
    assert not legacy.exists()

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["paths"]["best_model"] == "run-1.pt"
    assert payload["source"]["source_weights"] == "run-1.pt"


def test_ensure_run_layout_migrates_legacy_test_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds1" / "run-legacy"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "test").mkdir(parents=True, exist_ok=True)
    (run_dir / "test_onnx").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "args.yaml").write_text("epochs: 1\n", encoding="utf-8")
    (run_dir / "train" / "weights" / "best.pt").write_text("legacy-best", encoding="utf-8")
    (run_dir / "test" / "pr.csv").write_text("legacy-pt", encoding="utf-8")
    (run_dir / "test_onnx" / "pr.csv").write_text("legacy-onnx", encoding="utf-8")
    (run_dir / "test_metrics.csv").write_text("m", encoding="utf-8")
    (run_dir / "confidence_recommendations_test.json").write_text("{}", encoding="utf-8")
    (run_dir / "test_artifacts_manifest.json").write_text('{"formats":{}}', encoding="utf-8")

    ensure_run_layout(str(run_dir))
    tests_root = run_tests_dir(str(run_dir))
    train_root = run_train_backend_dir(str(run_dir), "ultralytics")

    assert (tests_root / "test-ultralytics" / "pr.csv").read_text(encoding="utf-8") == "legacy-pt"
    assert (tests_root / "test_onnx" / "pr.csv").read_text(encoding="utf-8") == "legacy-onnx"
    assert (tests_root / "test_metrics.csv").is_file()
    assert (tests_root / "confidence_recommendations_test.json").is_file()
    assert (tests_root / "test_artifacts_manifest.json").is_file()
    assert (train_root / "args.yaml").is_file()
    assert (train_root / "weights" / "best.pt").is_file()
    assert not (run_dir / "train").exists()
    assert not (run_dir / "test").exists()
    assert not (run_dir / "test_onnx").exists()
    assert not (run_dir / "test_metrics.csv").exists()
    assert not (run_dir / "confidence_recommendations_test.json").exists()
    assert not (run_dir / "test_artifacts_manifest.json").exists()


def test_ensure_run_layout_migration_does_not_overwrite_new_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds1" / "run-conflict"
    tests_root = run_tests_dir(str(run_dir))
    (run_dir / "test").mkdir(parents=True, exist_ok=True)
    (tests_root / "test-ultralytics").mkdir(parents=True, exist_ok=True)
    (run_dir / "test" / "pr.csv").write_text("legacy", encoding="utf-8")
    (tests_root / "test-ultralytics" / "pr.csv").write_text("new", encoding="utf-8")
    (run_dir / "test_metrics.csv").write_text("legacy-metrics", encoding="utf-8")
    (tests_root / "test_metrics.csv").write_text("new-metrics", encoding="utf-8")

    ensure_run_layout(str(run_dir))
    ensure_run_layout(str(run_dir))

    assert (tests_root / "test-ultralytics" / "pr.csv").read_text(encoding="utf-8") == "new"
    assert (tests_root / "test_metrics.csv").read_text(encoding="utf-8") == "new-metrics"
    assert not (run_dir / "test" / "pr.csv").exists()
    assert not (run_dir / "test_metrics.csv").exists()


def test_ensure_run_layout_merges_parallel_test_ultralytics_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds1" / "run-parallel-ultra"
    parallel = run_dir / "test-ultralytics"
    parallel.mkdir(parents=True, exist_ok=True)
    (parallel / "BoxPR_curve.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01")

    ensure_run_layout(str(run_dir))
    dest = run_tests_dir(str(run_dir)) / "test-ultralytics"
    assert (dest / "BoxPR_curve.png").is_file()
    assert not parallel.exists()


def test_ensure_run_layout_no_empty_train_dir_without_legacy(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds1" / "run-fresh"
    run_dir.mkdir(parents=True)
    ensure_run_layout(str(run_dir))
    train_root = run_train_backend_dir(str(run_dir), "ultralytics")
    assert not train_root.exists() or not any(train_root.iterdir())


def test_consolidate_train_merges_suffix(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds1" / "run-train-suffix"
    run_dir.mkdir(parents=True)
    (run_dir / "train-ultralytics").mkdir()
    suffix = run_dir / "train-ultralytics-2"
    (suffix / "weights").mkdir(parents=True)
    (suffix / "results.csv").write_text("epoch,mAP\n1,0.5\n", encoding="utf-8")
    (suffix / "weights" / "best.pt").write_bytes(b"best")

    consolidate_train_backend_dir(str(run_dir))
    train_root = run_train_backend_dir(str(run_dir), "ultralytics")
    assert (train_root / "results.csv").is_file()
    assert (train_root / "weights" / "best.pt").is_file()
    assert not suffix.exists()


def test_consolidate_test_drops_empty_suffix(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds1" / "run-test-suffix"
    tests = run_dir / "tests"
    tests.mkdir(parents=True)
    (tests / "test-ultralytics").mkdir()
    (tests / "test-ultralytics" / "pr.csv").write_text("recall,precision\n0.5,0.6\n", encoding="utf-8")
    (tests / "test-ultralytics2").mkdir()

    normalize_ultralytics_run_layout(str(run_dir))
    assert (tests / "test-ultralytics" / "pr.csv").is_file()
    assert not (tests / "test-ultralytics2").exists()


def test_relocate_val_recs_root_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds1" / "run-val-recs"
    run_dir.mkdir(parents=True)
    (run_dir / "val-recs-pt").mkdir()
    (run_dir / "val-recs-pt_uni").mkdir()

    relocate_or_remove_legacy_val_recs_at_run_root(str(run_dir))
    assert not (run_dir / "val-recs-pt").exists()
    assert not (run_dir / "val-recs-pt_uni").exists()


def test_relocate_val_recs_root_with_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds1" / "run-val-recs-files"
    run_dir.mkdir(parents=True)
    (run_dir / "val-recs-pt").mkdir()
    (run_dir / "val-recs-pt" / "args.yaml").write_text("split: val\n", encoding="utf-8")

    relocate_or_remove_legacy_val_recs_at_run_root(str(run_dir))
    dest = run_tests_dir(str(run_dir)) / "val-recs-pt" / "args.yaml"
    assert dest.is_file()
    assert not (run_dir / "val-recs-pt").exists()


def test_prune_empty_keeps_models_pt(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds1" / "run-prune"
    (run_dir / "models").mkdir(parents=True)
    (run_dir / "models" / "run-prune.pt").write_bytes(b"weights")
    tests = run_dir / "tests"
    tests.mkdir()
    (tests / "test-ultralytics2").mkdir()

    normalize_ultralytics_run_layout(str(run_dir))
    assert (run_dir / "models" / "run-prune.pt").is_file()
    assert not (tests / "test-ultralytics2").exists()


def test_canonicalize_idempotent(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds1" / "run-idempotent"
    run_dir.mkdir(parents=True)
    (run_dir / "val-recs-pt").mkdir()
    normalize_ultralytics_run_layout(str(run_dir))
    normalize_ultralytics_run_layout(str(run_dir))
    assert not (run_dir / "val-recs-pt").exists()


def test_prune_empty_sidecar_dirs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds1" / "run-scratch"
    run_dir.mkdir(parents=True)
    (run_dir / ".ultralytics_scratch").mkdir()
    (run_dir / ".ultralytics_predict_scratch").mkdir()
    prune_empty_sidecar_dirs(str(run_dir))
    assert not (run_dir / ".ultralytics_scratch").exists()
    assert not (run_dir / ".ultralytics_predict_scratch").exists()

