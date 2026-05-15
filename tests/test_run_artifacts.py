from __future__ import annotations

import json
from pathlib import Path

from smartrain.core.runtime.run_artifacts import (
    canonical_run_model_path,
    ensure_run_layout,
    materialize_canonical_run_model,
    resolve_run_model_with_legacy_fallback,
    run_train_backend_dir,
    run_tests_dir,
)


def test_resolve_run_model_with_legacy_fallback_prefers_canonical(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds1" / "run-1"
    (run_dir / "train-ultralytics" / "weights").mkdir(parents=True, exist_ok=True)
    canonical = Path(canonical_run_model_path(str(run_dir), ".pt"))
    legacy = run_dir / "train-ultralytics" / "weights" / "best.pt"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"canonical")
    legacy.write_bytes(b"legacy")

    resolved = resolve_run_model_with_legacy_fallback(str(run_dir), ".pt")
    assert resolved == canonical


def test_materialize_canonical_run_model_moves_legacy_and_normalizes_metadata(tmp_path: Path) -> None:
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

    canonical = materialize_canonical_run_model(str(run_dir), ext=".pt", move=True, normalize_metadata=True)
    assert canonical == Path(canonical_run_model_path(str(run_dir), ".pt"))
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

