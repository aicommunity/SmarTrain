from __future__ import annotations

import json
from pathlib import Path

from smartrain.run_artifacts import materialize_canonical_run_model, resolve_run_model_with_legacy_fallback


def test_resolve_run_model_with_legacy_fallback_prefers_canonical(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds1" / "run-1"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    canonical = run_dir / "run-1.pt"
    legacy = run_dir / "train" / "weights" / "best.pt"
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
    assert canonical == (run_dir / "run-1.pt")
    assert canonical.is_file()
    assert not legacy.exists()

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["paths"]["best_model"] == "run-1.pt"
    assert payload["source"]["source_weights"] == "run-1.pt"
