from __future__ import annotations

from pathlib import Path

from smartrain.services.analyze.confidence_ensure import _resolve_pt_weights_for_confidence


def test_confidence_resolve_pt_finds_legacy_train_weights(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run_a"
    legacy = run_dir / "train" / "weights" / "best.pt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"pt-bytes")

    found = _resolve_pt_weights_for_confidence(str(run_dir))
    assert found is not None
    assert Path(found).is_file()
    # Prefer canonical models/<stem>.pt after materialize
    assert Path(found).parent.name == "models"


def test_confidence_resolve_pt_missing_returns_none(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run_empty"
    run_dir.mkdir(parents=True)
    assert _resolve_pt_weights_for_confidence(str(run_dir)) is None
