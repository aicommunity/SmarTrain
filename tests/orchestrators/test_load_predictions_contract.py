from __future__ import annotations

import json
from pathlib import Path

from smartrain.orchestrators.canonical_gateway import CanonicalGatewayOptions, load_predictions


def _minimal_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "ds" / "run_pred"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": "ds"}, "task_type": "detection"}}),
        encoding="utf-8",
    )
    (run_dir / "noise_prediction.json").write_text("[]", encoding="utf-8")
    (run_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    return run_dir


def test_load_predictions_strict_skips_pred_glob_files(tmp_path: Path) -> None:
    run_dir = _minimal_run(tmp_path)
    loose = load_predictions(str(run_dir), options=CanonicalGatewayOptions(validate=False, predictions_strict=False))
    strict = load_predictions(str(run_dir), options=CanonicalGatewayOptions(validate=False, predictions_strict=True))
    loose_names = {Path(p.items_path).name for p in loose}
    strict_names = {Path(p.items_path).name for p in strict}
    assert "noise_prediction.json" in loose_names
    assert "noise_prediction.json" not in strict_names
    assert "predictions.jsonl" in strict_names


def test_load_predictions_finds_deep_diagnostics_jsonl(tmp_path: Path) -> None:
    run_dir = _minimal_run(tmp_path)
    dd = run_dir / "deep_diagnostics"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "debug_test.jsonl").write_text("{}\n", encoding="utf-8")
    strict = load_predictions(str(run_dir), options=CanonicalGatewayOptions(validate=False, predictions_strict=True))
    paths = {p.items_path for p in strict}
    assert any(str(p).endswith("deep_diagnostics/debug_test.jsonl") for p in paths)
