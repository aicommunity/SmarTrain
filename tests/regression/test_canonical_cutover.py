from __future__ import annotations

from pathlib import Path

from smartrain.inference_cli import _resolve_model
from smartrain.workflows.testing.model_test_cli import _infer_task_from_training_metadata
from smartrain.results_analyzer import _canonical_read_enabled
from smartrain.canonical.policy import emit_legacy_read_deprecation_warnings
from smartrain.workspace_paths import WorkspaceLayout, deploy_workspace


def test_cutover_defaults_to_canonical_mode(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    emit_legacy_read_deprecation_warnings.cache_clear()
    monkeypatch.delenv("SMARTTRAIN_CANONICAL_READ", raising=False)
    monkeypatch.delenv("SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK", raising=False)
    assert _canonical_read_enabled() is True


def test_cutover_allows_emergency_legacy_fallback_only_when_explicit(monkeypatch, tmp_path: Path, capsys) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv("SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK", "1")
    monkeypatch.setenv("SMARTTRAIN_CANONICAL_READ", "0")
    emit_legacy_read_deprecation_warnings.cache_clear()

    assert _canonical_read_enabled() is True

    class _Ctx:
        task_type = "segmentation"
        run_id = "r1"

    monkeypatch.setattr("smartrain.orchestrators.canonical_gateway.resolve_task_context", lambda *_a, **_k: _Ctx())
    assert _infer_task_from_training_metadata(str(tmp_path)) == "segment"

    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best.onnx"
    model_path.write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")

    import argparse

    args = argparse.Namespace(model_name=None, run=str(run_dir), weights=None)
    resolved, _name, source = _resolve_model(args, WorkspaceLayout(str(tmp_path)))
    assert source == "runs"
    assert resolved == model_path.resolve()

    captured = capsys.readouterr()
    assert "[DEPRECATION]" in captured.err
    assert "SMARTTRAIN_CANONICAL_READ" in captured.err
    assert "SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK" in captured.err
