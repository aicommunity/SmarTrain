from __future__ import annotations

from pathlib import Path

from smartrain.workflows.inference.inference_cli import _resolve_model
from smartrain.workflows.testing.model_test_cli import _infer_task_from_training_metadata
from smartrain.workflows.analyze.results_analyzer import _unified_read_enabled
from smartrain.core.runtime.workspace_paths import WorkspaceLayout, deploy_workspace


def test_cutover_defaults_to_canonical_mode(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.delenv("SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK", raising=False)
    assert _unified_read_enabled() is True


def test_legacy_read_env_vars_are_ignored(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.setenv("SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK", "1")

    assert _unified_read_enabled() is True

    class _Ctx:
        task_type = "segmentation"
        run_id = "r1"

    monkeypatch.setattr("smartrain.run_model_contract.gateway.resolve_task_context", lambda *_a, **_k: _Ctx())
    assert _infer_task_from_training_metadata(str(tmp_path)) == "segment"
