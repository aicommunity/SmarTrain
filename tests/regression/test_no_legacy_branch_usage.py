from __future__ import annotations

from pathlib import Path

from smartrain.workflows.inference.inference_cli import _resolve_model
from smartrain.results_analyzer import _read_test_metrics_for_run
from smartrain.core.runtime.workspace_paths import WorkspaceLayout, deploy_workspace


def test_inference_run_path_uses_canonical_branch_by_default(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.delenv("SMARTTRAIN_CANONICAL_READ", raising=False)
    monkeypatch.delenv("SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK", raising=False)

    run_dir = tmp_path / "runs" / "ds_a" / "run_a"
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "from_gateway.onnx"
    model_path.write_bytes(b"x")

    class _M:
        weights_path = str(model_path)
        model_id = "m1"

    class _R:
        run_id = "r1"

    class _P:
        models = [_M()]
        runs = [_R()]

    class _C:
        run_id = "r1"

    monkeypatch.setattr("smartrain.orchestrators.canonical_gateway.load_target", lambda *_a, **_k: _P())
    monkeypatch.setattr("smartrain.orchestrators.canonical_gateway.resolve_task_context", lambda *_a, **_k: _C())

    import argparse

    args = argparse.Namespace(model_name=None, run=str(run_dir), weights=None)
    resolved, name, source = _resolve_model(args, WorkspaceLayout(str(tmp_path)))
    assert source == "runs"
    assert name == "r1"
    assert resolved == model_path.resolve()


def test_results_metrics_reader_does_not_call_legacy_by_default(monkeypatch, tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    monkeypatch.delenv("SMARTTRAIN_CANONICAL_READ", raising=False)
    monkeypatch.delenv("SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK", raising=False)

    run_dir = tmp_path / "runs" / "ds_a" / "run_b"
    run_dir.mkdir(parents=True, exist_ok=True)

    class _Metric:
        primary_metrics = {"mAP50-95": 0.61}
        secondary_metrics = {"Box-F1": 0.72}

    monkeypatch.setattr("smartrain.orchestrators.canonical_gateway.load_metrics", lambda *_a, **_k: [_Metric()])

    row = _read_test_metrics_for_run(str(run_dir))
    assert row.get("mAP50-95") == 0.61
    assert row.get("Box-F1") == 0.72
