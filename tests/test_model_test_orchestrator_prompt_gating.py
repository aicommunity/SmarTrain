"""Orchestrator prompt gating after removal of argv --formats heuristic (H2)."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from smartrain.services.model_test_orchestrator import run_model_test_after_setup


def test_non_interactive_runs_skips_backend_and_artifact_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typer --nit sets interactive=False; orchestrator must not re-prompt for backends."""
    prompt_export = MagicMock(return_value=["pt"])
    prompt_artifacts = MagicMock(return_value=[])
    discover = MagicMock(return_value={"pt": ["/tmp/run/weights/best.pt"]})

    monkeypatch.setattr(
        "smartrain.core.workflow_adapters.testing_runtime_api.discover_run_artifact_candidates",
        discover,
    )
    monkeypatch.setattr(
        "smartrain.core.workflow_adapters.testing_runtime_api.prompt_export_backends_interactive",
        prompt_export,
    )
    monkeypatch.setattr(
        "smartrain.core.workflow_adapters.testing_runtime_api.prompt_artifact_selection_interactive",
        prompt_artifacts,
    )
    monkeypatch.setattr(
        "smartrain.core.workflow_adapters.testing_runtime_api.print_test_plan",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "smartrain.core.workflow_adapters.testing_runtime_api.collect_interactive_rerun_decisions",
        lambda **_: {},
    )
    monkeypatch.setattr(
        "smartrain.core.workflow_adapters.testing_runtime_api.has_complete_test_artifacts",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "smartrain.cli_support.cli_contracts.emit_replay",
        lambda **_: "smartrain test --nit",
    )

    parser = argparse.ArgumentParser()
    args = SimpleNamespace(
        task="detect",
        non_interactive=True,
        force=False,
        missing_only=True,
        formats="pt,onnx",
        deep_diagnostics=False,
    )
    request = SimpleNamespace(interactive_used=False)

    run_model_test_after_setup(
        parser=parser,
        args=args,
        request=request,
        workspace_root="/tmp/ws",
        interactive=False,
        root_dir="/tmp/run",
        primary_path="/tmp/run/weights/best.pt",
        target_kind="runs",
        target_label=None,
        data_yaml="/tmp/ds/data.yaml",
        formats=["pt"],
        onnx_provider_policy="gpu_preferred",
        requested_imgsz=None,
        requested_conf=None,
        requested_iou=None,
    )

    discover.assert_not_called()
    prompt_export.assert_not_called()
    prompt_artifacts.assert_not_called()
