from __future__ import annotations

from typing import Any

from smartrain.workflows.testing import model_test_runtime_api as _api

SUPPORTED_TEST_FORMATS = _api.SUPPORTED_TEST_FORMATS
has_complete_test_artifacts = _api.has_complete_test_artifacts
persist_target_test_artifacts_state = _api.persist_target_test_artifacts_state


def discover_run_artifact_candidates(root_dir: str) -> dict[str, list[str]]:
    return _api.discover_run_artifact_candidates(root_dir)


def prompt_export_backends_interactive(root_dir: str, candidates: dict[str, list[str]]) -> set[str]:
    return _api.prompt_export_backends_interactive(root_dir, candidates)


def prompt_artifact_selection_interactive(narrowed: dict[str, list[str]]) -> list[tuple[str, str]]:
    return _api.prompt_artifact_selection_interactive(narrowed)


def print_test_plan(**kwargs: Any) -> None:
    _api.print_test_plan(**kwargs)


def collect_interactive_rerun_decisions(**kwargs: Any) -> dict[str, bool]:
    return _api.collect_interactive_rerun_decisions(**kwargs)


def artifact_key(format_name: str, target_path: str) -> str:
    return _api.artifact_key(format_name, target_path)


def should_rerun_existing_match(**kwargs: Any) -> bool:
    return _api.should_rerun_existing_match(**kwargs)


def resolve_existing_artifact(**kwargs: Any) -> str:
    return _api.resolve_existing_artifact(**kwargs)


def check_native_format_preflight(fmt: str) -> tuple[bool, str | None]:
    return _api.check_native_format_preflight(fmt)


def run_native_backend_isolated(**kwargs: Any) -> tuple[bool, str | None]:
    return _api.run_native_backend_isolated(**kwargs)


def check_onnx_format_preflight(policy: str) -> tuple[bool, str | None]:
    return _api.check_onnx_format_preflight(policy)


def run_ultralytics_backend(**kwargs: Any):
    return _api.run_ultralytics_backend(**kwargs)


def run_native_format_backend(**kwargs: Any):
    return _api.run_native_format_backend(**kwargs)


def complete_missing_test_artifacts(root_dir: str, **kwargs: Any) -> None:
    _api.complete_missing_test_artifacts(root_dir, **kwargs)
