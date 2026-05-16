"""Public runtime API for model-test orchestration services."""

from __future__ import annotations

from typing import Any

from smartrain.workflows.testing.model_test_service import (
    SUPPORTED_TEST_FORMATS,
    has_complete_test_artifacts,
    persist_target_test_artifacts_state,
)


def _cli():
    from smartrain.workflows.testing import model_test_cli as cli

    return cli


def discover_run_artifact_candidates(root_dir: str) -> dict[str, list[str]]:
    return _cli()._discover_run_artifact_candidates(root_dir)


def prompt_export_backends_interactive(root_dir: str, candidates: dict[str, list[str]]) -> set[str]:
    return _cli()._prompt_export_backends_interactive(root_dir, candidates)


def prompt_artifact_selection_interactive(
    narrowed: dict[str, list[str]],
) -> list[tuple[str, str]]:
    return _cli()._prompt_artifact_selection_interactive(narrowed)


def print_test_plan(
    *,
    target_kind: str,
    target_label: str | None,
    root_dir: str,
    data_yaml: str,
    formats: list[str],
    split_name: str,
) -> None:
    _cli()._print_test_plan(
        target_kind=target_kind,
        target_label=target_label,
        root_dir=root_dir,
        data_yaml=data_yaml,
        formats=formats,
        split_name=split_name,
    )


def collect_interactive_rerun_decisions(**kwargs: Any) -> dict[str, bool]:
    return _cli()._collect_interactive_rerun_decisions(**kwargs)


def artifact_key(format_name: str, target_path: str) -> str:
    return _cli()._artifact_key(format_name, target_path)


def should_rerun_existing_match(**kwargs: Any) -> bool:
    return _cli()._should_rerun_existing_match(**kwargs)


def resolve_existing_artifact(
    *,
    root_dir: str,
    primary_path: str,
    format_name: str,
    target_kind: str,
) -> str:
    return _cli()._resolve_existing_artifact(
        root_dir=root_dir,
        primary_path=primary_path,
        format_name=format_name,
        target_kind=target_kind,
    )


def check_native_format_preflight(fmt: str) -> tuple[bool, str | None]:
    return _cli()._check_native_format_preflight(fmt)


def run_native_backend_isolated(**kwargs: Any) -> tuple[bool, str | None]:
    return _cli()._run_native_backend_isolated(**kwargs)


def check_onnx_format_preflight(policy: str) -> tuple[bool, str | None]:
    return _cli()._check_onnx_format_preflight(policy)


def run_ultralytics_backend(**kwargs: Any):
    from smartrain.services.testing.backends.format_runners import run_ultralytics_backend as _run

    return _run(**kwargs)


def run_native_format_backend(**kwargs: Any):
    from smartrain.services.testing.backends.format_runners import run_native_format_backend as _run

    return _run(**kwargs)


def complete_missing_test_artifacts(root_dir: str, **kwargs: Any) -> None:
    _cli().complete_missing_test_artifacts(root_dir, **kwargs)
