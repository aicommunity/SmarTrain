"""Mutable CLI bindings for workflow facade and tests."""

from __future__ import annotations

import subprocess

from smartrain.cli_entrypoints.support.cli_prompts import (  # noqa: F401
    print_numbered_options,
    prompt_choice,
    prompt_text,
    prompt_yes_no,
)
from smartrain.core.runtime.interactive_contract import is_interactive_allowed  # noqa: F401
from smartrain.core.runtime.run_artifacts import preferred_run_model_path  # noqa: F401
from smartrain.services.testing.model_test_service import (  # noqa: F401
    has_complete_test_artifacts,
    has_matching_test_artifacts,
)

__all__ = [
    "has_complete_test_artifacts",
    "has_matching_test_artifacts",
    "is_interactive_allowed",
    "preferred_run_model_path",
    "print_numbered_options",
    "prompt_choice",
    "prompt_text",
    "prompt_yes_no",
    "subprocess",
]

_DELEGATED_BINDINGS = frozenset(
    {
        "_check_onnx_format_preflight",
        "_check_native_format_preflight",
        "_pick_interactive_target",
        "_prompt_export_backends_interactive",
        "_prompt_artifact_selection_interactive",
        "_resolve_existing_artifact",
        "_run_native_backend_isolated",
    }
)


def __getattr__(name: str):
    if name in _DELEGATED_BINDINGS:
        from smartrain.services.testing import model_test_cli_service as impl

        return getattr(impl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
