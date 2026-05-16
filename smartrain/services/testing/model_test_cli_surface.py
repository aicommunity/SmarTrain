"""Mutable CLI bindings for workflow facade and tests."""

from __future__ import annotations

import subprocess

from smartrain.cli_support.cli_prompts import print_numbered_options, prompt_choice, prompt_text, prompt_yes_no
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.run_artifacts import canonical_run_model_path
from smartrain.services.testing.model_test_service import (
    has_complete_test_artifacts,
    has_matching_test_artifacts,
)

# Filled when ``model_test_cli_service`` finishes loading.
_check_onnx_format_preflight = None
_check_native_format_preflight = None
_pick_interactive_target = None
_prompt_export_backends_interactive = None
_prompt_artifact_selection_interactive = None
_resolve_existing_artifact = None
_run_native_backend_isolated = None
