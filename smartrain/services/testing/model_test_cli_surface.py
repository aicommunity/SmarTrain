"""Mutable CLI bindings for workflow facade and tests."""

from __future__ import annotations

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
