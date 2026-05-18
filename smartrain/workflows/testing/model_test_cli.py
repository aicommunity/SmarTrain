"""Workflow facade: model-test CLI."""

from __future__ import annotations

from smartrain.services.testing import model_test_cli_service as _impl
from smartrain.services.testing.model_test_cli_surface import *  # noqa: F403

main = _impl.main
build_model_test_arg_parser = _impl.build_model_test_arg_parser


def __getattr__(name: str):
    return getattr(_impl, name)
