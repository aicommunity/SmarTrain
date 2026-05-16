#!/usr/bin/env python3
"""CLI facade: implementation in smartrain.services.analyze.cli_commands."""

from __future__ import annotations

from typing import Any

from smartrain.services.analyze import cli_commands as _impl
from smartrain.services.analyze.prompts import prompt_choice, prompt_int, prompt_text

build_analyze_arg_parser = _impl.build_analyze_arg_parser
main = _impl.main


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))
