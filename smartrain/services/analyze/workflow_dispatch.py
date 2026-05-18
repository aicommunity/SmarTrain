"""Analyze CLI dispatch surface (patchable in tests; no services→workflows imports)."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    from smartrain.services.analyze import cli_commands as _impl

    return getattr(_impl, name)
