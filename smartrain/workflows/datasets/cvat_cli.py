"""CLI facade: implementation in smartrain.services.datasets.cvat_cli."""

from __future__ import annotations

from typing import Any

from smartrain.services.datasets import cvat_cli as _impl

build_cvat_arg_parser = _impl.build_cvat_arg_parser
main = _impl.main


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))

