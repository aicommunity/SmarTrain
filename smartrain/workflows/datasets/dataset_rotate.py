"""CLI facade: implementation in smartrain.services.datasets.dataset_rotate."""

from __future__ import annotations

from typing import Any

from smartrain.services.datasets import dataset_rotate as _impl

build_rotate_arg_parser = _impl.build_rotate_arg_parser
main = _impl.main


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))
