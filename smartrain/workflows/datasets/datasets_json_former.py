"""CLI facade: implementation in smartrain.services.datasets.datasets_json_former."""

from __future__ import annotations

from typing import Any

from smartrain.services.datasets import datasets_json_former as _impl

build_datasets_json_arg_parser = _impl.build_datasets_json_arg_parser
main = _impl.main


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))

