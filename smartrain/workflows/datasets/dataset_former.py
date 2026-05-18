"""CLI facade: implementation in smartrain.services.datasets.dataset_former."""

from __future__ import annotations

from typing import Any

from smartrain.services.datasets import dataset_former as _impl

build_dataset_former_arg_parser = _impl.build_dataset_former_arg_parser
main = _impl.main

parse_fusion_split_arg = _impl.parse_fusion_split_arg
TRAIN_PART = _impl.TRAIN_PART
VAL_PART = _impl.VAL_PART
TEST_PART = _impl.TEST_PART


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))

