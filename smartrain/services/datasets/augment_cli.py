"""Compatibility wrapper around dataset_augment."""

from __future__ import annotations

from smartrain.services.datasets import dataset_augment as _impl

build_augment_arg_parser = _impl.build_augment_arg_parser
main = _impl.main

__all__ = [
    "build_augment_arg_parser",
    "main",
]
