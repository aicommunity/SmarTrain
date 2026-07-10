"""CLI facade: implementation in smartrain.services.datasets.dataset_convert_cli."""

from __future__ import annotations

from smartrain.services.datasets import dataset_convert_cli as _impl

build_dataset_convert_arg_parser = _impl.build_dataset_convert_arg_parser
main = _impl.main
