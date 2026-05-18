"""CLI facade: datasets JSON scan entry."""

from __future__ import annotations

from smartrain.services.datasets import datasets_json_former as _impl

build_datasets_json_arg_parser = _impl.build_datasets_json_arg_parser
main = _impl.main
