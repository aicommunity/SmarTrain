from __future__ import annotations

from smartrain.workflows.datasets import datasets_json_former


def build_datasets_json_arg_parser():
    return datasets_json_former.build_datasets_json_arg_parser()


def main(argv: list[str] | None = None):
    return datasets_json_former.main(argv)

