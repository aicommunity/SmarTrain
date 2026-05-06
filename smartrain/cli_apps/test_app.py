from __future__ import annotations

from smartrain.workflows.testing import model_test_cli


def build_arg_parser():
    return model_test_cli.build_model_test_arg_parser()


def main(argv: list[str] | None = None):
    return model_test_cli.main(argv)

