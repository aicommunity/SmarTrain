from __future__ import annotations

from smartrain import inference_cli


def build_arg_parser():
    return inference_cli.build_inference_arg_parser()


def main(argv: list[str] | None = None):
    return inference_cli.main(argv)

