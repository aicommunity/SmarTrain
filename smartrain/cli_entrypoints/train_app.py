from __future__ import annotations

from smartrain.workflows.training import train_entry


def build_arg_parser():
    return train_entry.build_train_arg_parser()


def main(argv: list[str] | None = None):
    return train_entry.main(argv)

