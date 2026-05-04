from __future__ import annotations

from smartrain import model_training_module


def build_arg_parser():
    return model_training_module.build_train_arg_parser()


def main(argv: list[str] | None = None):
    return model_training_module.main(argv)

