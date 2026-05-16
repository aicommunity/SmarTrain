from __future__ import annotations

from smartrain.services.training import train_cli_parsers as _parsers
from smartrain.services.training.train_cli_main import main as _main
from smartrain.workflows.training import train_wiring

build_train_arg_parser = _parsers.build_train_arg_parser


def main(argv: list[str] | None = None):
    return _main(
        argv,
        run_resume_command_cb=train_wiring.run_resume_command,
        run_calc_confidence_command_cb=train_wiring.run_calc_confidence_command,
    )
