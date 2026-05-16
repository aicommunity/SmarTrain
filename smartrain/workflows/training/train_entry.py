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
        parse_args_cb=train_wiring.parse_train_args_cb,
        run_interactive_train_setup_cb=train_wiring.run_interactive_train_setup_cb,
        load_ultralytics_yaml_cb=train_wiring.load_ultralytics_yaml_cb,
        resolve_cli_paths_with_profile_cb=train_wiring.resolve_cli_paths_with_profile_cb,
        run_train_after_setup_cb=train_wiring.run_train_after_setup_cb,
        normalize_model_spec_cb=train_wiring.normalize_model_spec_cb,
    )
