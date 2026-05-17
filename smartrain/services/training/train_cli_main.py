"""Train CLI main pipeline (argparse dispatch without resume subcommands)."""

from __future__ import annotations

import sys
from typing import Any, Callable

from smartrain.cli_entrypoints.support.cli_contracts import emit_replay, make_command_request
from smartrain.core.runtime.device_selector import default_device_value, device_display_name, resolve_device_request
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.device_selector import validate_device_available
from smartrain.core.training.train_profile import (
    apply_cli_smartrain_overrides,
    load_train_profile,
    merge_cli_into_ultralytics_cfg,
)
from smartrain.external_providers.registry import list_provider_specs
from smartrain.core.training.external_model_ref import parse_external_model_ref, validate_external_model_ref
from smartrain.services.training.train_cli_parsers import (
    BATCH,
    EPOCHS,
    IMG_SIZE,
    MODEL_VERSION,
    build_train_arg_parser,
    parse_train_args,
)
from smartrain.services.training.train_config_merge_service import merge_sources_with_priority
from smartrain.services.training.train_cli_callbacks import (
    apply_external_provider_defaults_cb,
    load_ultralytics_yaml_cb as _default_load_ultralytics_yaml_cb,
    normalize_model_spec_cb as _default_normalize_model_spec_cb,
    resolve_cli_paths_with_profile_cb as _default_resolve_cli_paths_with_profile_cb,
    run_interactive_train_setup_cb as _default_run_interactive_train_setup_cb,
)
from smartrain.services.train_service import run_train_after_setup as _default_run_train_after_setup
from smartrain.services.training.training_cli_orchestration_service import (
    handle_aux_train_commands,
    run_train_cli_pipeline,
)


def ensure_device_available_or_raise(device: str | None) -> None:
    validate_device_available(device)


def main(
    argv: list[str] | None = None,
    *,
    run_resume_command_cb: Callable[[list[str]], int] | None = None,
    run_calc_confidence_command_cb: Callable[[list[str]], int] | None = None,
    parse_args_cb: Callable[[list[str]], Any] | None = None,
    run_interactive_train_setup_cb: Callable[[Any], bool] | None = None,
    load_ultralytics_yaml_cb: Callable[[str | None], dict[str, Any]] | None = None,
    resolve_cli_paths_with_profile_cb: Callable[[Any, dict[str, Any]], tuple[str, str, str]] | None = None,
    run_train_after_setup_cb: Callable[..., Any] | None = None,
    normalize_model_spec_cb: Callable[..., str] | None = None,
) -> Any:
    if argv is None:
        argv = sys.argv[1:]
    request = make_command_request("train", argv, interactive_allowed=is_interactive_allowed(argv))
    if run_resume_command_cb is None or run_calc_confidence_command_cb is None:
        raise RuntimeError(
            "train_cli_main requires run_resume_command_cb and run_calc_confidence_command_cb "
            "(inject from workflows.training.train_wiring via train_entry)."
        )
    aux_rc = handle_aux_train_commands(
        argv,
        run_resume_command_cb=run_resume_command_cb,
        run_calc_confidence_command_cb=run_calc_confidence_command_cb,
    )
    if aux_rc is not None:
        return aux_rc
    return run_train_cli_pipeline(
        argv,
        request=request,
        parse_args_cb=parse_args_cb or parse_train_args,
        apply_external_provider_defaults_cb=apply_external_provider_defaults_cb,
        list_provider_specs_cb=list_provider_specs,
        parse_external_model_ref_cb=parse_external_model_ref,
        validate_external_model_ref_cb=validate_external_model_ref,
        build_train_arg_parser_cb=build_train_arg_parser,
        run_interactive_train_setup_cb=run_interactive_train_setup_cb or _default_run_interactive_train_setup_cb,
        emit_replay_cb=emit_replay,
        load_train_profile_cb=load_train_profile,
        load_ultralytics_yaml_cb=load_ultralytics_yaml_cb or _default_load_ultralytics_yaml_cb,
        merge_sources_with_priority_cb=merge_sources_with_priority,
        merge_cli_into_ultralytics_cfg_cb=merge_cli_into_ultralytics_cfg,
        apply_cli_smartrain_overrides_cb=apply_cli_smartrain_overrides,
        resolve_device_request_cb=resolve_device_request,
        resolve_cli_paths_with_profile_cb=resolve_cli_paths_with_profile_cb or _default_resolve_cli_paths_with_profile_cb,
        normalize_model_spec_cb=normalize_model_spec_cb or _default_normalize_model_spec_cb,
        ensure_device_available_or_raise_cb=ensure_device_available_or_raise,
        device_display_name_cb=device_display_name,
        run_train_after_setup_cb=run_train_after_setup_cb or _default_run_train_after_setup,
        default_device_value_cb=default_device_value,
        model_version_default=MODEL_VERSION,
        epochs_default=EPOCHS,
        batch_default=BATCH,
        img_size_default=IMG_SIZE,
    )
