"""Pre-wired callbacks for ``run_train_cli_pipeline`` (single-arg CLI surface)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smartrain.cli_entrypoints.support.cli_prompts import prompt_choice

from smartrain.core.training.train_profile import (
    dataset_root_from_data_yaml,
    extract_smartrain_options,
    load_train_profile,
    resolve_profile_data_path,
)
from smartrain.providers.core.global_index import list_provider_records, reconcile_stale_provider_paths
from smartrain.services.training.train_interactive_helpers_service import (
    apply_external_provider_defaults as _apply_external_provider_defaults,
    get_installed_external_provider_record as _svc_get_installed_external_provider_record,
    installed_external_provider_ids as _svc_installed_external_provider_ids,
    installed_external_provider_records as _svc_installed_external_provider_records,
)
from smartrain.services.training.train_interactive_setup_service import run_interactive_train_setup as _run_interactive_train_setup
from smartrain.services.training.train_cli_parsers import BATCH, EPOCHS, IMG_SIZE, MODEL_VERSION
from smartrain.services.training.train_config_merge_service import (
    _ULTRALYTICS_YAML_IGNORED_KEYS,
    merge_sources_with_priority,
)
from smartrain.services.training.train_config_kwargs_service import load_ultralytics_yaml
from smartrain.services.training.train_base_runs_service import (
    base_run_summary as _base_run_summary_svc,
    collect_available_base_runs as _collect_available_base_runs_svc,
    print_available_base_runs,
    prompt_base_run_args_yaml as _prompt_base_run_args_yaml_svc,
)
from smartrain.services.training.train_interactive_helpers_service import (
    format_numbered_columns as _format_numbered_columns_svc,
    load_available_datasets,
    model_matches_task,
    pick_model_interactive as _pick_model_interactive_svc,
    prompt_dataset_name as _prompt_dataset_name_svc,
    train_model_picker_options as _train_model_picker_options_svc,
    installed_external_provider_ids as _installed_external_provider_ids_svc,
)
from smartrain.services.training.train_prompts import (
    prompt_input,
    prompt_int,
    prompt_optional_float,
    prompt_optional_int,
    prompt_train_device,
    prompt_yes_no,
)
from smartrain.core.runtime.device_selector import default_device_value
from smartrain.core.runtime.workspace_paths import (
    DATASETS_INFO_FILE,
    WORKSPACE_ENV_VAR,
    WorkspaceLayout,
    resolve_dataset_root,
    resolve_workspace_root,
)
from smartrain.services.training.train_cli_paths_service import (
    resolve_cli_paths_with_profile as _resolve_cli_paths_with_profile,
)
from smartrain.services.training.train_model_resolution_service import normalize_model_spec
from smartrain.services.training.train_runtime_data_yaml_service import (
    resolve_training_data_path as _resolve_training_data_path,
)

_MANUAL_MODEL_ENTRY = "<manual>"


def load_ultralytics_yaml_cb(path: str | None) -> dict[str, Any]:
    return load_ultralytics_yaml(path, load_train_profile_cb=load_train_profile)


def normalize_model_spec_cb(spec, *, add_pt_when_missing: bool = False) -> str:
    return normalize_model_spec(
        spec,
        default_model=MODEL_VERSION,
        add_pt_when_missing=add_pt_when_missing,
    )


def resolve_training_data_path_for_cli(layout: WorkspaceLayout, data_arg: str) -> str:
    return _resolve_training_data_path(
        layout,
        data_arg,
        datasets_info_file=DATASETS_INFO_FILE,
        resolve_dataset_root_cb=resolve_dataset_root,
    )


def resolve_cli_paths_with_profile_cb(args, u_cfg: dict) -> tuple[str | None, str, str]:
    return _resolve_cli_paths_with_profile(
        args,
        u_cfg,
        workspace_env_var=WORKSPACE_ENV_VAR,
        resolve_workspace_root_cb=resolve_workspace_root,
        workspace_layout_cb=WorkspaceLayout,
        resolve_training_data_path_cb=resolve_training_data_path_for_cli,
        resolve_profile_data_path_cb=resolve_profile_data_path,
        dataset_root_from_data_yaml_cb=dataset_root_from_data_yaml,
    )


def _installed_external_provider_records() -> list[dict[str, Any]]:
    return _svc_installed_external_provider_records(
        reconcile_paths_cb=reconcile_stale_provider_paths,
        list_records_cb=list_provider_records,
    )


def _get_installed_external_provider_record(provider_id: str) -> dict[str, Any] | None:
    return _svc_get_installed_external_provider_record(
        provider_id,
        installed_records_cb=_installed_external_provider_records,
    )


def _installed_external_provider_ids() -> list[str]:
    return _svc_installed_external_provider_ids(installed_records_cb=_installed_external_provider_records)


def apply_external_provider_defaults_cb(args) -> None:
    _apply_external_provider_defaults(args, get_installed_record_cb=_get_installed_external_provider_record)


def _prompt_dataset_name_cb(available: list[str]) -> str:
    return _prompt_dataset_name_svc(available, prompt_choice_cb=prompt_choice)


def _base_run_summary_cb(args_yaml: Path) -> dict[str, str]:
    return _base_run_summary_svc(args_yaml, load_ultralytics_yaml_cb=load_ultralytics_yaml_cb)


def _collect_available_base_runs_cb(layout: WorkspaceLayout, selected_dataset: str) -> list[dict[str, str]]:
    return _collect_available_base_runs_svc(
        layout,
        selected_dataset,
        base_run_summary_cb=_base_run_summary_cb,
    )


def _prompt_base_run_args_yaml_cb(runs: list[dict[str, str]], default_path: str | None = None) -> str | None:
    return _prompt_base_run_args_yaml_svc(
        runs,
        default_path,
        prompt_input_cb=prompt_input,
    )


def _train_model_picker_options_cb(default_model: str) -> list[str]:
    return _train_model_picker_options_svc(
        default_model,
        installed_records_cb=_installed_external_provider_records,
        manual_model_entry=_MANUAL_MODEL_ENTRY,
    )


def _pick_model_interactive_cb(options: list[str], default_alias: str) -> str:
    return _pick_model_interactive_svc(
        options,
        default_alias,
        format_columns_cb=_format_numbered_columns_svc,
        prompt_input_cb=prompt_input,
    )


def run_interactive_train_setup_cb(args) -> bool:
    return _run_interactive_train_setup(
        args,
        model_version=MODEL_VERSION,
        manual_model_entry=_MANUAL_MODEL_ENTRY,
        epochs_default=EPOCHS,
        batch_default=BATCH,
        img_size_default=IMG_SIZE,
        ultralytics_yaml_ignored_keys=set(_ULTRALYTICS_YAML_IGNORED_KEYS),
        resolve_workspace_root_cb=resolve_workspace_root,
        workspace_layout_cb=WorkspaceLayout,
        prompt_input_cb=prompt_input,
        load_available_datasets_cb=load_available_datasets,
        prompt_dataset_name_cb=_prompt_dataset_name_cb,
        collect_available_base_runs_cb=_collect_available_base_runs_cb,
        print_available_base_runs_cb=print_available_base_runs,
        prompt_base_run_args_yaml_cb=_prompt_base_run_args_yaml_cb,
        load_ultralytics_yaml_cb=load_ultralytics_yaml_cb,
        extract_smartrain_options_cb=extract_smartrain_options,
        normalize_model_spec_cb=normalize_model_spec_cb,
        train_model_picker_options_cb=_train_model_picker_options_cb,
        model_matches_task_cb=model_matches_task,
        pick_model_interactive_cb=_pick_model_interactive_cb,
        installed_external_provider_ids_cb=_installed_external_provider_ids,
        prompt_int_cb=prompt_int,
        prompt_train_device_cb=prompt_train_device,
        prompt_yes_no_cb=prompt_yes_no,
        prompt_optional_int_cb=prompt_optional_int,
        prompt_optional_float_cb=prompt_optional_float,
        default_device_value_cb=default_device_value,
    )
