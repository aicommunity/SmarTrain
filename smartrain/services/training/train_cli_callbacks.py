"""Pre-wired callbacks for ``run_train_cli_pipeline`` (single-arg CLI surface)."""

from __future__ import annotations

from typing import Any

from smartrain.core.training.train_profile import extract_smartrain_options, load_train_profile
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
    collect_available_base_runs,
    print_available_base_runs,
    prompt_base_run_args_yaml,
)
from smartrain.services.training.train_interactive_helpers_service import (
    load_available_datasets,
    model_matches_task,
    pick_model_interactive,
    prompt_dataset_name,
    train_model_picker_options,
    installed_external_provider_ids,
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
from smartrain.core.runtime.workspace_paths import WorkspaceLayout, resolve_workspace_root
from smartrain.services.training.train_model_resolution_service import normalize_model_spec

_MANUAL_MODEL_ENTRY = "<manual>"


def load_ultralytics_yaml_cb(path: str | None) -> dict[str, Any]:
    return load_ultralytics_yaml(path, load_train_profile_cb=load_train_profile)


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
        prompt_dataset_name_cb=prompt_dataset_name,
        collect_available_base_runs_cb=collect_available_base_runs,
        print_available_base_runs_cb=print_available_base_runs,
        prompt_base_run_args_yaml_cb=prompt_base_run_args_yaml,
        load_ultralytics_yaml_cb=load_ultralytics_yaml_cb,
        extract_smartrain_options_cb=extract_smartrain_options,
        normalize_model_spec_cb=normalize_model_spec,
        train_model_picker_options_cb=train_model_picker_options,
        model_matches_task_cb=model_matches_task,
        pick_model_interactive_cb=pick_model_interactive,
        installed_external_provider_ids_cb=_installed_external_provider_ids,
        prompt_int_cb=prompt_int,
        prompt_train_device_cb=prompt_train_device,
        prompt_yes_no_cb=prompt_yes_no,
        prompt_optional_int_cb=prompt_optional_int,
        prompt_optional_float_cb=prompt_optional_float,
        default_device_value_cb=default_device_value,
    )
