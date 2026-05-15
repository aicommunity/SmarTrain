import json
import os
import argparse
import platform
import shutil
import socket
import subprocess
import sys
import traceback
import gc
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from smartrain.core.runtime.mpl_runtime import configure_matplotlib_before_ultralytics, ensure_matplotlib_training_runtime

configure_matplotlib_before_ultralytics()
from ultralytics import YOLO

from smartrain.cli_support.cli_argparse import CliArgumentParser
from smartrain.cli_support.cli_prompts import print_numbered_options, prompt_text
from smartrain.cli_support.cli_contracts import emit_replay, make_command_request
from smartrain.workflows.datasets.dataset_hash import calculate_dataset_hash
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.workflows.training.train_resume import (
    RUN_STATUS_RESUMABLE_INCOMPLETE,
    RUN_STATUS_TRAINING_COMPLETE_TEST_PENDING,
    RunDiagnosis,
    resolve_dataset_path_for_resume,
    list_incomplete_runs,
    resume_training_in_run,
    update_resume_test_metadata,
    update_resume_metadata,
)
from smartrain.core.training.train_profile import (
    apply_cli_smartrain_overrides,
    dataset_root_from_data_yaml,
    extract_smartrain_options,
    load_train_profile,
    merge_cli_into_ultralytics_cfg,
    resolve_profile_data_path,
    task_to_metadata_task_type,
)
from smartrain.core.runtime.device_selector import (
    default_device_value,
    device_display_name,
    resolve_device_request,
    validate_device_available,
)
from smartrain.core.training.train_model_catalog import (
    TrainModelCatalog,
    is_supported_external_provider_model,
)
from smartrain.providers.core.global_index import (
    get_provider_location,
    list_provider_records,
    reconcile_stale_provider_paths,
)
from smartrain.external_providers.runner import run_external_infer, run_external_train
from smartrain.core.training.external_model_ref import parse_external_model_ref, validate_external_model_ref
from smartrain.external_providers.registry import list_provider_specs
from smartrain.core.runtime.path_portable import relativize_if_under
from smartrain.services.train_service import run_train_after_setup
from smartrain.workflows.training.training_cli_orchestration_service import (
    handle_aux_train_commands as _svc_handle_aux_train_commands,
    run_train_cli_pipeline as _svc_run_train_cli_pipeline,
)
from smartrain.workflows.training.train_options import (
    prompt_input as _train_prompt_input,
    prompt_train_device as _train_prompt_device,
)
from smartrain.workflows.training.train_system_profile_service import (
    bytes_to_gb as _svc_bytes_to_gb,
    collect_system_profile as _svc_collect_system_profile,
    linux_cpu_model_name as _svc_linux_cpu_model_name,
    linux_fs_type_for_mount as _svc_linux_fs_type_for_mount,
    linux_mem_total_bytes as _svc_linux_mem_total_bytes,
    linux_physical_core_count as _svc_linux_physical_core_count,
    resolve_mount_point as _svc_resolve_mount_point,
)
from smartrain.workflows.training.train_runtime_data_yaml_service import (
    build_runtime_data_yaml as _svc_build_runtime_data_yaml,
    pick_split_relative_dir as _svc_pick_split_relative_dir,
    resolve_training_data_path as _svc_resolve_training_data_path,
    split_dir_from_dataset_yaml as _svc_split_dir_from_dataset_yaml,
)
from smartrain.workflows.training.train_metadata_io_service import (
    get_relative_path as _svc_get_relative_path,
    relative_to_workspace as _svc_relative_to_workspace,
    write_json_atomic as _svc_write_json_atomic,
)
from smartrain.workflows.training.train_resume_backoff_service import (
    complete_missing_test_with_backoff as _svc_complete_missing_test_with_backoff,
    default_resume_test_batch as _svc_default_resume_test_batch,
    is_cuda_oom_error as _svc_is_cuda_oom_error,
    next_backoff_batch as _svc_next_backoff_batch,
)
from smartrain.workflows.training.train_cli_paths_service import (
    resolve_cli_paths_with_profile as _svc_resolve_cli_paths_with_profile,
)
from smartrain.workflows.training.train_config_kwargs_service import (
    finalize_train_kwargs as _svc_finalize_train_kwargs,
    load_ultralytics_yaml as _svc_load_ultralytics_yaml,
)
from smartrain.workflows.training.train_model_resolution_service import (
    extract_effective_loaded_model as _svc_extract_effective_loaded_model,
    extract_model_family_scale as _svc_extract_model_family_scale,
    normalize_model_spec as _svc_normalize_model_spec,
)
from smartrain.workflows.training.train_base_runs_service import (
    base_run_summary as _svc_base_run_summary,
    collect_available_base_runs as _svc_collect_available_base_runs,
    extract_run_timestamp as _svc_extract_run_timestamp,
    print_available_base_runs as _svc_print_available_base_runs,
    prompt_base_run_args_yaml as _svc_prompt_base_run_args_yaml,
)
from smartrain.workflows.training.train_interactive_helpers_service import (
    apply_external_provider_defaults as _svc_apply_external_provider_defaults,
    format_numbered_columns as _svc_format_numbered_columns,
    get_installed_external_provider_record as _svc_get_installed_external_provider_record,
    installed_external_provider_ids as _svc_installed_external_provider_ids,
    installed_external_provider_records as _svc_installed_external_provider_records,
    load_available_datasets as _svc_load_available_datasets,
    model_matches_task as _svc_model_matches_task,
    pick_model_interactive as _svc_pick_model_interactive,
    prompt_dataset_name as _svc_prompt_dataset_name,
    train_model_picker_options as _svc_train_model_picker_options,
)
from smartrain.workflows.training.train_interactive_setup_service import (
    get_interactive_default as _svc_get_interactive_default,
    run_interactive_train_setup as _svc_run_interactive_train_setup,
)
from smartrain.services.train_runtime_helpers import (
    build_run_name as _shared_build_run_name,
    ensure_external_best_checkpoint_layout as _shared_ensure_external_best_checkpoint_layout,
    json_safe_train_summary as _shared_json_safe_train_summary,
    load_batch_from_training_metadata as _shared_load_batch_from_training_metadata,
    maybe_free_cuda_memory as _shared_maybe_free_cuda_memory,
    normalize_external_run_layout as _shared_normalize_external_run_layout,
    run_mfel_external_val_fallback as _shared_run_mfel_external_val_fallback,
    resolve_external_eval_source as _shared_resolve_external_eval_source,
    write_external_fallback_metrics as _shared_write_external_fallback_metrics,
)
from smartrain.core.training.confidence_recommendation import (
    compute_confidence_recommendations,
    recommendation_file_path,
    recommendations_complete,
    read_recommendation_file,
    write_not_available_recommendations,
    write_recommendation_file,
)
from smartrain.core.runtime.workspace_paths import (
    WORKSPACE_ENV_VAR,
    WorkspaceLayout,
    resolve_workspace_root,
    resolve_dataset_root,
    DATASETS_INFO_FILE,
)
from smartrain.core.runtime.run_discovery import find_run_directories
from smartrain.workflows.testing.model_test_service import (
    complete_missing_test_artifacts,
    format_metrics_path,
    format_metrics_path_for_split,
    persist_target_test_artifacts_state,
    sync_test_artifacts_manifest,
)
from smartrain.core.runtime.run_artifacts import (
    canonical_run_model_path,
    canonicalize_run_ultralytics_layout,
    materialize_canonical_run_model,
    resolve_run_model_with_legacy_fallback,
    run_tmp_dir,
    run_tests_dir,
    run_test_backend_dir,
    run_train_backend_dir,
    ensure_run_layout,
)


MODEL_VERSION = "yolov8n"
EPOCHS = 50
BATCH = 16
IMG_SIZE = 640
_ULTRALYTICS_YAML_IGNORED_KEYS = frozenset(
    {
        "data",
        "project",
        "name",
        "exist_ok",
        # In Ultralytics, `cfg` can point to external YAML with hyperparameters,
        # but smart-train already reads the user-specified `--ultralytics_yaml`,
        # so `cfg` is often a "remainder" and can refer to a file,
        # which is not on the current machine.
        "cfg",
        # we set device through the environment/CLI; values ​​from saved args.yaml
        # (eg '0,1,2') often do not correspond to the available GPUs on the machine.
        "device",
        "model_dir",
        "target_path",
        "workspace",
    }
)
_MANUAL_MODEL_ENTRY = "<manual>"


def _bytes_to_gb(value: int | float | None) -> float | None:
    return _svc_bytes_to_gb(value)


def _linux_cpu_model_name() -> str | None:
    return _svc_linux_cpu_model_name()


def _linux_physical_core_count() -> int | None:
    return _svc_linux_physical_core_count()


def _linux_mem_total_bytes() -> int | None:
    return _svc_linux_mem_total_bytes()


def _resolve_mount_point(path: str) -> str:
    return _svc_resolve_mount_point(path)


def _linux_fs_type_for_mount(mount_point: str) -> str | None:
    return _svc_linux_fs_type_for_mount(mount_point)


def collect_system_profile(run_dir: str) -> dict[str, Any]:
    return _svc_collect_system_profile(run_dir)


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        description="Training models (without arguments, interactive mode starts)"
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Root workspace (otherwise {WORKSPACE_ENV_VAR}); runs in runs/, resolution --data by datasets",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="YAML profile smart-train (basic config). Can be mixed with --ultralytics_yaml; priority CLI > --ultralytics_yaml > --config",
    )
    parser.add_argument(
        "--ultralytics_yaml",
        type=str,
        default=None,
        help="External Ultralytics args.yaml; incompatible keys (data/project/name/exist_ok/...) are ignored with a warning",
    )
    parser.add_argument(
        "--base-run-args-yaml",
        type=str,
        default=None,
        help="Path to args.yaml of the base run (used as a source of defaults in interactive mode)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Directory with data.yaml (absolute/relative) or record name from datasets/datasets_info.json; "
        "with --workspace it is usually set explicitly (the data value from --ultralytics_yaml is not used)",
    )

    parser.add_argument(
        "--task",
        type=str,
        default=argparse.SUPPRESS,
        help="Ultralytics task: detect, segment, classify, pose, obb (default from profile or detect)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=argparse.SUPPRESS,
        help=(
            f"Model (default {MODEL_VERSION} or from profile --config). "
            "Specify full alias/weights including scale (n/s/m/l/x), e.g. yolo11x.pt."
        ),
    )
    parser.add_argument(
        "--external-provider",
        type=str,
        default=None,
        help="Use external provider id for training (runs via isolated provider venv).",
    )
    parser.add_argument(
        "--external-repo",
        type=str,
        default=None,
        help="Override external provider repository path.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Epoches (default {EPOCHS} or from profile)",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Batch (default {BATCH} or from profile)",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=argparse.SUPPRESS,
        help=f"imgsz (default {IMG_SIZE} or from profile)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=argparse.SUPPRESS,
        help="Compute device for training (e.g. 0, 1, cpu). Default: GPU 0 if available, otherwise cpu.",
    )

    parser.add_argument(
        "--target-path",
        type=str,
        default=None,
        help="Base directory for runs (defaults to workspace/runs when using workspace)",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to the folder with the model",
    )

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Perform testing only without training",
    )

    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        dest="non_interactive",
        help="Do not ask for confirmation if there is an existing results folder (for queue and CI)",
    )

    parser.add_argument(
        "--val-imgsz",
        type=int,
        default=None,
        help="Image size for val/test (default as --img-size when training)",
    )
    parser.add_argument(
        "--val-conf",
        type=float,
        default=None,
        help="conf threshold for val() (Ultralytics)",
    )
    parser.add_argument(
        "--val-iou",
        type=float,
        default=None,
        help="IoU threshold for val() (Ultralytics)",
    )
    parser.add_argument(
        "--val-batch",
        type=int,
        default=None,
        help="Batch for val/test (by default: as a training batch; for --test-only it is taken from training_metadata.json if available)",
    )
    parser.add_argument(
        "--conf-rec-beta-recall",
        type=float,
        default=2.0,
        help="Beta for objective B (recall-priority F-beta) in confidence recommendations.",
    )
    parser.add_argument(
        "--conf-rec-beta-precision",
        type=float,
        default=0.5,
        help="Beta for objective C (precision-priority F-beta) in confidence recommendations.",
    )
    parser.add_argument(
        "--conf-rec-fallback",
        type=float,
        default=0.25,
        help="Fallback confidence value when recommendations cannot be computed.",
    )
    parser.add_argument(
        "--conf-rec-disable",
        action="store_true",
        help="Disable confidence threshold recommendation computation.",
    )

    parser.add_argument(
        "--weighted-sampling",
        action="store_true",
        help="Weighted image sampling (classes with fewer objects more often); ultralytics patch",
    )

    parser.add_argument(
        "--clearml",
        action="store_true",
        help="Logging hyperparameters in ClearML (need pip install clearml)",
    )
    parser.add_argument(
        "--clearml-project",
        type=str,
        default=None,
        help="ClearML project name (aka CLEARML_PROJECT or smartrain)",
    )

    return parser


def parse_args(argv=None):
    return build_train_arg_parser().parse_args(argv)


def build_train_resume_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="smartrain train resume",
        description="Resume an interrupted training run",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR})",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Absolute or workspace-relative run directory to resume",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        dest="non_interactive",
        help="Non-interactive mode. Requires --run-dir.",
    )
    parser.add_argument(
        "--test-batch",
        type=int,
        default=None,
        help="Start batch for resume test stage (training_complete_test_pending).",
    )
    parser.add_argument(
        "--test-batch-min",
        type=int,
        default=1,
        help="Minimum batch for OOM backoff in resume test stage.",
    )
    parser.add_argument(
        "--test-batch-backoff",
        type=int,
        default=2,
        help="OOM backoff divider for test batch (e.g. 2 means batch/2).",
    )
    return parser


def build_train_calc_confidence_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="smartrain train calc-confidence",
        description="Calculate confidence recommendations for existing runs",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR})",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Absolute or workspace-relative run directory. Can be used multiple times.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all discovered run directories.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        dest="non_interactive",
        help="Non-interactive mode. If no --run-dir is given, all runs are processed.",
    )
    parser.add_argument(
        "--val-batch",
        type=int,
        default=None,
        help="Batch size for val/test recompute in calc-confidence (default: 1).",
    )
    return parser


def _resume_display_value(diag: RunDiagnosis) -> str:
    dataset_name = os.path.basename(os.path.dirname(diag.run_dir.rstrip(os.sep)))
    run_name = os.path.basename(diag.run_dir.rstrip(os.sep))
    reason = diag.reasons[0] if diag.reasons else "n/a"
    return f"{dataset_name}/{run_name} | {diag.status} | {reason}"


def _select_resume_candidate_interactive(candidates: list[RunDiagnosis]) -> RunDiagnosis | None:
    if not candidates:
        print("[ERROR] No incomplete runs found.")
        return None
    options = [_resume_display_value(d) for d in candidates]
    print_numbered_options("Incomplete run to resume", options)
    while True:
        raw = prompt_text("Choose run number", default="1").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
        print(f"[ERROR] Incorrect selection: {raw!r}")


def _select_runs_for_calc_confidence_interactive(run_dirs: list[str]) -> list[str]:
    if not run_dirs:
        return []
    print("\n[INFO] Available runs:")
    for idx, rd in enumerate(run_dirs, start=1):
        print(f"  {idx}. {rd}")
    raw = prompt_text(
        "Select runs by numbers (comma-separated) or 'all'",
        default="all",
    ).strip()
    if not raw or raw.lower() == "all":
        return run_dirs
    selected: list[str] = []
    seen: set[str] = set()
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        if not t.isdigit():
            raise ValueError(f"Invalid token {t!r}. Expected numbers or 'all'.")
        pos = int(t)
        if pos < 1 or pos > len(run_dirs):
            raise ValueError(f"Selection {pos} out of range 1..{len(run_dirs)}.")
        item = run_dirs[pos - 1]
        if item not in seen:
            seen.add(item)
            selected.append(item)
    return selected


def _resolve_run_dirs_for_calc_confidence(
    workspace_root: str,
    run_dir_args: list[str],
    select_all: bool,
    non_interactive: bool,
) -> list[str]:
    discovered = sorted(set(os.path.abspath(x) for x in find_run_directories(WorkspaceLayout(workspace_root).runs)))
    if run_dir_args:
        out: list[str] = []
        for raw in run_dir_args:
            rd = str(raw).strip()
            if not rd:
                continue
            if not os.path.isabs(rd):
                rd = os.path.join(WorkspaceLayout(workspace_root).runs, rd)
            out.append(os.path.abspath(rd))
        return sorted(set(out))
    if select_all or non_interactive:
        return discovered
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive mode requires a terminal (TTY).")
    return _select_runs_for_calc_confidence_interactive(discovered)


def _run_calc_confidence_command(argv: list[str]) -> int:
    args = build_train_calc_confidence_arg_parser().parse_args(argv)
    try:
        workspace_root = resolve_workspace_root(args.workspace)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1
    try:
        run_dirs = _resolve_run_dirs_for_calc_confidence(
            workspace_root=workspace_root,
            run_dir_args=list(getattr(args, "run_dir", []) or []),
            select_all=bool(getattr(args, "all", False)),
            non_interactive=bool(getattr(args, "non_interactive", False)),
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2
    if not run_dirs:
        print("[INFO] No runs selected.")
        return 0
    val_batch = getattr(args, "val_batch", None)
    if val_batch is None:
        if bool(getattr(args, "non_interactive", False)) or not sys.stdin.isatty():
            val_batch = 1
        else:
            raw_batch = prompt_text(
                "Val/Test batch for confidence recompute",
                default="1",
            ).strip()
            try:
                val_batch = int(raw_batch) if raw_batch else 1
            except ValueError:
                print(f"[ERROR] Invalid --val-batch value: {raw_batch!r}")
                return 2
    if int(val_batch) <= 0:
        print(f"[ERROR] --val-batch must be > 0, got: {val_batch}")
        return 2

    processed = 0
    updated = 0
    skipped = 0
    failed = 0
    for run_dir in run_dirs:
        processed += 1
        before_test = recommendations_complete(
            read_recommendation_file(recommendation_file_path(run_dir, "test"))
        )
        before_val = recommendations_complete(
            read_recommendation_file(recommendation_file_path(run_dir, "val"))
        )
        try:
            _ensure_resume_confidence_recommendations(run_dir, workspace_root, val_batch=int(val_batch))
        except Exception as e:
            failed += 1
            print(f"[ERROR] {run_dir}: {e}")
            continue
        after_test = recommendations_complete(
            read_recommendation_file(recommendation_file_path(run_dir, "test"))
        )
        after_val = recommendations_complete(
            read_recommendation_file(recommendation_file_path(run_dir, "val"))
        )
        if after_test and after_val and (not (before_test and before_val)):
            updated += 1
            print(f"[OK] {run_dir}: confidence recommendations computed.")
        elif before_test and before_val:
            skipped += 1
            print(f"[INFO] {run_dir}: recommendations already present.")
        else:
            skipped += 1
            print(f"[WARN] {run_dir}: recommendations still incomplete.")

    print(
        f"[INFO] calc-confidence summary: processed={processed}, "
        f"updated={updated}, skipped={skipped}, failed={failed}"
    )
    return 1 if failed > 0 else 0


def _run_resume_command(argv: list[str]) -> int:
    args = build_train_resume_arg_parser().parse_args(argv)
    try:
        workspace_root = resolve_workspace_root(args.workspace)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1

    if args.non_interactive and not args.run_dir:
        print("[ERROR] In non-interactive mode, --run-dir is required.")
        return 2
    if args.test_batch is not None and int(args.test_batch) <= 0:
        print(f"[ERROR] --test-batch must be > 0, got: {args.test_batch}")
        return 2
    if int(args.test_batch_min) <= 0:
        print(f"[ERROR] --test-batch-min must be > 0, got: {args.test_batch_min}")
        return 2
    if int(args.test_batch_backoff) <= 1:
        print(f"[ERROR] --test-batch-backoff must be > 1, got: {args.test_batch_backoff}")
        return 2

    candidates = list_incomplete_runs(workspace_root)
    chosen: RunDiagnosis | None = None
    if args.run_dir:
        run_dir = args.run_dir
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(WorkspaceLayout(workspace_root).runs, run_dir)
        chosen = next((d for d in candidates if os.path.abspath(run_dir) == d.run_dir), None)
        if chosen is None:
            from smartrain.workflows.training.train_resume import diagnose_run

            chosen = diagnose_run(run_dir)
    else:
        if not sys.stdin.isatty():
            print("[ERROR] Interactive resume mode requires a terminal (TTY).")
            return 1
        chosen = _select_resume_candidate_interactive(candidates)
        if chosen is None:
            return 1

    if chosen.status not in (
        RUN_STATUS_RESUMABLE_INCOMPLETE,
        RUN_STATUS_TRAINING_COMPLETE_TEST_PENDING,
    ):
        print(f"[ERROR] Run is not resumable: {chosen.run_dir}")
        print(f"[INFO] Status: {chosen.status}")
        print(f"[INFO] Reasons: {', '.join(chosen.reasons)}")
        return 2

    if chosen.status == RUN_STATUS_TRAINING_COMPLETE_TEST_PENDING:
        try:
            ensure_matplotlib_training_runtime(non_interactive=True)
            _maybe_free_cuda_memory()
            _complete_missing_test_with_backoff(
                chosen.run_dir,
                workspace_root=workspace_root,
                initial_batch=args.test_batch,
                min_batch=int(args.test_batch_min),
                backoff=int(args.test_batch_backoff),
            )
            print(f"[OK] Missing test stage completed: {chosen.run_dir}")
            return 0
        except Exception as e:
            from smartrain.workflows.training.train_resume import diagnose_run

            update_resume_test_metadata(
                chosen.run_dir,
                success=False,
                error=str(e),
                diagnosis=diagnose_run(chosen.run_dir),
            )
            print(f"[ERROR] Failed to complete missing test stage: {chosen.run_dir}")
            print(f"[ERROR] {e}")
            return 1

    try:
        from smartrain.workflows.training.train_resume import diagnose_run

        resume_training_in_run(chosen.run_dir)
        _ensure_resume_confidence_recommendations(chosen.run_dir, workspace_root)
        update_resume_metadata(
            chosen.run_dir,
            success=True,
            error=None,
            diagnosis=diagnose_run(chosen.run_dir),
        )
        print(f"[OK] Resume completed: {chosen.run_dir}")
        return 0
    except Exception as e:
        from smartrain.workflows.training.train_resume import diagnose_run

        update_resume_metadata(
            chosen.run_dir,
            success=False,
            error=str(e),
            diagnosis=diagnose_run(chosen.run_dir),
        )
        print(f"[ERROR] Failed to resume run: {chosen.run_dir}")
        print(f"[ERROR] {e}")
        return 1


def _is_cuda_oom_error(err: Exception) -> bool:
    return _svc_is_cuda_oom_error(err)


def _default_resume_test_batch(run_dir: str) -> int:
    return _svc_default_resume_test_batch(run_dir)


def _next_backoff_batch(current: int, min_batch: int, backoff: int) -> int:
    return _svc_next_backoff_batch(current, min_batch, backoff)


def _complete_missing_test_with_backoff(
    run_dir: str,
    *,
    workspace_root: str,
    initial_batch: int | None,
    min_batch: int,
    backoff: int,
) -> None:
    _svc_complete_missing_test_with_backoff(
        run_dir,
        workspace_root=workspace_root,
        initial_batch=initial_batch,
        min_batch=min_batch,
        backoff=backoff,
        complete_missing_test_artifacts_cb=complete_missing_test_artifacts,
        pt_test_runner_cb=_resume_ultralytics_pt_test_runner,
        update_metadata_cb=update_resume_test_metadata,
        maybe_free_cuda_memory_cb=_maybe_free_cuda_memory,
    )


def _ensure_resume_confidence_recommendations(
    run_dir: str,
    workspace_root: str,
    val_batch: int = 1,
) -> None:
    test_payload = read_recommendation_file(recommendation_file_path(run_dir, "test"))
    val_payload = read_recommendation_file(recommendation_file_path(run_dir, "val"))
    if recommendations_complete(test_payload) and recommendations_complete(val_payload):
        return

    best_pt = canonical_run_model_path(run_dir, ".pt")
    if not os.path.isfile(best_pt):
        print(
            "[WARN] Resume post-check: recommendations missing but canonical run model is absent; "
            "cannot recompute confidence recommendations."
        )
        return

    dataset_path = resolve_dataset_path_for_resume(run_dir, workspace_root)
    if not dataset_path:
        print(
            "[WARN] Resume post-check: recommendations missing but dataset path is unresolved; "
            "cannot recompute confidence recommendations."
        )
        return

    print("[INFO] Resume post-check: recomputing missing confidence recommendations (val/test).")
    _maybe_free_cuda_memory()
    # Recompute path is primarily for backfilling recommendations on existing runs.
    # Keep val/test memory profile conservative to avoid OOM on small GPUs.
    test_yolo(run_dir, dataset_path, val_batch=max(1, int(val_batch)), non_interactive=True)


def _prompt_input(label: str, default: str = "", completer=None, show_default_hint: bool = True) -> str:
    return _train_prompt_input(
        label=label,
        default=default,
        completer=completer,
        show_default_hint=show_default_hint,
    )


def _prompt_yes_no(label: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    default_text = "y" if default else "n"
    raw = _prompt_input(f"{label} [{suffix}]: ", default=default_text, show_default_hint=False).strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true", "yes", "d")


def _prompt_int(label: str, default: int) -> int:
    while True:
        raw = _prompt_input(f"{label}: ", default=str(default)).strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"[ERROR] Expected integer, received: {raw!r}")


def _prompt_optional_int(label: str, default: int | None = None) -> int | None:
    default_text = "" if default is None else str(default)
    while True:
        raw = _prompt_input(f"{label}: ", default=default_text).strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"[ERROR] Expecting an integer or empty value, received: {raw!r}")


def _prompt_optional_float(label: str, default: float | None = None) -> float | None:
    default_text = "" if default is None else str(default)
    while True:
        raw = _prompt_input(f"{label}: ", default=default_text).strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print(f"[ERROR] Expecting a number or empty value, received: {raw!r}")


def _prompt_train_device(default: str | None = None) -> str:
    return _train_prompt_device(default=default)


def _load_available_datasets(layout: WorkspaceLayout) -> list[str]:
    return _svc_load_available_datasets(layout)


def _prompt_dataset_name(available: list[str]) -> str:
    from smartrain.cli_support.cli_prompts import prompt_choice

    return _svc_prompt_dataset_name(available, prompt_choice_cb=prompt_choice)


def _train_model_picker_options(default_model: str) -> list[str]:
    return _svc_train_model_picker_options(
        default_model,
        installed_records_cb=_installed_external_provider_records,
        manual_model_entry=_MANUAL_MODEL_ENTRY,
    )


def _installed_external_provider_ids() -> list[str]:
    return _svc_installed_external_provider_ids(installed_records_cb=_installed_external_provider_records)


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


def _apply_external_provider_defaults(args) -> None:
    _svc_apply_external_provider_defaults(args, get_installed_record_cb=_get_installed_external_provider_record)


def _model_matches_task(alias: str, task: str) -> bool:
    return _svc_model_matches_task(alias, task)


def _format_numbered_columns(items: list[str], *, columns: int = 4) -> list[str]:
    return _svc_format_numbered_columns(items, columns=columns)


def _pick_model_interactive(options: list[str], default_alias: str) -> str:
    return _svc_pick_model_interactive(
        options,
        default_alias,
        format_columns_cb=_format_numbered_columns,
        prompt_input_cb=_prompt_input,
    )


def _extract_run_timestamp(run_name: str, run_dir: Path) -> datetime:
    return _svc_extract_run_timestamp(run_name, run_dir)


def _base_run_summary(args_yaml: Path) -> dict[str, str]:
    return _svc_base_run_summary(args_yaml, load_ultralytics_yaml_cb=_load_ultralytics_yaml)


def _collect_available_base_runs(layout: WorkspaceLayout, selected_dataset: str) -> list[dict[str, str]]:
    return _svc_collect_available_base_runs(
        layout,
        selected_dataset,
        base_run_summary_cb=_base_run_summary,
    )


def _print_available_base_runs(selected_dataset: str, runs: list[dict[str, str]]) -> None:
    _svc_print_available_base_runs(selected_dataset, runs)


def _prompt_base_run_args_yaml(runs: list[dict[str, str]], default_path: str | None = None) -> str | None:
    return _svc_prompt_base_run_args_yaml(
        runs,
        default_path,
        prompt_input_cb=_prompt_input,
    )


def _get_interactive_default(args, attr: str, fallback, baseline_cfg: dict[str, Any], baseline_key: str):
    return _svc_get_interactive_default(args, attr, fallback, baseline_cfg, baseline_key)


def _run_interactive_train_setup(args) -> bool:
    return _svc_run_interactive_train_setup(
        args,
        model_version=MODEL_VERSION,
        manual_model_entry=_MANUAL_MODEL_ENTRY,
        epochs_default=EPOCHS,
        batch_default=BATCH,
        img_size_default=IMG_SIZE,
        ultralytics_yaml_ignored_keys=_ULTRALYTICS_YAML_IGNORED_KEYS,
        resolve_workspace_root_cb=resolve_workspace_root,
        workspace_layout_cb=WorkspaceLayout,
        prompt_input_cb=_prompt_input,
        load_available_datasets_cb=_load_available_datasets,
        prompt_dataset_name_cb=_prompt_dataset_name,
        collect_available_base_runs_cb=_collect_available_base_runs,
        print_available_base_runs_cb=_print_available_base_runs,
        prompt_base_run_args_yaml_cb=_prompt_base_run_args_yaml,
        load_ultralytics_yaml_cb=_load_ultralytics_yaml,
        extract_smartrain_options_cb=extract_smartrain_options,
        normalize_model_spec_cb=_normalize_model_spec,
        train_model_picker_options_cb=_train_model_picker_options,
        model_matches_task_cb=_model_matches_task,
        pick_model_interactive_cb=_pick_model_interactive,
        installed_external_provider_ids_cb=_installed_external_provider_ids,
        prompt_int_cb=_prompt_int,
        prompt_train_device_cb=_prompt_train_device,
        prompt_yes_no_cb=_prompt_yes_no,
        prompt_optional_int_cb=_prompt_optional_int,
        prompt_optional_float_cb=_prompt_optional_float,
        default_device_value_cb=default_device_value,
    )


def resolve_training_data_path(layout: WorkspaceLayout, data_arg: str) -> str:
    return _svc_resolve_training_data_path(
        layout,
        data_arg,
        datasets_info_file=DATASETS_INFO_FILE,
        resolve_dataset_root_cb=resolve_dataset_root,
    )


def _validate_dataset_dir(dataset_path: str) -> None:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Yaml file not found: {data_yaml}")


def _split_dir_from_dataset_yaml(dataset_path: str, raw: dict, split_key: str) -> str | None:
    return _svc_split_dir_from_dataset_yaml(dataset_path, raw, split_key)


def _pick_split_relative_dir(dataset_path: str, split_aliases: tuple[str, ...]) -> str | None:
    return _svc_pick_split_relative_dir(dataset_path, split_aliases)


def _build_runtime_data_yaml(dataset_path: str, run_dir: str, *, stage: str) -> str:
    return _svc_build_runtime_data_yaml(
        dataset_path,
        run_dir,
        stage=stage,
        ensure_run_layout_cb=ensure_run_layout,
        run_tmp_dir_cb=run_tmp_dir,
    )


def _resolve_cli_paths_with_profile(args, u_cfg: dict) -> tuple[str | None, str, str]:
    return _svc_resolve_cli_paths_with_profile(
        args,
        u_cfg,
        workspace_env_var=WORKSPACE_ENV_VAR,
        resolve_workspace_root_cb=resolve_workspace_root,
        workspace_layout_cb=WorkspaceLayout,
        resolve_training_data_path_cb=resolve_training_data_path,
        resolve_profile_data_path_cb=resolve_profile_data_path,
        dataset_root_from_data_yaml_cb=dataset_root_from_data_yaml,
    )


def _finalize_train_kwargs(ultralytics_cfg: dict[str, Any], data_yaml: str, model_dir: str) -> dict[str, Any]:
    """Force Ultralytics train directory under ``model_dir`` (``train-ultralytics``, ``exist_ok=True``)."""
    return _svc_finalize_train_kwargs(ultralytics_cfg, data_yaml, model_dir)


def _load_ultralytics_yaml(path: str | None) -> dict[str, Any]:
    return _svc_load_ultralytics_yaml(path, load_train_profile_cb=load_train_profile)


def _normalize_model_spec(spec: Any, *, add_pt_when_missing: bool = False) -> str:
    return _svc_normalize_model_spec(
        spec,
        default_model=MODEL_VERSION,
        add_pt_when_missing=add_pt_when_missing,
    )


def _extract_effective_loaded_model(model: Any, fallback: str) -> str:
    return _svc_extract_effective_loaded_model(model, fallback)


def _extract_model_family_scale(spec: str) -> tuple[str, str] | None:
    return _svc_extract_model_family_scale(spec)


def _build_run_name(
    provider_id: str,
    model_version: str,
    epochs: int,
    batch: int,
    dataset_hash: str | None,
    *,
    timestamp: datetime | None = None,
) -> str:
    return _shared_build_run_name(
        provider_id,
        model_version,
        epochs,
        batch,
        dataset_hash,
        timestamp=timestamp,
    )


def build_run_name(
    provider_id: str,
    model_version: str,
    epochs: int,
    batch: int,
    dataset_hash: str | None,
    *,
    timestamp: datetime | None = None,
) -> str:
    """Public helper for train-service composition without private symbol access."""
    return _build_run_name(
        provider_id,
        model_version,
        epochs,
        batch,
        dataset_hash,
        timestamp=timestamp,
    )


def _normalize_external_run_layout(run_dir: str) -> None:
    _shared_normalize_external_run_layout(run_dir)


def _materialize_canonical_run_model(run_dir: str, source_path: str | None = None) -> str | None:
    target = materialize_canonical_run_model(
        run_dir,
        ext=".pt",
        source_path=source_path,
        move=True,
        normalize_metadata=True,
    )
    return str(target) if target is not None else None


def _find_external_best_checkpoint(run_dir: str) -> str | None:
    found = resolve_run_model_with_legacy_fallback(run_dir, ".pt")
    return str(found) if found is not None else None


def _ensure_external_best_checkpoint_layout(run_dir: str) -> str | None:
    return _shared_ensure_external_best_checkpoint_layout(run_dir)


def _resolve_external_eval_source(dataset_path: str) -> str:
    return _shared_resolve_external_eval_source(dataset_path)


def resolve_external_eval_source(dataset_path: str) -> str:
    """Public helper for external eval source resolution."""
    return _resolve_external_eval_source(dataset_path)


def _write_external_fallback_metrics(model_dir: str, *, provider_id: str, rc: int) -> str:
    return _shared_write_external_fallback_metrics(model_dir, provider_id=provider_id, rc=rc)


def _run_mfel_external_val_fallback(
    *,
    repo_path: str,
    venv_path: str,
    model_path: str,
    data_yaml: str,
    model_dir: str,
    imgsz: int,
    conf: float | None,
    iou: float | None,
    batch: int | None,
    device: str | None,
) -> int:
    return _shared_run_mfel_external_val_fallback(
        repo_path=repo_path,
        venv_path=venv_path,
        model_path=model_path,
        data_yaml=data_yaml,
        model_dir=model_dir,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        batch=batch,
        device=device,
    )


def _merge_sources_with_priority(
    *,
    config_profile: dict[str, Any],
    ultralytics_profile: dict[str, Any],
    args: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Base: --config
    u_cfg, sm_opts = extract_smartrain_options(config_profile)

    # Overlay: --ultralytics_yaml (minus ignored keys)
    if ultralytics_profile:
        ignored = sorted(k for k in ultralytics_profile.keys() if k in _ULTRALYTICS_YAML_IGNORED_KEYS)
        if ignored:
            print(
                "[WARNING] --ultralytics_yaml: keys ignored: "
                + ", ".join(ignored)
            )
        filtered = {k: v for k, v in ultralytics_profile.items() if k not in _ULTRALYTICS_YAML_IGNORED_KEYS}
        u_from_ultra, sm_from_ultra = extract_smartrain_options(filtered)
        cli_key_map = {
            "model": "model",
            "epochs": "epochs",
            "batch": "batch",
            "imgsz": "img_size",
            "task": "task",
        }
        overridden_by_cli: list[str] = []
        for yaml_key, cli_attr in cli_key_map.items():
            if yaml_key in u_from_ultra and getattr(args, cli_attr, None) is not None:
                overridden_by_cli.append(yaml_key)
        if overridden_by_cli:
            print(
                "[WARNING] --ultralytics_yaml: The following keys will be overridden by the CLI: "
                + ", ".join(sorted(overridden_by_cli))
            )
        u_cfg.update(u_from_ultra)
        sm_opts.update(sm_from_ultra)
    return u_cfg, sm_opts


def train_yolo(
    dataset_path: str,
    target_dir: str,
    non_interactive: bool = False,
    workspace_root: str | None = None,
    ultralytics_cfg: dict[str, Any] | None = None,
    smartrain_opts: dict[str, Any] | None = None,
):
    mpl_rt = ensure_matplotlib_training_runtime(non_interactive=non_interactive)
    ultralytics_cfg = ultralytics_cfg or {}
    smartrain_opts = smartrain_opts or {}

    training_start_time = datetime.now()
    _validate_dataset_dir(dataset_path)

    dataset_name = os.path.basename(os.path.normpath(dataset_path))

    model_version = _normalize_model_spec(
        ultralytics_cfg.get("model", MODEL_VERSION),
        add_pt_when_missing=True,
    )
    ultralytics_cfg["model"] = model_version
    epochs = int(ultralytics_cfg.get("epochs", EPOCHS))

    try:
        dataset_hash = calculate_dataset_hash(dataset_path)
        print(f"[INFO] Dataset hash: {dataset_hash}")
    except Exception as e:
        print(f"[WARNING] Failed to calculate dataset hash: {e}")
        dataset_hash = None

    batch = int(ultralytics_cfg.get("batch", BATCH))
    img_size = int(ultralytics_cfg.get("imgsz", IMG_SIZE))
    folder_name = _build_run_name(
        "ultralytics",
        model_version,
        epochs,
        batch,
        dataset_hash,
        timestamp=training_start_time,
    )

    model_dir = os.path.join(target_dir, dataset_name, folder_name)

    if os.path.exists(model_dir):
        if non_interactive:
            print(f"[INFO] The folder already exists, continue without prompting: {model_dir}")
        else:
            while True:
                answer = input(
                    f"[WARNING] A folder with the same name already exists: {model_dir}. Continue training? (y/n): \n"
                ).strip().lower()
                if answer == "y":
                    break
                elif answer == "n":
                    sys.exit(1)
                else:
                    print("Please enter 'y' or 'n' only.\n")
    else:
        os.makedirs(model_dir, exist_ok=True)

    data_yaml = _build_runtime_data_yaml(dataset_path, model_dir, stage="train")
    train_kw = _finalize_train_kwargs(ultralytics_cfg, data_yaml, model_dir)
    if non_interactive or mpl_rt.force_ultralytics_plots_false:
        train_kw.setdefault("plots", False)
    _ensure_initial_training_metadata(
        model_dir=model_dir,
        dataset_path=dataset_path,
        model_version=model_version.replace(".pt", ""),
        epochs=epochs,
        batch=batch,
        img_size=img_size,
        training_start_time=training_start_time,
        dataset_hash=dataset_hash,
        workspace_root=workspace_root,
        task_type=task_to_metadata_task_type(train_kw.get("task")),
    )

    clearml_task = None
    if smartrain_opts.get("clearml"):
        try:
            from clearml import Task
        except ImportError as e:
            raise ImportError(
                "For --clearml, install: pip install 'smartrain[clearml]' or pip install clearml"
            ) from e
        cm_proj = (
            smartrain_opts.get("clearml_project")
            or os.environ.get("CLEARML_PROJECT")
            or "smartrain"
        )
        clearml_task = Task.init(
            project_name=cm_proj,
            task_name=os.path.basename(model_dir),
            task_type=Task.TaskTypes.training,
        )
        train_kw = clearml_task.connect(train_kw)

    if smartrain_opts.get("weighted_sampling"):
        from smartrain.workflows.datasets.weighted_yolo_dataset import setup_weighted_sampling_env

        setup_weighted_sampling_env()

    print("\n" + "=" * 60)
    print(f"[INFO] Training models: {model_kw_model(train_kw)}")
    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Task: {train_kw.get('task', 'detect')}")
    print(f"[INFO] Configuration: {data_yaml}")
    print(f"[INFO] Saving results in {model_dir}")
    print("=" * 60 + "\n")

    requested_model = _normalize_model_spec(train_kw.get("model", model_version), add_pt_when_missing=True)
    train_kw["model"] = requested_model
    model = YOLO(requested_model)
    loaded_model = _extract_effective_loaded_model(model, fallback=requested_model)
    print(f"[INFO] Requested model: {requested_model}")
    print(f"[INFO] Loaded model: {loaded_model}")

    req_fs = _extract_model_family_scale(requested_model)
    loaded_fs = _extract_model_family_scale(loaded_model)
    if req_fs and loaded_fs and req_fs != loaded_fs:
        mismatch_msg = (
            "[ERROR] Model family/scale mismatch: "
            f"requested {req_fs[0]}{req_fs[1]}, loaded {loaded_fs[0]}{loaded_fs[1]}. "
            "Silent model replacement is blocked."
        )
        if non_interactive:
            raise RuntimeError(mismatch_msg)
        while True:
            answer = input(f"{mismatch_msg} Continue anyway? (y/n): ").strip().lower()
            if answer == "y":
                print("[WARNING] User confirmed training with mismatched model.")
                break
            if answer == "n":
                raise RuntimeError("Training aborted by user due to model mismatch.")
            print("Please enter 'y' or 'n' only.\n")

    if smartrain_opts.get("weighted_sampling"):
        from smartrain.workflows.datasets.weighted_yolo_dataset import register_weighted_sampling_callback

        register_weighted_sampling_callback(model)

    training_end_time = None
    canonical_best_path = canonical_run_model_path(model_dir, ".pt")
    model_path = canonical_best_path
    try:
        model.train(**train_kw)
        training_end_time = datetime.now()
    except Exception as e:
        training_end_time = datetime.now()
        print(
            f"[ERROR] Training failed ({model_version}, dataset {dataset_name}, {epochs} epochs): {e}"
        )
    finally:
        if clearml_task is not None:
            try:
                clearml_task.close()
            except Exception:
                pass
        try:
            canonicalize_run_ultralytics_layout(model_dir)
        except Exception:
            pass

    try:
        best_src = resolve_run_model_with_legacy_fallback(model_dir)
        materialized = _materialize_canonical_run_model(
            model_dir,
            source_path=str(best_src) if best_src is not None else None,
            move=True,
            normalize_metadata=True,
        )
        model_path = str(materialized) if materialized is not None else canonical_best_path
        print("\n" + "-" * 60)
        if os.path.exists(model_path):
            print("[OK] Training complete.")
            print(f"[INFO] Model saved at path:\n{model_path}")
    except Exception:
        pass

    weights_dest = resolve_run_model_with_legacy_fallback(model_dir)
    training_weights_ok = weights_dest is not None and weights_dest.is_file()

    meta_extras = {
        "train_kw": {k: v for k, v in train_kw.items() if k != "data"},
        "task_type": task_to_metadata_task_type(train_kw.get("task")),
        "training_ok": training_weights_ok,
        "mpl_runtime": mpl_rt.as_dict(),
    }
    return model_dir, training_start_time, training_end_time, dataset_hash, workspace_root, meta_extras


def model_kw_model(train_kw: dict) -> str:
    return str(train_kw.get("model", ""))


def _resume_ultralytics_pt_test_runner(
    model_dir: str,
    dataset_path: str,
    training_start_time=None,
    training_end_time=None,
    train_img_size=None,
    val_imgsz=None,
    val_conf=None,
    val_iou=None,
    val_batch=None,
    conf_rec_disable: bool = False,
    conf_rec_beta_recall: float = 2.0,
    conf_rec_beta_precision: float = 0.5,
    conf_rec_fallback: float = 0.25,
    *,
    non_interactive: bool = False,
):
    """Resume-time PT test: same artifacts as `smartrain test` (Ultralytics val + plots)."""
    from smartrain.orchestrators.canonical_gateway import resolve_task_context
    from smartrain.workflows.testing.model_test_backends import run_ultralytics_backend

    mpl_rt = ensure_matplotlib_training_runtime(non_interactive=bool(non_interactive))
    data_yaml = _build_runtime_data_yaml(dataset_path, model_dir, stage="test")
    weights = canonical_run_model_path(model_dir, ".pt")
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"canonical run model is missing: {weights}")
    try:
        ctx = resolve_task_context(model_dir)
        t_raw = (ctx.task_type or "detect").strip().lower()
    except Exception:
        t_raw = "detect"
    if t_raw in {"classification", "classify", "cls"}:
        task_type = "classification"
    elif t_raw in {"segmentation", "segment", "seg"}:
        task_type = "segmentation"
    else:
        task_type = "detection"

    res = run_ultralytics_backend(
        root_dir=model_dir,
        weights_path=weights,
        dataset_yaml_path=data_yaml,
        format_name="pt",
        imgsz=val_imgsz if val_imgsz is not None else train_img_size,
        val_conf=val_conf,
        val_iou=val_iou,
        val_batch=val_batch,
        conf_rec_disable=bool(conf_rec_disable),
        conf_rec_beta_recall=float(conf_rec_beta_recall),
        conf_rec_beta_precision=float(conf_rec_beta_precision),
        conf_rec_fallback=float(conf_rec_fallback),
        deep_diagnostics=False,
        collect_performance=False,
        perf_warmup_images=5,
        runtime_device=None,
        task_type=task_type,
    )
    if not res.success:
        raise RuntimeError(str(res.error or "run_ultralytics_backend failed"))

    inference_record: dict[str, Any] = {
        "imgsz": val_imgsz if val_imgsz is not None else train_img_size,
        "conf": val_conf,
        "iou": val_iou,
        "batch": val_batch,
        "matplotlib_runtime": mpl_rt.as_dict(),
    }
    if isinstance(res.inference, dict):
        inference_record.update(res.inference)
    return res.test_start_time or datetime.now(), res.test_end_time or datetime.now(), inference_record


def test_yolo(
    model_dir,
    dataset_path,
    training_start_time=None,
    training_end_time=None,
    train_img_size=None,
    val_imgsz=None,
    val_conf=None,
    val_iou=None,
    val_batch=None,
    conf_rec_disable: bool = False,
    conf_rec_beta_recall: float = 2.0,
    conf_rec_beta_precision: float = 0.5,
    conf_rec_fallback: float = 0.25,
    *,
    non_interactive: bool = False,
):
    mpl_rt = ensure_matplotlib_training_runtime(non_interactive=non_interactive)
    test_start_time = datetime.now()

    data_yaml = _build_runtime_data_yaml(dataset_path, model_dir, stage="test")
    imgsz = val_imgsz if val_imgsz is not None else train_img_size

    inference_record = {
        "imgsz": imgsz,
        "conf": val_conf,
        "iou": val_iou,
        "batch": val_batch,
        "matplotlib_runtime": mpl_rt.as_dict(),
    }

    model_path = canonical_run_model_path(model_dir, ".pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"canonical run model is missing: {model_path}")
    trained_model = YOLO(model_path)

    val_kwargs = {
        "data": data_yaml,
        "split": "test",
        "project": str(run_tests_dir(model_dir)),
        "name": "test-ultralytics",
        "exist_ok": True,
        "plots": False,
        "save": False,
    }
    if imgsz is not None:
        val_kwargs["imgsz"] = imgsz
    if val_conf is not None:
        val_kwargs["conf"] = val_conf
    if val_iou is not None:
        val_kwargs["iou"] = val_iou
    if val_batch is not None:
        val_kwargs["batch"] = int(val_batch)

    print("\n" + "=" * 60)
    print(f"[INFO] Model testing: {model_dir}")
    print(f"[INFO] Dataset: {dataset_path}")
    print(f"[INFO] Configuration: {data_yaml}")
    print(f"[INFO] Saving results in {model_dir}")
    if imgsz is not None:
        print(f"[INFO] val imgsz={imgsz}, batch={val_batch}, conf={val_conf}, iou={val_iou}")
    print("=" * 60 + "\n")

    test_end_time = None
    try:
        result = trained_model.val(**val_kwargs)

        if not conf_rec_disable:
            _ensure_confidence_recommendations(
                trained_model=trained_model,
                primary_test_result=result,
                model_dir=model_dir,
                data_yaml=data_yaml,
                imgsz=imgsz,
                val_conf=val_conf,
                val_iou=val_iou,
                val_batch=val_batch,
                beta_recall=conf_rec_beta_recall,
                beta_precision=conf_rec_beta_precision,
                fallback_confidence=conf_rec_fallback,
            )

        test_end_time = datetime.now()
        csv_file = save_metrics_csv(result, model_dir)
        target_pt = model_path if os.path.isfile(model_path) else None
        persist_target_test_artifacts_state(
            model_dir,
            format_name="pt",
            target_path=target_pt,
            backend="ultralytics",
            status="ok" if os.path.exists(csv_file) else "incomplete",
        )

        print("\n" + "-" * 60)
        if os.path.exists(csv_file):
            print("[OK] Testing complete.")
            print(f"[INFO] Results saved at path:\n{csv_file}")
        else:
            print("[ERROR].csv file not found. Check Ultralytics log.")
        print("-" * 60 + "\n")
    except Exception as e:
        test_end_time = datetime.now()
        print(f"[ERROR] Failed to test {model_dir} on dataset {dataset_path}: {e}")
        raise
    finally:
        try:
            canonicalize_run_ultralytics_layout(model_dir)
        except Exception:
            pass

    return test_start_time, test_end_time, inference_record


def _recommendation_summary_for_metadata(model_dir: str) -> dict[str, Any] | None:
    out: dict[str, Any] = {"files": {}, "status": {}}
    found = False
    for split in ("val", "test"):
        p = recommendation_file_path(model_dir, split)
        payload = read_recommendation_file(p)
        if not isinstance(payload, dict):
            continue
        found = True
        out["files"][split] = os.path.basename(p)
        out["status"][split] = payload.get("status")
    return out if found else None


def _ensure_confidence_recommendations(
    *,
    trained_model: Any,
    primary_test_result: Any,
    model_dir: str,
    data_yaml: str,
    imgsz: int | None,
    val_conf: float | None,
    val_iou: float | None,
    val_batch: int | None,
    beta_recall: float,
    beta_precision: float,
    fallback_confidence: float,
) -> None:
    test_path = recommendation_file_path(model_dir, "test")
    val_path = recommendation_file_path(model_dir, "val")
    has_test = recommendations_complete(read_recommendation_file(test_path))
    has_val = recommendations_complete(read_recommendation_file(val_path))
    if has_test and has_val:
        print("[INFO] Confidence recommendations already exist (val/test), skipping recompute.")
        return

    if not has_test:
        test_payload = compute_confidence_recommendations(
            primary_test_result,
            split="test",
            beta_recall=float(beta_recall),
            beta_precision=float(beta_precision),
            fallback_confidence=float(fallback_confidence),
        )
        write_recommendation_file(test_path, test_payload)
        print(f"[OK] Confidence recommendations (test): {test_path}")

    if has_val:
        return

    val_kwargs: dict[str, Any] = {
        "data": data_yaml,
        "split": "val",
        "plots": False,
        "save": False,
        "verbose": False,
        "project": str(run_tests_dir(model_dir)),
        "name": "val-ultralytics-recs",
        "exist_ok": True,
    }
    if imgsz is not None:
        val_kwargs["imgsz"] = imgsz
    if val_conf is not None:
        val_kwargs["conf"] = val_conf
    if val_iou is not None:
        val_kwargs["iou"] = val_iou
    if val_batch is not None:
        val_kwargs["batch"] = int(val_batch)
    try:
        val_result = trained_model.val(**val_kwargs)
        try:
            with open(format_metrics_path_for_split(model_dir, "val", "pt"), "w", encoding="utf-8") as f:
                f.write(val_result.to_csv())
        except Exception:
            pass
        val_payload = compute_confidence_recommendations(
            val_result,
            split="val",
            beta_recall=float(beta_recall),
            beta_precision=float(beta_precision),
            fallback_confidence=float(fallback_confidence),
        )
        write_recommendation_file(val_path, val_payload)
        print(f"[OK] Confidence recommendations (val): {val_path}")
    except Exception as exc:
        write_not_available_recommendations(
            model_dir=model_dir,
            split="val",
            reason=f"val_split_failed: {exc}",
            beta_recall=float(beta_recall),
            beta_precision=float(beta_precision),
            fallback_confidence=float(fallback_confidence),
        )
        print(f"[WARN] Failed to compute val confidence recommendations: {exc}")
    try:
        canonicalize_run_ultralytics_layout(model_dir)
    except Exception:
        pass


def save_metrics_csv(test_result, model_dir):
    csv_file = os.path.join(str(run_tests_dir(model_dir)), "test_metrics.csv")

    csv_data = test_result.to_csv()
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(csv_data)

    return csv_file


def _relative_to_workspace(path: str, workspace_root: str) -> str:
    return _svc_relative_to_workspace(path, workspace_root)


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    _svc_write_json_atomic(path, payload)


def _ensure_initial_training_metadata(
    *,
    model_dir: str,
    dataset_path: str,
    model_version: str,
    epochs: int,
    batch: int,
    img_size: int,
    training_start_time: datetime,
    dataset_hash: str | None,
    workspace_root: str | None,
    task_type: str,
) -> None:
    metadata_file = os.path.join(model_dir, "training_metadata.json")
    payload: dict[str, Any] = {}
    if os.path.isfile(metadata_file):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                payload = existing
        except Exception:
            payload = {}

    ti = payload.setdefault("training_info", {})
    if not isinstance(ti, dict):
        ti = {}
        payload["training_info"] = ti
    ti.setdefault("framework", "ultralytics")
    provider = ti.setdefault("provider", {})
    if not isinstance(provider, dict):
        provider = {}
        ti["provider"] = provider
    provider.setdefault("type", "builtin")
    provider.setdefault("id", "ultralytics")
    ti.setdefault("task_type", task_type or "detection")
    ti.setdefault("model", model_version)
    ds = ti.setdefault("dataset", {})
    if not isinstance(ds, dict):
        ds = {}
        ti["dataset"] = ds
    ds.setdefault("name", os.path.basename(os.path.normpath(dataset_path)))
    ds.setdefault("path_relative", _get_relative_path(dataset_path, model_dir))
    ds.setdefault("hash", dataset_hash)
    hp = ti.setdefault("hyperparameters", {})
    if not isinstance(hp, dict):
        hp = {}
        ti["hyperparameters"] = hp
    hp.setdefault("epochs", epochs)
    hp.setdefault("batch_size", batch)
    hp.setdefault("image_size", img_size)

    ts = payload.setdefault("timestamps", {})
    if not isinstance(ts, dict):
        ts = {}
        payload["timestamps"] = ts
    tr_ts = ts.setdefault("training", {})
    if not isinstance(tr_ts, dict):
        tr_ts = {}
        ts["training"] = tr_ts
    tr_ts.setdefault("start", training_start_time.isoformat())
    tr_ts.setdefault("end", None)
    tr_ts.setdefault("duration_seconds", None)
    te_ts = ts.setdefault("testing", {})
    if not isinstance(te_ts, dict):
        te_ts = {}
        ts["testing"] = te_ts
    te_ts.setdefault("start", None)
    te_ts.setdefault("end", None)
    te_ts.setdefault("duration_seconds", None)

    status = payload.setdefault("status", {})
    if not isinstance(status, dict):
        status = {}
        payload["status"] = status
    tr_status = status.setdefault("training", {})
    if not isinstance(tr_status, dict):
        tr_status = {}
        status["training"] = tr_status
    tr_status.setdefault("success", None)
    tr_status.setdefault("error", None)
    te_status = status.setdefault("testing", {})
    if not isinstance(te_status, dict):
        te_status = {}
        status["testing"] = te_status
    te_status.setdefault("success", None)
    te_status.setdefault("error", None)

    paths = payload.setdefault("paths", {})
    if not isinstance(paths, dict):
        paths = {}
        payload["paths"] = paths
    paths.setdefault("model_directory", ".")
    paths.setdefault("best_model", None)

    if workspace_root is not None:
        wb = payload.setdefault("workspace", {})
        if not isinstance(wb, dict):
            wb = {}
            payload["workspace"] = wb
        wb.setdefault("root", ".")
        wb.setdefault("dataset_path_relative", _relative_to_workspace(dataset_path, workspace_root))
        wb.setdefault("run_directory_relative", _relative_to_workspace(model_dir, workspace_root))

    _write_json_atomic(metadata_file, payload)


def save_training_metadata(
    model_dir,
    dataset_path,
    model_version=None,
    training_start_time=None,
    training_end_time=None,
    test_start_time=None,
    test_end_time=None,
    epochs=None,
    batch=None,
    img_size=None,
    training_success=True,
    training_error=None,
    test_success=True,
    test_error=None,
    dataset_hash=None,
    inference=None,
    workspace_root=None,
    task_type=None,
    ultralytics_train_summary=None,
    training_provider: str = "ultralytics",
    external_provider_id: str | None = None,
    system_profile: dict[str, Any] | None = None,
    matplotlib_runtime: dict[str, Any] | None = None,
    confidence_recommendation_config: dict[str, Any] | None = None,
):
    ds_abs = os.path.abspath(dataset_path)
    dataset_block: dict[str, Any] = {
        "name": os.path.basename(os.path.normpath(dataset_path)),
        "path_relative": _get_relative_path(dataset_path, model_dir),
        "hash": dataset_hash,
    }
    if workspace_root is not None:
        wr_abs = os.path.abspath(workspace_root)
        if ds_abs == wr_abs or ds_abs.startswith(wr_abs + os.sep):
            rel_uw = relativize_if_under(workspace_root, ds_abs)
            if rel_uw is not None:
                dataset_block["path_under_workspace"] = rel_uw
        else:
            dataset_block["path_absolute"] = ds_abs
    else:
        dataset_block["path_absolute"] = ds_abs

    metadata = {
        "training_info": {
            "framework": "ultralytics" if training_provider == "ultralytics" else "external",
            "provider": {
                "type": "builtin" if training_provider == "ultralytics" else "external",
                "id": external_provider_id if training_provider != "ultralytics" else "ultralytics",
            },
            "task_type": task_type or "detection",
            "model": model_version,
            "dataset": dataset_block,
            "hyperparameters": {
                "epochs": epochs,
                "batch_size": batch,
                "image_size": img_size,
            },
        },
        "timestamps": {
            "training": {
                "start": training_start_time.isoformat() if training_start_time else None,
                "end": training_end_time.isoformat() if training_end_time else None,
                "duration_seconds": (training_end_time - training_start_time).total_seconds()
                if training_start_time and training_end_time
                else None,
            },
            "testing": {
                "start": test_start_time.isoformat() if test_start_time else None,
                "end": test_end_time.isoformat() if test_end_time else None,
                "duration_seconds": (test_end_time - test_start_time).total_seconds()
                if test_start_time and test_end_time
                else None,
            },
        },
        "status": {
            "training": {
                "success": training_success,
                "error": training_error,
            },
            "testing": {
                "success": test_success,
                "error": test_error,
            },
        },
        "paths": {
            "model_directory": ".",
            "best_model": os.path.basename(canonical_run_model_path(model_dir, ".pt"))
            if os.path.exists(canonical_run_model_path(model_dir, ".pt"))
            else None,
        },
    }

    if ultralytics_train_summary:
        metadata["training_info"]["ultralytics_train"] = ultralytics_train_summary

    if workspace_root is not None:
        metadata["workspace"] = {
            "root": ".",
            "dataset_path_relative": _relative_to_workspace(dataset_path, workspace_root),
            "run_directory_relative": _relative_to_workspace(model_dir, workspace_root),
        }

    if inference:
        metadata["inference"] = {k: v for k, v in inference.items() if v is not None}
    sp_out: dict[str, Any] = dict(system_profile) if system_profile else {}
    if matplotlib_runtime:
        sp_out["matplotlib_runtime"] = matplotlib_runtime
    if sp_out:
        metadata["system_profile"] = sp_out
    rec_summary = _recommendation_summary_for_metadata(model_dir)
    if rec_summary:
        if confidence_recommendation_config:
            rec_summary["config"] = confidence_recommendation_config
        metadata["recommendations"] = {"confidence": rec_summary}
    test_manifest = sync_test_artifacts_manifest(
        model_dir,
        target_by_format={"pt": metadata["paths"].get("best_model")},
        backend_by_format={"pt": "ultralytics"},
    )
    formats_payload = test_manifest.get("formats")
    if isinstance(formats_payload, dict) and formats_payload:
        metadata["test_artifacts_by_format"] = formats_payload

    metadata_file = os.path.join(model_dir, "training_metadata.json")

    try:
        _write_json_atomic(metadata_file, metadata)
        print(f"[INFO] Training metadata saved: {metadata_file}")
    except Exception as e:
        print(f"[WARNING] Failed to save metadata: {e}")


def _get_relative_path(target_path, base_path):
    return _svc_get_relative_path(str(target_path), str(base_path))


def _json_safe_train_summary(train_kw: dict[str, Any] | None) -> dict[str, Any] | None:
    return _shared_json_safe_train_summary(train_kw)


def json_safe_train_summary(train_kw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Public helper for serializing train kwargs into metadata-safe payload."""
    return _json_safe_train_summary(train_kw)


def _load_batch_from_training_metadata(model_dir: str) -> int | None:
    """
    In --test-only mode we want to test with the same batch that was used during training.
    We take it from training_metadata.json if the file exists and the format is expected.
    """
    return _shared_load_batch_from_training_metadata(model_dir)


def load_batch_from_training_metadata(model_dir: str) -> int | None:
    """Public helper for retrieving saved batch in test-only flow."""
    return _load_batch_from_training_metadata(model_dir)


def _maybe_free_cuda_memory() -> None:
    _shared_maybe_free_cuda_memory()


def _ensure_device_available_or_raise(device: str | None) -> None:
    validate_device_available(device)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    request = make_command_request("train", argv, interactive_allowed=is_interactive_allowed(argv))
    aux_rc = _svc_handle_aux_train_commands(
        argv,
        run_resume_command_cb=_run_resume_command,
        run_calc_confidence_command_cb=_run_calc_confidence_command,
    )
    if aux_rc is not None:
        return aux_rc
    return _svc_run_train_cli_pipeline(
        argv,
        request=request,
        parse_args_cb=parse_args,
        apply_external_provider_defaults_cb=_apply_external_provider_defaults,
        list_provider_specs_cb=list_provider_specs,
        parse_external_model_ref_cb=parse_external_model_ref,
        validate_external_model_ref_cb=validate_external_model_ref,
        build_train_arg_parser_cb=build_train_arg_parser,
        run_interactive_train_setup_cb=_run_interactive_train_setup,
        emit_replay_cb=emit_replay,
        load_train_profile_cb=load_train_profile,
        load_ultralytics_yaml_cb=_load_ultralytics_yaml,
        merge_sources_with_priority_cb=_merge_sources_with_priority,
        merge_cli_into_ultralytics_cfg_cb=merge_cli_into_ultralytics_cfg,
        apply_cli_smartrain_overrides_cb=apply_cli_smartrain_overrides,
        resolve_device_request_cb=resolve_device_request,
        resolve_cli_paths_with_profile_cb=_resolve_cli_paths_with_profile,
        normalize_model_spec_cb=_normalize_model_spec,
        ensure_device_available_or_raise_cb=_ensure_device_available_or_raise,
        device_display_name_cb=device_display_name,
        run_train_after_setup_cb=run_train_after_setup,
        default_device_value_cb=default_device_value,
        model_version_default=MODEL_VERSION,
        epochs_default=EPOCHS,
        batch_default=BATCH,
        img_size_default=IMG_SIZE,
    )


if __name__ == "__main__":
    main()
