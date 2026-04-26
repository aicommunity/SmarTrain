import copy
import json
import os
import argparse
import platform
import re
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
from ultralytics import YOLO

from smartrain.cli_argparse import CliArgumentParser
from smartrain.cli_prompts import print_numbered_options, prompt_text
from smartrain.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.dataset_hash import calculate_dataset_hash
from smartrain.interactive_contract import is_interactive_allowed
from smartrain.train_resume import (
    RUN_STATUS_RESUMABLE_INCOMPLETE,
    RUN_STATUS_TRAINING_COMPLETE_TEST_PENDING,
    RunDiagnosis,
    resolve_dataset_path_for_resume,
    list_incomplete_runs,
    resume_training_in_run,
    update_resume_test_metadata,
    update_resume_metadata,
)
from smartrain.train_profile import (
    apply_cli_smartrain_overrides,
    dataset_root_from_data_yaml,
    extract_smartrain_options,
    load_train_profile,
    merge_cli_into_ultralytics_cfg,
    resolve_profile_data_path,
    task_to_metadata_task_type,
)
from smartrain.device_selector import default_device_value, discover_device_options, is_cuda_device
from smartrain.train_model_catalog import (
    TrainModelCatalog,
    is_supported_external_provider_model,
)
from smartrain.train_model_resolver import TrainModelResolver
from smartrain.provider_global_index import (
    get_provider_location,
    list_provider_records,
    reconcile_stale_provider_paths,
)
from smartrain.external_providers.runner import run_external_infer, run_external_train
from smartrain.external_model_ref import parse_external_model_ref, validate_external_model_ref
from smartrain.external_providers.registry import list_provider_specs
from smartrain.path_portable import relativize_if_under
from smartrain.workspace_paths import (
    WORKSPACE_ENV_VAR,
    WorkspaceLayout,
    resolve_workspace_root,
    resolve_dataset_root,
    DATASETS_INFO_FILE,
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
    if value is None:
        return None
    try:
        return round(float(value) / (1024.0**3), 3)
    except (TypeError, ValueError):
        return None


def _linux_cpu_model_name() -> str | None:
    cpuinfo = "/proc/cpuinfo"
    if not os.path.isfile(cpuinfo):
        return None
    try:
        with open(cpuinfo, "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        value = parts[1].strip()
                        if value:
                            return value
    except Exception:
        return None
    return None


def _linux_physical_core_count() -> int | None:
    cpuinfo = "/proc/cpuinfo"
    if not os.path.isfile(cpuinfo):
        return None
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    try:
        with open(cpuinfo, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    if current:
                        entries.append(current)
                        current = {}
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                current[key.strip().lower()] = value.strip()
        if current:
            entries.append(current)
    except Exception:
        return None
    cores: set[tuple[str, str]] = set()
    for item in entries:
        physical = item.get("physical id")
        core = item.get("core id")
        if physical is None or core is None:
            continue
        cores.add((physical, core))
    if cores:
        return len(cores)
    cpu_cores = entries[0].get("cpu cores") if entries else None
    if cpu_cores:
        try:
            n = int(cpu_cores)
            if n > 0:
                return n
        except ValueError:
            return None
    return None


def _linux_mem_total_bytes() -> int | None:
    meminfo = "/proc/meminfo"
    if not os.path.isfile(meminfo):
        return None
    try:
        with open(meminfo, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        if kb > 0:
                            return kb * 1024
    except Exception:
        return None
    return None


def _resolve_mount_point(path: str) -> str:
    cur = os.path.abspath(path)
    while not os.path.ismount(cur):
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return cur


def _linux_fs_type_for_mount(mount_point: str) -> str | None:
    mounts_file = "/proc/mounts"
    if not os.path.isfile(mounts_file):
        return None
    normalized = os.path.abspath(mount_point)
    best_match = ""
    best_fs = None
    try:
        with open(mounts_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_raw = parts[1].replace("\\040", " ")
                fs_type = parts[2]
                if normalized == mount_raw or normalized.startswith(mount_raw + os.sep):
                    if len(mount_raw) > len(best_match):
                        best_match = mount_raw
                        best_fs = fs_type
    except Exception:
        return None
    return best_fs


def collect_system_profile(run_dir: str) -> dict[str, Any]:
    warnings: list[str] = []
    cpu_model = _linux_cpu_model_name() or (platform.processor() or None)
    logical_cores = os.cpu_count()
    physical_cores = _linux_physical_core_count()
    if cpu_model is None:
        warnings.append("cpu_model_unavailable")
    if physical_cores is None:
        warnings.append("cpu_physical_cores_unavailable")

    ram_total = _linux_mem_total_bytes()
    if ram_total is None:
        warnings.append("ram_total_unavailable")

    gpu_devices: list[dict[str, Any]] = []
    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gpu_devices.append(
                    {
                        "index": idx,
                        "name": torch.cuda.get_device_name(idx),
                        "total_vram_bytes": int(getattr(props, "total_memory", 0) or 0),
                        "total_vram_gb": _bytes_to_gb(getattr(props, "total_memory", 0) or 0),
                    }
                )
    except Exception:
        warnings.append("gpu_probe_failed")

    mount_point = _resolve_mount_point(run_dir)
    disk_usage = shutil.disk_usage(run_dir)
    fs_type = _linux_fs_type_for_mount(mount_point)
    if fs_type is None:
        warnings.append("filesystem_type_unavailable")

    gpu_total_bytes = sum(int(x.get("total_vram_bytes", 0) or 0) for x in gpu_devices) if gpu_devices else 0

    return {
        "cpu": {
            "model": cpu_model,
            "architecture": platform.machine() or None,
            "logical_cores": int(logical_cores) if logical_cores else None,
            "physical_cores": int(physical_cores) if physical_cores else None,
        },
        "ram": {
            "total_bytes": int(ram_total) if ram_total else None,
            "total_gb": _bytes_to_gb(ram_total),
        },
        "gpu": {
            "cuda_available": cuda_available,
            "devices": gpu_devices,
            "total_vram_bytes": int(gpu_total_bytes) if gpu_total_bytes else 0,
            "total_vram_gb": _bytes_to_gb(gpu_total_bytes),
        },
        "disk": {
            "mount_point": mount_point,
            "filesystem": fs_type,
            "total_bytes": int(disk_usage.total),
            "used_bytes": int(disk_usage.used),
            "free_bytes": int(disk_usage.free),
            "total_gb": _bytes_to_gb(disk_usage.total),
            "used_gb": _bytes_to_gb(disk_usage.used),
            "free_gb": _bytes_to_gb(disk_usage.free),
        },
        "platform": {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
        },
        "capture_warnings": warnings,
    }


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
        "--weighted-sampling",
        action="store_true",
        help="Weighted image sampling (classes with fewer objects more often); ultralytics patch",
    )

    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="After successful training, export best.pt to ONNX",
    )
    parser.add_argument(
        "--export-onnx-fp32",
        action="store_true",
        help="With --export-onnx, do not use half=True",
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

    candidates = list_incomplete_runs(workspace_root)
    chosen: RunDiagnosis | None = None
    if args.run_dir:
        run_dir = args.run_dir
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(WorkspaceLayout(workspace_root).runs, run_dir)
        chosen = next((d for d in candidates if os.path.abspath(run_dir) == d.run_dir), None)
        if chosen is None:
            from smartrain.train_resume import diagnose_run

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
            from smartrain.train_resume import diagnose_run

            dataset_path = resolve_dataset_path_for_resume(chosen.run_dir, workspace_root)
            if not dataset_path:
                raise RuntimeError(
                    "Cannot resolve dataset path for test stage. "
                    "Expected valid dataset in runtime yaml/metadata/workspace datasets catalog."
                )
            _maybe_free_cuda_memory()
            test_yolo(chosen.run_dir, dataset_path)
            update_resume_test_metadata(
                chosen.run_dir,
                success=True,
                error=None,
                diagnosis=diagnose_run(chosen.run_dir),
            )
            print(f"[OK] Missing test stage completed: {chosen.run_dir}")
            return 0
        except Exception as e:
            from smartrain.train_resume import diagnose_run

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
        from smartrain.train_resume import diagnose_run

        resume_training_in_run(chosen.run_dir)
        update_resume_metadata(
            chosen.run_dir,
            success=True,
            error=None,
            diagnosis=diagnose_run(chosen.run_dir),
        )
        print(f"[OK] Resume completed: {chosen.run_dir}")
        return 0
    except Exception as e:
        from smartrain.train_resume import diagnose_run

        update_resume_metadata(
            chosen.run_dir,
            success=False,
            error=str(e),
            diagnosis=diagnose_run(chosen.run_dir),
        )
        print(f"[ERROR] Failed to resume run: {chosen.run_dir}")
        print(f"[ERROR] {e}")
        return 1


def _prompt_input(label: str, default: str = "", completer=None, show_default_hint: bool = True) -> str:
    from prompt_toolkit import prompt

    prompt_label = f"{label} [default: {default}]: " if (default != "" and show_default_hint) else label
    value = str(prompt(prompt_label, default="", completer=completer, complete_while_typing=True)).strip()
    if value:
        return value
    if default != "":
        if sys.stdin.isatty():
            try:
                sys.stdout.write("\x1b[1A\r")
                sys.stdout.write(f"{prompt_label}{default}\n")
                sys.stdout.flush()
            except Exception:
                print(default)
        else:
            print(default)
    return str(default)


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
    options = discover_device_options()
    labels = [o.label for o in options]
    by_label = {o.label: o.value for o in options}
    effective_default = str(default).strip() if default is not None else default_device_value()
    default_label = next((o.label for o in options if o.value == effective_default), labels[0])
    print_numbered_options("Train devices", labels)
    picked = _prompt_input("Train device (--device, number/value): ", default=default_label).strip()
    if not picked:
        return by_label[default_label]
    if picked in by_label:
        return by_label[picked]
    if picked.isdigit():
        idx = int(picked)
        if 1 <= idx <= len(options):
            return options[idx - 1].value
    for option in options:
        if picked == option.value:
            return option.value
    print(f"[WARNING] Unknown device {picked!r}; fallback to default {by_label[default_label]!r}")
    return by_label[default_label]


def _load_available_datasets(layout: WorkspaceLayout) -> list[str]:
    info_path = layout.work_datasets_info_path()
    if not os.path.isfile(info_path):
        return []
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    return sorted(str(k) for k in data.keys())


def _prompt_dataset_name(available: list[str]) -> str:
    from smartrain.cli_prompts import prompt_choice

    return prompt_choice("Dataset", available, default=available[0])


def _train_model_picker_options(default_model: str) -> list[str]:
    catalog = TrainModelCatalog()
    options = list(catalog.supported_aliases())
    external_records = _installed_external_provider_records()
    if external_records:
        # Copy-paste friendly aliases for external providers.
        for rec in external_records:
            pid = str(rec.get("provider_id", "")).strip().lower()
            if not pid:
                continue
            repo_path = str(rec.get("repo_path", "")).strip() or None
            ext_catalog = TrainModelCatalog(provider=pid, provider_repo_path=repo_path)
            options.extend(f"{pid}:{alias}" for alias in ext_catalog.supported_aliases())
    if default_model and default_model not in options:
        options.append(default_model)
    options.append(_MANUAL_MODEL_ENTRY)
    dedup: list[str] = []
    seen: set[str] = set()
    for item in options:
        if item in seen:
            continue
        seen.add(item)
        dedup.append(item)
    return dedup


def _installed_external_provider_ids() -> list[str]:
    return [str(r.get("provider_id", "")).strip().lower() for r in _installed_external_provider_records()]


def _installed_external_provider_records() -> list[dict[str, Any]]:
    reconcile_stale_provider_paths()
    recs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in list_provider_records():
        if str(rec.get("install_state", "")).strip().lower() != "installed":
            continue
        pid = str(rec.get("provider_id", "")).strip().lower()
        if not pid or pid in seen:
            continue
        repo_path = Path(str(rec.get("repo_path", "")).strip()).expanduser()
        venv_path = Path(str(rec.get("venv_path", "")).strip()).expanduser()
        if not repo_path.is_dir() or not venv_path.is_dir():
            continue
        seen.add(pid)
        recs.append(rec)
    recs.sort(key=lambda x: str(x.get("provider_id", "")).strip().lower())
    return recs


def _get_installed_external_provider_record(provider_id: str) -> dict[str, Any] | None:
    key = str(provider_id or "").strip().lower()
    if not key:
        return None
    for rec in _installed_external_provider_records():
        pid = str(rec.get("provider_id", "")).strip().lower()
        if pid == key:
            return rec
    return None


def _apply_external_provider_defaults(args) -> None:
    provider = str(getattr(args, "external_provider", "") or "").strip().lower()
    if not provider:
        return
    if not hasattr(args, "model"):
        rec = _get_installed_external_provider_record(provider)
        repo_path = str(rec.get("repo_path", "")).strip() if isinstance(rec, dict) else None
        aliases = TrainModelCatalog(provider=provider, provider_repo_path=repo_path or None).supported_aliases()
        if aliases:
            args.model = aliases[0]
    if not hasattr(args, "epochs"):
        args.epochs = 70
    if not hasattr(args, "batch"):
        args.batch = 8
    if not hasattr(args, "img_size"):
        args.img_size = 640


def _model_matches_task(alias: str, task: str) -> bool:
    low = alias.lower()
    if ":" in low:
        _, low = low.split(":", 1)
    task_low = (task or "").strip().lower()
    if task_low == "segment":
        return "-seg" in low
    if task_low == "classify":
        return "-cls" in low
    if task_low == "pose":
        return "-pose" in low
    if task_low == "obb":
        return "-obb" in low
    # detect by default: hide non-detection heads
    return all(marker not in low for marker in ("-seg", "-cls", "-pose", "-obb"))


def _format_numbered_columns(items: list[str], *, columns: int = 4) -> list[str]:
    if not items:
        return []
    indexed = [f"{idx + 1}) {name}" for idx, name in enumerate(items)]
    term_w = shutil.get_terminal_size(fallback=(120, 20)).columns
    col_width = max(len(x) for x in indexed) + 2
    cols = max(1, min(columns, max(1, term_w // col_width)))
    rows = (len(indexed) + cols - 1) // cols
    lines: list[str] = []
    for row in range(rows):
        parts: list[str] = []
        for col in range(cols):
            pos = col * rows + row
            if pos >= len(indexed):
                continue
            cell = indexed[pos]
            if col < cols - 1:
                parts.append(cell.ljust(col_width))
            else:
                parts.append(cell)
        lines.append("".join(parts).rstrip())
    return lines


def _pick_model_interactive(options: list[str], default_alias: str) -> str:
    print("[INFO] Model options:")
    for line in _format_numbered_columns(options, columns=4):
        print(f"  {line}")
    while True:
        raw = _prompt_input(
            "Model (--model, number/value): ",
            default=default_alias,
        ).strip()
        if not raw:
            return default_alias
        if raw in options:
            return raw
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print(f"[ERROR] Incorrect selection: {raw!r}")


def _extract_run_timestamp(run_name: str, run_dir: Path) -> datetime:
    m = re.match(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2})", run_name)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y-%m-%d_%H-%M")
        except ValueError:
            pass
    return datetime.fromtimestamp(run_dir.stat().st_mtime)


def _base_run_summary(args_yaml: Path) -> dict[str, str]:
    summary = {
        "provider": "unknown",
        "model": "unknown",
        "task": "unknown",
        "batch": "?",
        "epochs": "?",
    }
    try:
        payload = _load_ultralytics_yaml(str(args_yaml))
    except Exception:
        return summary
    provider = (
        str(payload.get("external_provider") or "").strip().lower()
        if isinstance(payload, dict)
        else ""
    )
    summary["provider"] = provider or "ultralytics"
    if isinstance(payload, dict):
        model = str(payload.get("model") or "").strip()
        task = str(payload.get("task") or "").strip().lower()
        batch = payload.get("batch")
        epochs = payload.get("epochs")
        if model:
            summary["model"] = Path(model).name
        if task:
            summary["task"] = task
        if batch is not None and str(batch).strip():
            summary["batch"] = str(batch)
        if epochs is not None and str(epochs).strip():
            summary["epochs"] = str(epochs)
    return summary


def _collect_available_base_runs(layout: WorkspaceLayout, selected_dataset: str) -> list[dict[str, str]]:
    out: list[dict[str, Any]] = []
    runs_root = Path(layout.runs)
    if not runs_root.is_dir():
        return []
    for ds_dir in sorted(runs_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds_name = ds_dir.name
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            args_train = run_dir / "train" / "args.yaml"
            args_root = run_dir / "args.yaml"
            args_path: Path | None = None
            if args_train.is_file():
                args_path = args_train
            elif args_root.is_file():
                args_path = args_root
            if args_path is None:
                continue
            if not run_dir.exists() or not args_path.exists():
                continue
            run_ts = _extract_run_timestamp(run_dir.name, run_dir)
            run_rel = str(run_dir.relative_to(runs_root))
            info = _base_run_summary(args_path)
            out.append(
                {
                    "dataset": ds_name,
                    "run_dir": str(run_dir),
                    "args_yaml": str(args_path),
                    "run_rel": run_rel,
                    "provider": info["provider"],
                    "model": info["model"],
                    "task": info["task"],
                    "batch": info["batch"],
                    "epochs": info["epochs"],
                    "_sort_ts": run_ts,
                }
            )
    out.sort(
        key=lambda x: (
            x["dataset"] != selected_dataset,
            x["_sort_ts"],
            x["run_rel"],
        )
    )
    for row in out:
        row.pop("_sort_ts", None)
    return out


def _print_available_base_runs(selected_dataset: str, runs: list[dict[str, str]]) -> None:
    if not runs:
        print("[INFO] No base runs found in runs/.")
        return
    print("[INFO] Available base runs (selected dataset first, oldest first):")
    switched_to_other = False
    for i, r in enumerate(runs, start=1):
        if not switched_to_other and r["dataset"] != selected_dataset:
            print("      ---- other datasets ----")
            switched_to_other = True
        mark = " [selected-dataset]" if r["dataset"] == selected_dataset else ""
        task_part = f" task:{r['task']}" if r.get("task", "unknown") not in ("", "detect", "unknown") else ""
        print(
            f"  {i:>3}. {r.get('run_rel', r['run_dir'])}{mark}"
            f" | provider:{r.get('provider', 'unknown')}"
            f" | model:{r.get('model', 'unknown')}"
            f" | b={r.get('batch', '?')} e={r.get('epochs', '?')}{task_part}"
        )


def _prompt_base_run_args_yaml(runs: list[dict[str, str]], default_path: str | None = None) -> str | None:
    if not runs:
        return default_path
    while True:
        raw = _prompt_input(
            "Base run (number or path to args.yaml, empty=no base): ",
            default=str(default_path or ""),
        ).strip()
        if not raw:
            return default_path
        if os.path.isfile(raw):
            return raw
        try:
            idx = int(raw)
        except ValueError:
            print(f"[ERROR] Expected run number or path to args.yaml, received: {raw!r}")
            continue
        if 1 <= idx <= len(runs):
            return runs[idx - 1]["args_yaml"]
        print(f"[ERROR] Number out of range 1..{len(runs)}")


def _get_interactive_default(args, attr: str, fallback, baseline_cfg: dict[str, Any], baseline_key: str):
    if hasattr(args, attr):
        val = getattr(args, attr)
        if val is not None and (fallback is None or val != fallback):
            return val
    if baseline_key in baseline_cfg:
        return baseline_cfg[baseline_key]
    return fallback


def _run_interactive_train_setup(args) -> bool:
    from prompt_toolkit.completion import WordCompleter

    print("[INFO] Interactive train mode (Enter = default).")
    installed_external = _installed_external_provider_records()
    if installed_external:
        print("[INFO] Installed external providers:")
        for rec in installed_external:
            print(f"  - {rec.get('provider_id')}: {rec.get('repo_path')}")

    try:
        ws = resolve_workspace_root(getattr(args, "workspace", None))
    except ValueError:
        ws_raw = _prompt_input("Workspace path: ", default=os.getcwd()).strip()
        if not ws_raw:
            print("[ERROR] Workspace not set.")
            return False
        ws = os.path.abspath(os.path.expanduser(ws_raw))
        args.workspace = ws

    layout = WorkspaceLayout(ws)
    dataset_names = _load_available_datasets(layout)
    if not dataset_names:
        print(
            "[ERROR] There are no available datasets in datasets/datasets_info.json."
            "Please scan first."
        )
        return False
    args.data = _prompt_dataset_name(dataset_names)
    baseline_u_cfg: dict[str, Any] = {}
    baseline_sm_opts: dict[str, Any] = {}
    available_runs = _collect_available_base_runs(layout, args.data)
    _print_available_base_runs(args.data, available_runs)
    baseline_args_yaml = _prompt_base_run_args_yaml(
        available_runs,
        default_path=str(getattr(args, "base_run_args_yaml", "") or "") or None,
    )
    args.base_run_args_yaml = baseline_args_yaml
    if baseline_args_yaml:
        try:
            baseline_profile = _load_ultralytics_yaml(baseline_args_yaml)
            baseline_filtered = {
                k: v for k, v in baseline_profile.items() if k not in _ULTRALYTICS_YAML_IGNORED_KEYS
            }
            baseline_u_cfg, baseline_sm_opts = extract_smartrain_options(baseline_filtered)
            print(f"[INFO] Baseline run used: {baseline_args_yaml}")
        except Exception as e:
            print(f"[WARNING] Failed to read args.yaml of base run: {e}")
            baseline_u_cfg, baseline_sm_opts = {}, {}

    args.ultralytics_yaml = (
        _prompt_input(
            "Path to external Ultralytics args.yaml (--ultralytics_yaml, empty=do not use): ",
            default=str(getattr(args, "ultralytics_yaml", "") or ""),
        ).strip()
        or None
    )
    if args.ultralytics_yaml:
        print(
            "[INFO] For --ultralytics_yaml: data/project/name/exist_ok and service path keys "
            "ignored; data is always taken from the selected dataset."
        )
    ultra_u_cfg: dict[str, Any] = {}
    ultra_sm_opts: dict[str, Any] = {}
    if args.ultralytics_yaml:
        try:
            ultra_profile = _load_ultralytics_yaml(args.ultralytics_yaml)
        except Exception as e:
            print(f"[ERROR] Failed to read --ultralytics_yaml: {e}")
            return False
        filtered = {
            k: v for k, v in ultra_profile.items() if k not in _ULTRALYTICS_YAML_IGNORED_KEYS
        }
        ultra_u_cfg, ultra_sm_opts = extract_smartrain_options(filtered)

    task_choices = ["detect", "segment", "classify", "pose", "obb"]
    if "task" in ultra_u_cfg:
        args.task = str(ultra_u_cfg["task"])
        print(f"[INFO] Task taken from --ultralytics_yaml: {args.task}")
    else:
        task_default = str(
            _get_interactive_default(args, "task", "detect", baseline_u_cfg, "task")
        )
        task_completer = WordCompleter(task_choices, ignore_case=True)
        args.task = (
            _prompt_input(
                "Task (detect/segment/classify/pose/obb): ",
                default=task_default,
                completer=task_completer,
            ).strip()
            or task_default
        )

    if "model" in ultra_u_cfg:
        args.model = _normalize_model_spec(str(ultra_u_cfg["model"]), add_pt_when_missing=True)
        print(f"[INFO] Model taken from --ultralytics_yaml: {args.model}")
    else:
        model_default = _normalize_model_spec(
            str(_get_interactive_default(args, "model", MODEL_VERSION, baseline_u_cfg, "model")),
            add_pt_when_missing=True,
        )
        all_options = _train_model_picker_options(model_default.replace(".pt", ""))
        task_options = [opt for opt in all_options if (opt == _MANUAL_MODEL_ENTRY or _model_matches_task(opt, args.task))]
        options = task_options or all_options
        default_alias = model_default.replace(".pt", "")
        if default_alias not in options:
            default_alias = options[0]
        model_choice = _pick_model_interactive(options, default_alias)
        if model_choice == _MANUAL_MODEL_ENTRY:
            model_choice = (
                _prompt_input(
                    "Manual model alias/path (--model): ",
                    default=model_default,
                ).strip()
                or model_default
            )
        selected_external_provider = None
        if ":" in model_choice:
            provider_part, model_part = model_choice.split(":", 1)
            known_external = set(_installed_external_provider_ids())
            if provider_part in known_external and model_part:
                selected_external_provider = provider_part
                model_choice = model_part
        if selected_external_provider:
            args.external_provider = selected_external_provider
            print(f"[INFO] External provider selected from model alias: {selected_external_provider}")
        else:
            args.external_provider = None
        args.model = _normalize_model_spec(model_choice, add_pt_when_missing=True)
    print(f"[INFO] Final model for launch: {args.model}")
    if "epochs" in ultra_u_cfg:
        args.epochs = int(ultra_u_cfg["epochs"])
        print(f"[INFO] Epochs taken from --ultralytics_yaml: {args.epochs}")
    else:
        args.epochs = _prompt_int(
            "Epoches (--epochs)",
            int(_get_interactive_default(args, "epochs", EPOCHS, baseline_u_cfg, "epochs")),
        )
    if "batch" in ultra_u_cfg:
        args.batch = int(ultra_u_cfg["batch"])
        print(f"[INFO] Batch taken from --ultralytics_yaml: {args.batch}")
    else:
        args.batch = _prompt_int(
            "Batch (--batch)",
            int(_get_interactive_default(args, "batch", BATCH, baseline_u_cfg, "batch")),
        )
    if "imgsz" in ultra_u_cfg:
        args.img_size = int(ultra_u_cfg["imgsz"])
        print(f"[INFO] Image size taken from --ultralytics_yaml: {args.img_size}")
    else:
        args.img_size = _prompt_int(
            "Images Size (--img-size)",
            int(_get_interactive_default(args, "img_size", IMG_SIZE, baseline_u_cfg, "imgsz")),
        )
    args.device = _prompt_train_device(
        str(_get_interactive_default(args, "device", default_device_value(), baseline_u_cfg, "device"))
    )

    default_target = str(getattr(args, "target_path", None) or layout.runs)
    args.target_path = (_prompt_input("Run directory (--target-path): ", default=default_target).strip()
                        or default_target)

    args.test_only = _prompt_yes_no("Test only without training (--test-only)?", default=bool(getattr(args, "test_only", False)))
    if args.test_only:
        model_dir_default = str(getattr(args, "model_dir", "") or "")
        while True:
            model_dir = _prompt_input("Path to model (--model-dir): ", default=model_dir_default).strip()
            if model_dir:
                args.model_dir = model_dir
                break
            print("[ERROR] --test-only requires --model-dir.")
    else:
        args.model_dir = getattr(args, "model_dir", None)

    args.val_imgsz = _prompt_optional_int(
        "Size val/test (--val-imgsz, empty=how train)",
        _get_interactive_default(args, "val_imgsz", None, baseline_u_cfg, "imgsz"),
    )
    args.val_conf = _prompt_optional_float(
        "conf threshold (--val-conf, empty=default Ultralytics)",
        _get_interactive_default(args, "val_conf", None, baseline_u_cfg, "conf"),
    )
    args.val_iou = _prompt_optional_float(
        "IoU threshold (--val-iou, empty=default Ultralytics)",
        _get_interactive_default(args, "val_iou", None, baseline_u_cfg, "iou"),
    )

    if "weighted_sampling" in ultra_sm_opts:
        args.weighted_sampling = bool(ultra_sm_opts["weighted_sampling"])
    else:
        args.weighted_sampling = _prompt_yes_no(
            "Enable weighted sampling (--weighted-sampling)?",
            default=bool(_get_interactive_default(args, "weighted_sampling", False, baseline_sm_opts, "weighted_sampling")),
        )
    if "export_onnx" in ultra_sm_opts:
        args.export_onnx = bool(ultra_sm_opts["export_onnx"])
    else:
        args.export_onnx = _prompt_yes_no(
            "Export ONNX after training (--export-onnx)?",
            default=bool(_get_interactive_default(args, "export_onnx", False, baseline_sm_opts, "export_onnx")),
        )
    if "export_onnx_half" in ultra_sm_opts:
        args.export_onnx_fp32 = not bool(ultra_sm_opts["export_onnx_half"])
    else:
        default_fp32 = bool(getattr(args, "export_onnx_fp32", False))
        if "export_onnx_half" in baseline_sm_opts:
            default_fp32 = not bool(baseline_sm_opts["export_onnx_half"])
        args.export_onnx_fp32 = _prompt_yes_no(
            "Use FP32 for ONNX (--export-onnx-fp32)?",
            default=default_fp32,
        )
    if "clearml" in ultra_sm_opts:
        args.clearml = bool(ultra_sm_opts["clearml"])
    else:
        args.clearml = _prompt_yes_no(
            "Log to ClearML (--clearml)?",
            default=bool(_get_interactive_default(args, "clearml", False, baseline_sm_opts, "clearml")),
        )
    if args.clearml:
        if "clearml_project" in ultra_sm_opts:
            args.clearml_project = str(ultra_sm_opts["clearml_project"]).strip() or None
        else:
            default_cm_project = str(getattr(args, "clearml_project", "") or "")
            if "clearml_project" in baseline_sm_opts:
                default_cm_project = str(baseline_sm_opts["clearml_project"] or "")
            args.clearml_project = (
                _prompt_input(
                    "ClearML Project (--clearml-project): ",
                    default=default_cm_project,
                ).strip()
                or None
            )
    args.non_interactive = _prompt_yes_no(
        "Do not ask for confirmation if the folder exists (--yes)?",
        default=bool(getattr(args, "non_interactive", False)),
    )
    return True


def resolve_training_data_path(layout: WorkspaceLayout, data_arg: str) -> str:
    expanded = os.path.abspath(os.path.expanduser(data_arg))
    yaml_here = os.path.join(expanded, "data.yaml")
    if os.path.isdir(expanded) and os.path.isfile(yaml_here):
        return expanded
    info_path = layout.work_datasets_info_path()
    if not os.path.isfile(info_path):
        raise FileNotFoundError(
            f"The directory with data.yaml for {data_arg!r} was not found and {info_path} is missing."
        )
    with open(info_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    if not isinstance(catalog, dict):
        raise ValueError(f"{info_path}: JSON object expected.")
    if data_arg not in catalog:
        names = ", ".join(sorted(catalog.keys()))
        hint = f" Known names: {names}." if names else ""
        raise ValueError(
            f"Dataset name {data_arg!r} is missing from datasets/{DATASETS_INFO_FILE}.{hint}"
        )
    entry = catalog[data_arg]
    if not isinstance(entry, dict):
        raise ValueError(f"The {data_arg!r} entry must be a JSON object.")
    return resolve_dataset_root(layout.root, data_arg, entry, layout.work_datasets)


def _validate_dataset_dir(dataset_path: str) -> None:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Yaml file not found: {data_yaml}")


def _split_dir_from_dataset_yaml(dataset_path: str, raw: dict, split_key: str) -> str | None:
    """
    Uses train/val/test from data.yaml when they point at an existing directory under dataset_path.
    Needed for CVAT-style layouts (single shared images/ bucket) where split subfolders are absent.
    """
    v = raw.get(split_key)
    if not isinstance(v, str) or not v.strip():
        return None
    rel = v.strip().replace("\\", "/").lstrip("./")
    if not rel:
        return None
    abs_p = os.path.normpath(os.path.join(dataset_path, rel))
    if os.path.isdir(abs_p):
        return rel
    return None


def _pick_split_relative_dir(dataset_path: str, split_aliases: tuple[str, ...]) -> str | None:
    """
    Searches for the split directory within the selected dataset_path.
    Returns a relative path (preferably with images/) or None.
    """
    candidates: list[str] = []
    for split in split_aliases:
        candidates.extend([f"{split}/images", f"images/{split}", split])
    for rel in candidates:
        abs_p = os.path.join(dataset_path, rel)
        if os.path.isdir(abs_p):
            return rel
    return None


def _build_runtime_data_yaml(dataset_path: str, run_dir: str, *, stage: str) -> str:
    """
    Creates a service data.yaml for Ultralytics with a link to the current dataset_path.
    This protects against old absolute paths in the original data.yaml (different machine).
    """
    src_yaml = os.path.join(dataset_path, "data.yaml")
    with open(src_yaml, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Incorrect YAML format data.yaml: {src_yaml}")

    train_rel = _pick_split_relative_dir(dataset_path, ("train",)) or _split_dir_from_dataset_yaml(
        dataset_path, raw, "train"
    )
    val_rel = _pick_split_relative_dir(dataset_path, ("val", "valid")) or _split_dir_from_dataset_yaml(
        dataset_path, raw, "val"
    )
    test_rel = _pick_split_relative_dir(dataset_path, ("test",)) or _split_dir_from_dataset_yaml(
        dataset_path, raw, "test"
    )
    if train_rel is None or val_rel is None:
        raise FileNotFoundError(
            f"Required train/val split folders not found inside {dataset_path}."
        )

    runtime_cfg = dict(raw)
    runtime_cfg["path"] = dataset_path
    runtime_cfg["train"] = train_rel
    runtime_cfg["val"] = val_rel
    if test_rel is not None:
        runtime_cfg["test"] = test_rel

    out_yaml = os.path.join(run_dir, f"_runtime_data_{stage}.yaml")
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_cfg, f, allow_unicode=True, sort_keys=False)
    print(
        f"[INFO] Runtime data.yaml ({stage}) generated for the selected dataset: {out_yaml}"
    )
    return out_yaml


def _resolve_cli_paths_with_profile(args, u_cfg: dict) -> tuple[str | None, str, str]:
    """
    workspace, dataset_root (directory with data.yaml), target_base.
    """
    try:
        ws = resolve_workspace_root(args.workspace)
    except ValueError:
        ws = None

    if ws is not None:
        layout = WorkspaceLayout(ws)
        os.makedirs(layout.runs, exist_ok=True)
        if args.data is not None:
            dataset_path = resolve_training_data_path(layout, args.data)
        elif u_cfg.get("data"):
            yp = resolve_profile_data_path(str(u_cfg["data"]))
            dataset_path = dataset_root_from_data_yaml(yp)
        else:
            raise ValueError(
                "When using workspace, specify --data or the data: field in the --config profile."
            )
        if args.target_path is not None:
            target_base = os.path.abspath(os.path.expanduser(args.target_path))
        else:
            target_base = layout.runs
        return ws, dataset_path, target_base

    if args.data is not None:
        dataset_path = os.path.abspath(os.path.expanduser(args.data))
    elif u_cfg.get("data"):
        yp = resolve_profile_data_path(str(u_cfg["data"]))
        dataset_path = dataset_root_from_data_yaml(yp)
    else:
        raise ValueError(
            f"Specify --workspace (or {WORKSPACE_ENV_VAR}) and --data (or data in YAML), "
            "or without workspace - --data and --target-path."
        )

    if args.target_path is None:
        raise ValueError(
            f"Without workspace, specify --target-path (base run directory) or specify {WORKSPACE_ENV_VAR}."
        )
    target_base = os.path.abspath(os.path.expanduser(args.target_path))
    return None, dataset_path, target_base


def _finalize_train_kwargs(ultralytics_cfg: dict[str, Any], data_yaml: str, model_dir: str) -> dict[str, Any]:
    k = copy.deepcopy(ultralytics_cfg)
    overwritten: list[str] = []
    if "data" in k:
        overwritten.append("data")
    if "project" in k:
        overwritten.append("project")
    if "name" in k:
        overwritten.append("name")
    if "exist_ok" in k:
        overwritten.append("exist_ok")
    k.pop("data", None)
    k["data"] = data_yaml
    k["project"] = model_dir
    k["name"] = "train"
    k["exist_ok"] = False
    k.setdefault("mode", "train")
    if overwritten:
        print(
            "[WARNING] Train service keys have been forced to be overridden: "
            + ", ".join(sorted(set(overwritten)))
        )
    return k


def _load_ultralytics_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = load_train_profile(path)
    if not isinstance(raw, dict):
        return {}
    return raw


def _normalize_model_spec(spec: Any, *, add_pt_when_missing: bool = False) -> str:
    resolver = TrainModelResolver()
    return resolver.resolve(
        None if spec is None else str(spec),
        default_model=MODEL_VERSION,
        add_pt_when_missing=add_pt_when_missing,
    ).normalized


def _extract_effective_loaded_model(model: Any, fallback: str) -> str:
    for candidate in (
        getattr(model, "ckpt_path", None),
        getattr(model, "model_name", None),
        (getattr(model, "overrides", {}) or {}).get("model"),
    ):
        if candidate:
            return str(candidate)
    return str(fallback)


def _extract_model_family_scale(spec: str) -> tuple[str, str] | None:
    token = Path(str(spec)).name.lower()
    if token.endswith(".pt"):
        token = token[:-3]
    for suffix in ("-seg", "-cls", "-pose", "-obb"):
        if token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    m = re.match(r"^(yolo(?:v)?\d+)([nslmx])$", token)
    if not m:
        return None
    return m.group(1), m.group(2)


def _build_run_name(
    provider_id: str,
    model_version: str,
    epochs: int,
    batch: int,
    dataset_hash: str | None,
    *,
    timestamp: datetime | None = None,
) -> str:
    ts = timestamp or datetime.now()
    timestamp_str = ts.strftime("%Y-%m-%d_%H-%M")
    provider = str(provider_id or "ultralytics").strip().lower().replace(" ", "-")
    model_token = Path(str(model_version)).name
    if model_token.endswith(".pt"):
        model_token = model_token[:-3]
    if model_token.endswith(".yaml"):
        model_token = model_token[:-5]
    model_token = re.sub(r"[^a-zA-Z0-9._+-]+", "-", model_token).strip("-") or "model"
    folder_name = f"{timestamp_str}_{provider}_{model_token}_{epochs}epochs_b{batch}"
    if dataset_hash:
        folder_name = f"{folder_name}-{dataset_hash}"
    return folder_name


def _normalize_external_run_layout(run_dir: str) -> None:
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        return
    train_dir = root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    for entry in list(root.iterdir()):
        if entry.name in {"train", "training_metadata.json"}:
            continue
        target = train_dir / entry.name
        if target.exists():
            continue
        entry.rename(target)


def _find_external_best_checkpoint(run_dir: str) -> str | None:
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        return None
    preferred = root / "train" / "weights" / "best.pt"
    if preferred.is_file():
        return str(preferred)
    candidates = [
        root / "weights" / "best.pt",
        root / "train" / "best.pt",
        root / "best.pt",
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    for cand in root.rglob("best.pt"):
        if cand.is_file():
            return str(cand)
    return None


def _ensure_external_best_checkpoint_layout(run_dir: str) -> str | None:
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        return None
    target = root / "train" / "weights" / "best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return str(target)
    src = _find_external_best_checkpoint(run_dir)
    if not src:
        return None
    src_path = Path(src).expanduser().resolve()
    if src_path == target:
        return str(target)
    if not target.exists():
        src_path.rename(target)
    return str(target)


def _resolve_external_eval_source(dataset_path: str) -> str:
    root = Path(dataset_path).expanduser().resolve()
    candidates = [
        root / "test" / "images",
        root / "val" / "images",
        root / "test",
        root / "val",
    ]
    for cand in candidates:
        if cand.is_dir():
            return str(cand)
    return str(root)


def _write_external_fallback_metrics(model_dir: str, *, provider_id: str, rc: int) -> str:
    test_dir = os.path.join(model_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    marker = os.path.join(test_dir, "fallback_infer.txt")
    with open(marker, "w", encoding="utf-8") as f:
        f.write("external infer fallback was used for test stage\n")
    csv_path = os.path.join(model_dir, "test_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("provider,test_mode,return_code\n")
        f.write(f"{provider_id},external_infer_fallback,{int(rc)}\n")
    return csv_path


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
    python_bin = os.path.join(venv_path, "Scripts" if os.name == "nt" else "bin", "python")
    launcher = (
        Path(__file__).resolve().parent
        / "external_providers"
        / "launchers"
        / "mfel_val_launcher.py"
    )
    cmd = [
        python_bin,
        str(launcher),
        "--repo",
        repo_path,
        "--model",
        model_path,
        "--data",
        data_yaml,
        "--imgsz",
        str(int(imgsz)),
        "--project",
        model_dir,
        "--name",
        "test",
    ]
    if conf is not None:
        cmd.extend(["--conf", str(float(conf))])
    if iou is not None:
        cmd.extend(["--iou", str(float(iou))])
    if batch is not None:
        cmd.extend(["--batch", str(int(batch))])
    if device:
        cmd.extend(["--device", str(device)])
    proc = subprocess.run(cmd, cwd=repo_path)
    return int(proc.returncode)


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
            if yaml_key in u_from_ultra and hasattr(args, cli_attr):
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
    ultralytics_cfg = ultralytics_cfg or {}
    smartrain_opts = smartrain_opts or {}

    training_start_time = datetime.now()
    _validate_dataset_dir(dataset_path)

    data_yaml = _build_runtime_data_yaml(dataset_path, target_dir, stage="train")
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

    train_kw = _finalize_train_kwargs(ultralytics_cfg, data_yaml, model_dir)
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
        from smartrain.weighted_yolo_dataset import setup_weighted_sampling_env

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
        from smartrain.weighted_yolo_dataset import register_weighted_sampling_callback

        register_weighted_sampling_callback(model)

    training_end_time = None
    onnx_rel = None
    best_path = os.path.join(model_dir, "train", "weights", "best.pt")
    try:
        model.train(**train_kw)
        training_end_time = datetime.now()
        model_path = best_path
        print("\n" + "-" * 60)
        if os.path.exists(model_path):
            print("[OK] Training complete.")
            print(f"[INFO] Model saved at path:\n{model_path}")
        if smartrain_opts.get("export_onnx") and os.path.exists(model_path):
            half = bool(smartrain_opts.get("export_onnx_half", True))
            simplify = bool(smartrain_opts.get("export_onnx_simplify", True))
            opset = smartrain_opts.get("export_onnx_opset", 17)
            dynamic = bool(smartrain_opts.get("export_onnx_dynamic", False))
            try:
                ex = model.export(
                    format="onnx",
                    dynamic=dynamic,
                    simplify=simplify,
                    opset=int(opset),
                    half=half,
                )
                if ex:
                    onnx_abs = str(ex) if isinstance(ex, (str, Path)) else str(getattr(ex, "path", ex))
                    if os.path.isfile(onnx_abs):
                        onnx_rel = os.path.relpath(onnx_abs, model_dir)
                    else:
                        cand = os.path.join(model_dir, "train", "weights", "best.onnx")
                        if os.path.isfile(cand):
                            onnx_rel = os.path.relpath(cand, model_dir)
                print(f"[INFO] ONNX export completed: {onnx_rel or '(see weights directory)'}")
            except Exception as ex_err:
                print(f"[WARNING] ONNX export failed: {ex_err}")
    except Exception as e:
        training_end_time = datetime.now()
        print(
            f"[ERROR] Failed to start training {model_version} on dataset {dataset_name}"
            f"on {epochs} eras: {e}"
        )
    finally:
        if clearml_task is not None:
            try:
                clearml_task.close()
            except Exception:
                pass

    meta_extras = {
        "train_kw": {k: v for k, v in train_kw.items() if k != "data"},
        "task_type": task_to_metadata_task_type(train_kw.get("task")),
        "onnx_relative": onnx_rel,
        "training_ok": os.path.isfile(best_path),
    }
    return model_dir, training_start_time, training_end_time, dataset_hash, workspace_root, meta_extras


def model_kw_model(train_kw: dict) -> str:
    return str(train_kw.get("model", ""))


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
):
    test_start_time = datetime.now()

    data_yaml = _build_runtime_data_yaml(dataset_path, model_dir, stage="test")
    imgsz = val_imgsz if val_imgsz is not None else train_img_size

    inference_record = {
        "imgsz": imgsz,
        "conf": val_conf,
        "iou": val_iou,
        "batch": val_batch,
    }

    model_path = os.path.join(model_dir, "train", "weights", "best.pt")
    trained_model = YOLO(model_path)

    val_kwargs = {
        "data": data_yaml,
        "split": "test",
        "project": model_dir,
        "name": "test",
        "exist_ok": False,
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

        test_end_time = datetime.now()
        csv_file = save_metrics_csv(result, model_dir)

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

    return test_start_time, test_end_time, inference_record


def save_metrics_csv(test_result, model_dir):
    base_name = "test_metrics"
    ext = ".csv"
    csv_file = os.path.join(model_dir, base_name + ext)

    counter = 1
    while os.path.exists(csv_file):
        csv_file = os.path.join(model_dir, f"{base_name}_{counter}{ext}")
        counter += 1

    csv_data = test_result.to_csv()
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(csv_data)

    return csv_file


def _relative_to_workspace(path: str, workspace_root: str) -> str:
    ap = os.path.abspath(path)
    wr = os.path.abspath(workspace_root)
    try:
        return os.path.relpath(ap, wr)
    except ValueError:
        return ap


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass


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
    onnx_relative=None,
    training_provider: str = "ultralytics",
    external_provider_id: str | None = None,
    system_profile: dict[str, Any] | None = None,
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
            "best_model": "train/weights/best.pt"
            if os.path.exists(os.path.join(model_dir, "train", "weights", "best.pt"))
            else None,
        },
    }

    if ultralytics_train_summary:
        metadata["training_info"]["ultralytics_train"] = ultralytics_train_summary
    if onnx_relative:
        metadata["paths"]["onnx"] = onnx_relative

    if workspace_root is not None:
        metadata["workspace"] = {
            "root": ".",
            "dataset_path_relative": _relative_to_workspace(dataset_path, workspace_root),
            "run_directory_relative": _relative_to_workspace(model_dir, workspace_root),
        }

    if inference:
        metadata["inference"] = {k: v for k, v in inference.items() if v is not None}
    if system_profile:
        metadata["system_profile"] = system_profile

    metadata_file = os.path.join(model_dir, "training_metadata.json")

    try:
        _write_json_atomic(metadata_file, metadata)
        print(f"[INFO] Training metadata saved: {metadata_file}")
    except Exception as e:
        print(f"[WARNING] Failed to save metadata: {e}")


def _get_relative_path(target_path, base_path):
    try:
        target = Path(os.path.abspath(target_path))
        base = Path(os.path.abspath(base_path))

        try:
            relative = os.path.relpath(target, base)
            return relative
        except ValueError:
            return target.as_posix()
    except Exception:
        return os.path.abspath(target_path)


def _json_safe_train_summary(train_kw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not train_kw:
        return None
    out: dict[str, Any] = {}
    for k, v in train_kw.items():
        if k in ("data",):
            continue
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def _load_batch_from_training_metadata(model_dir: str) -> int | None:
    """
    In --test-only mode we want to test with the same batch that was used during training.
    We take it from training_metadata.json if the file exists and the format is expected.
    """
    try:
        meta_path = os.path.join(model_dir, "training_metadata.json")
        if not os.path.isfile(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        bs = (
            meta.get("training_info", {})
            .get("hyperparameters", {})
            .get("batch_size")
        )
        if bs is None:
            return None
        bs_i = int(bs)
        return bs_i if bs_i > 0 else None
    except Exception:
        return None


def _maybe_free_cuda_memory() -> None:
    """
    Mitigating OOM between train and val/test in one process.
    """
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Sometimes it helps to collect the IPC cache, but is not required.
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        # torch may not be available in non-GPU/torch environments; it's not critical.
        pass


def _ensure_device_available_or_raise(device: str | None) -> None:
    if not is_cuda_device(device):
        return
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"CUDA device requested ({device}), but torch is unavailable: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({device}), but torch.cuda.is_available()=False. "
            f"torch={getattr(torch, '__version__', 'unknown')} cuda_runtime={getattr(torch.version, 'cuda', 'unknown')}"
        )


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "resume":
        return _run_resume_command(argv[1:])
    args = parse_args(argv)
    _apply_external_provider_defaults(args)
    known_provider_ids = {spec.id for spec in list_provider_specs()}
    try:
        parsed_ref = validate_external_model_ref(
            parse_external_model_ref(getattr(args, "model", None)),
            known_provider_ids=known_provider_ids,
        )
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 2
    if parsed_ref.is_external and parsed_ref.provider_id and not getattr(args, "external_provider", None):
        args.external_provider = parsed_ref.provider_id
        args.model = parsed_ref.model_ref
        print(f"[INFO] External provider inferred from --model: {parsed_ref.provider_id}")
    parser = build_train_arg_parser()
    interactive_mode = is_interactive_allowed(argv)
    replay_cmd = None
    if interactive_mode:
        if not sys.stdin.isatty():
            print(
                "[ERROR] Interactive train mode requires a terminal (TTY)."
                "Either run in terminal or pass arguments."
            )
            return
        try:
            ok = _run_interactive_train_setup(args)
        except Exception as e:
            print(f"[ERROR] Train interactive mode error: {e}")
            return
        if not ok:
            return
        replay_cmd = build_non_interactive_command("train", parser, args)
        print_replay_command("before launch", replay_cmd)

    profile = load_train_profile(args.config) if args.config else {}
    ultra_profile = _load_ultralytics_yaml(getattr(args, "ultralytics_yaml", None))
    u_cfg, sm_opts = _merge_sources_with_priority(
        config_profile=profile,
        ultralytics_profile=ultra_profile,
        args=args,
    )
    merge_cli_into_ultralytics_cfg(
        u_cfg,
        model=getattr(args, "model", None),
        epochs=getattr(args, "epochs", None),
        batch=getattr(args, "batch", None),
        imgsz=getattr(args, "img_size", None),
        task=getattr(args, "task", None),
        device=getattr(args, "device", None),
        defaults={
            "model": MODEL_VERSION,
            "epochs": EPOCHS,
            "batch": BATCH,
            "imgsz": IMG_SIZE,
            "task": "detect",
            "device": default_device_value(),
        },
    )
    apply_cli_smartrain_overrides(sm_opts, args)

    if getattr(args, "export_onnx_fp32", False):
        sm_opts["export_onnx_half"] = False

    try:
        workspace_root, data, target_dir = _resolve_cli_paths_with_profile(args, u_cfg)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    u_cfg.pop("data", None)

    model_version = _normalize_model_spec(u_cfg.get("model", MODEL_VERSION), add_pt_when_missing=True)
    u_cfg["model"] = model_version
    _ensure_device_available_or_raise(str(u_cfg.get("device")) if u_cfg.get("device") is not None else None)
    epochs = int(u_cfg.get("epochs", EPOCHS))
    batch = int(u_cfg.get("batch", BATCH))
    img_size = u_cfg.get("imgsz", IMG_SIZE)
    try:
        img_size = int(img_size) if img_size is not None else IMG_SIZE
    except (TypeError, ValueError):
        img_size = IMG_SIZE

    external_provider = str(getattr(args, "external_provider", "") or "").strip()
    if external_provider:
        rec = _get_installed_external_provider_record(external_provider)
        repo_for_catalog = str(rec.get("repo_path", "")).strip() if isinstance(rec, dict) else None
        requested_model = str(getattr(args, "model", "") or model_version)
        if not os.path.isfile(requested_model):
            is_supported = is_supported_external_provider_model(
                external_provider,
                requested_model,
                provider_repo_path=repo_for_catalog or None,
            )
            if not is_supported:
                ext_aliases = TrainModelCatalog(
                    provider=external_provider,
                    provider_repo_path=repo_for_catalog or None,
                ).supported_aliases()
                known = ", ".join(ext_aliases) if ext_aliases else "<none>"
                print(
                    f"[ERROR] Model {requested_model!r} is not supported by external provider "
                    f"{external_provider!r}. Supported aliases: {known}"
                )
                return 2
        training_start_time = datetime.now()
        location = get_provider_location(external_provider)
        if location is None and not getattr(args, "external_repo", None):
            print(
                f"[ERROR] External provider {external_provider!r} is not installed. "
                "Use `smartrain providers install` or pass --external-repo."
            )
            return 1
        repo_path = str(getattr(args, "external_repo", "") or "").strip() or (location.repo_path if location else "")
        venv_path = location.venv_path if location else os.path.join(repo_path, "venv")
        if not venv_path:
            print(f"[ERROR] Missing venv for external provider {external_provider!r}. Reinstall provider.")
            return 1
        try:
            dataset_hash = calculate_dataset_hash(data)
        except Exception:
            dataset_hash = None
        run_name = _build_run_name(external_provider, model_version, epochs, batch, dataset_hash)
        print(f"[INFO] External run name: {run_name}")
        rc = run_external_train(
            external_provider,
            repo_path,
            venv_path,
            dataset_path=data,
            model=model_version,
            epochs=epochs,
            batch=batch,
            imgsz=img_size,
            device=str(u_cfg.get("device")) if u_cfg.get("device") is not None else None,
            target_dir=target_dir,
            run_name=run_name,
        )
        training_end_time = datetime.now()
        dataset_name = os.path.basename(os.path.normpath(data))
        external_run_dir = os.path.join(target_dir, dataset_name, run_name)
        os.makedirs(external_run_dir, exist_ok=True)
        _normalize_external_run_layout(external_run_dir)
        _ensure_external_best_checkpoint_layout(external_run_dir)
        test_success = False
        test_error = None
        test_start_time = None
        test_end_time = None
        inference_info = None
        if rc == 0:
            try:
                _maybe_free_cuda_memory()
                val_batch = args.val_batch if args.val_batch is not None else batch
                test_start_time, test_end_time, inference_info = test_yolo(
                    external_run_dir,
                    data,
                    training_start_time=training_start_time,
                    training_end_time=training_end_time,
                    train_img_size=img_size,
                    val_imgsz=args.val_imgsz,
                    val_conf=args.val_conf,
                    val_iou=args.val_iou,
                    val_batch=val_batch,
                )
                test_success = True
            except Exception as e:
                test_error = f"{str(e)}\n{traceback.format_exc()}"
                print(f"[ERROR] Error during external provider testing: {e}")
                best_model = _ensure_external_best_checkpoint_layout(external_run_dir)
                if best_model:
                    fallback_start = datetime.now()
                    fallback_source = _resolve_external_eval_source(data)
                    fallback_conf = float(args.val_conf) if args.val_conf is not None else 0.25
                    fallback_imgsz = int(args.val_imgsz) if args.val_imgsz is not None else int(img_size)
                    if external_provider == "mfel-yolo":
                        fallback_rc = _run_mfel_external_val_fallback(
                            repo_path=repo_path,
                            venv_path=venv_path,
                            model_path=best_model,
                            data_yaml=os.path.join(data, "data.yaml"),
                            model_dir=external_run_dir,
                            imgsz=fallback_imgsz,
                            conf=args.val_conf,
                            iou=args.val_iou,
                            batch=args.val_batch if args.val_batch is not None else batch,
                            device=str(u_cfg.get("device")) if u_cfg.get("device") is not None else None,
                        )
                    else:
                        fallback_rc = run_external_infer(
                            external_provider,
                            repo_path,
                            venv_path,
                            model_path=best_model,
                            source_path=fallback_source,
                            conf=fallback_conf,
                            imgsz=fallback_imgsz,
                            device=str(u_cfg.get("device")) if u_cfg.get("device") is not None else None,
                            target_dir=external_run_dir,
                            run_name="test",
                        )
                    fallback_end = datetime.now()
                    if fallback_rc == 0:
                        if external_provider == "mfel-yolo":
                            # keep test metrics contract in run root
                            test_results_csv = os.path.join(external_run_dir, "test", "results.csv")
                            if os.path.isfile(test_results_csv):
                                shutil.copy2(
                                    test_results_csv, os.path.join(external_run_dir, "test_metrics.csv")
                                )
                            else:
                                _write_external_fallback_metrics(
                                    external_run_dir, provider_id=external_provider, rc=fallback_rc
                                )
                        else:
                            _write_external_fallback_metrics(
                                external_run_dir, provider_id=external_provider, rc=fallback_rc
                            )
                        test_start_time = fallback_start
                        test_end_time = fallback_end
                        inference_info = {
                            "imgsz": fallback_imgsz,
                            "conf": fallback_conf,
                            "mode": "external_infer_fallback",
                        }
                        test_success = True
                        test_error = None
                    else:
                        test_success = False
                        test_error = (
                            f"{test_error}\nExternal infer fallback failed with return code {fallback_rc}"
                        )
                else:
                    test_success = False
        save_training_metadata(
            model_dir=external_run_dir,
            dataset_path=data,
            model_version=model_version.replace(".pt", ""),
            training_start_time=training_start_time,
            training_end_time=training_end_time,
            test_start_time=test_start_time,
            test_end_time=test_end_time,
            epochs=epochs,
            batch=batch,
            img_size=img_size,
            training_success=(rc == 0),
            training_error=None if rc == 0 else f"external provider returned code {rc}",
            test_success=test_success if rc == 0 else False,
            test_error=test_error if rc == 0 else "test skipped because external train failed",
            dataset_hash=dataset_hash,
            inference=inference_info,
            workspace_root=workspace_root,
            task_type=task_to_metadata_task_type(u_cfg.get("task")),
            training_provider=external_provider,
            external_provider_id=external_provider,
            system_profile=collect_system_profile(external_run_dir),
        )
        try:
            marker = {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "provider": {"type": "external", "id": external_provider},
                "model": model_version,
                "dataset_path": data,
                "target_dir": target_dir,
                "run_dir": external_run_dir,
                "repo_path": repo_path,
                "venv_path": venv_path,
                "return_code": int(rc),
            }
            marker_path = os.path.join(target_dir, "_external_train_last.json")
            os.makedirs(target_dir, exist_ok=True)
            with open(marker_path, "w", encoding="utf-8") as f:
                json.dump(marker, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return rc

    training_success = False
    training_error = None
    test_success = True
    test_error = None
    training_start_time = None
    training_end_time = None
    test_start_time = None
    test_end_time = None
    model_dir = None
    inference_info = None
    dataset_hash = None
    meta_extras: dict[str, Any] = {}

    if not args.test_only:
        try:
            (
                model_dir,
                training_start_time,
                training_end_time,
                dataset_hash,
                _,
                meta_extras,
            ) = train_yolo(
                dataset_path=data,
                target_dir=target_dir,
                non_interactive=args.non_interactive,
                workspace_root=workspace_root,
                ultralytics_cfg=u_cfg,
                smartrain_opts=sm_opts,
            )
            training_success = bool(meta_extras.get("training_ok"))
        except Exception as e:
            training_success = False
            training_error = str(e)
            training_end_time = datetime.now()
            print(f"[ERROR] Error during training: {e}")
            training_error = f"{str(e)}\n{traceback.format_exc()}"
            try:
                dataset_hash = calculate_dataset_hash(data)
            except Exception:
                dataset_hash = None
            if not model_dir:
                dataset_name = os.path.basename(os.path.normpath(data))
                folder_name = _build_run_name(
                    "ultralytics",
                    model_version,
                    epochs,
                    batch,
                    dataset_hash,
                    timestamp=training_start_time,
                )
                model_dir = os.path.join(target_dir, dataset_name, folder_name)
                os.makedirs(model_dir, exist_ok=True)
            meta_extras = {
                "task_type": task_to_metadata_task_type(u_cfg.get("task")),
                "train_kw": {k: v for k, v in u_cfg.items() if k != "data"},
                "training_ok": False,
            }

        if training_success and model_dir:
            try:
                _maybe_free_cuda_memory()
                val_batch = args.val_batch if args.val_batch is not None else batch
                test_start_time, test_end_time, inference_info = test_yolo(
                    model_dir,
                    data,
                    training_start_time=training_start_time,
                    training_end_time=training_end_time,
                    train_img_size=img_size,
                    val_imgsz=args.val_imgsz,
                    val_conf=args.val_conf,
                    val_iou=args.val_iou,
                    val_batch=val_batch,
                )
            except Exception as e:
                test_success = False
                test_error = str(e)
                test_end_time = datetime.now()
                print(f"[ERROR] Error during testing: {e}")
                test_error = f"{str(e)}\n{traceback.format_exc()}"

        if model_dir:
            save_training_metadata(
                model_dir=model_dir,
                dataset_path=data,
                model_version=model_version.replace(".pt", ""),
                training_start_time=training_start_time,
                training_end_time=training_end_time,
                test_start_time=test_start_time,
                test_end_time=test_end_time,
                epochs=epochs,
                batch=batch,
                img_size=img_size,
                training_success=training_success,
                training_error=training_error,
                test_success=test_success,
                test_error=test_error,
                dataset_hash=dataset_hash,
                inference=inference_info,
                workspace_root=workspace_root,
                task_type=meta_extras.get("task_type") or task_to_metadata_task_type(u_cfg.get("task")),
                ultralytics_train_summary=_json_safe_train_summary(meta_extras.get("train_kw")),
                onnx_relative=meta_extras.get("onnx_relative"),
                training_provider="ultralytics",
                external_provider_id=None,
                system_profile=collect_system_profile(model_dir),
            )
    else:
        model_dir = args.model_dir
        if model_dir:
            try:
                val_batch = (
                    args.val_batch
                    if args.val_batch is not None
                    else (_load_batch_from_training_metadata(model_dir) or batch)
                )
                test_start_time, test_end_time, inference_info = test_yolo(
                    model_dir,
                    data,
                    train_img_size=img_size,
                    val_imgsz=args.val_imgsz,
                    val_conf=args.val_conf,
                    val_iou=args.val_iou,
                    val_batch=val_batch,
                )
            except Exception as e:
                test_success = False
                test_error = str(e)
                test_end_time = datetime.now()
                print(f"[ERROR] Error during testing: {e}")
                test_error = f"{str(e)}\n{traceback.format_exc()}"

            save_training_metadata(
                model_dir=model_dir,
                dataset_path=data,
                test_start_time=test_start_time,
                test_end_time=test_end_time,
                test_success=test_success,
                test_error=test_error,
                inference=inference_info,
                workspace_root=workspace_root,
                task_type=task_to_metadata_task_type(u_cfg.get("task")),
                training_provider="ultralytics",
                external_provider_id=None,
                system_profile=collect_system_profile(model_dir),
            )
        else:
            print("[ERROR] Model path not specified")
    if replay_cmd:
        print_replay_command("after execution", replay_cmd)


if __name__ == "__main__":
    main()
