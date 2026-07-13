from __future__ import annotations

import argparse
import atexit
import os
import sys

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_replay import print_replay_command  # backward-compatible symbol for tests/mocks
from smartrain.cli_entrypoints.support.cli_prompts import print_numbered_options, prompt_choice, prompt_text, prompt_yes_no
from smartrain.cli_entrypoints.support.cli_contracts import emit_replay, make_command_request
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.run_discovery import find_run_directories
from smartrain.core.runtime.ultralytics_ephemeral import best_effort_prune_workspace_runs_detect
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.core.runtime.device_selector import (
    default_device_value,
    device_display_name,
    prompt_device_selection,
    resolve_device_request,
    validate_device_available,
)
from smartrain.services.inference_arg_parser import build_inference_arg_parser
from smartrain.services.inference_dataset_export import resolve_export_options, validate_export_options
from smartrain.services.inference_service import run_inference_job
from smartrain.services.inference_runtime_helpers import (
    DATA_MODES,
    ON_EMPTY_MODES,
    ROI_POLICIES,
    discover_model_entries,
    infer_img_size_with_source_safe,
    load_catalog,
    resolve_model,
)
from smartrain.workflows.models.model_context import DEFAULT_INFERENCE_IMGSZ, FALLBACK_IMGSZ_SOURCE

# Test / integration imports (canonical model resolution).
_resolve_model = resolve_model

__all__ = [
    "build_inference_arg_parser",
    "main",
    "_resolve_model",
    "print_replay_command",
]


def _interactive_fill(args: argparse.Namespace, layout: WorkspaceLayout) -> bool:
    print("[INFO] Interactive inference mode (Enter = default).")
    model_entries = discover_model_entries(layout)
    model_options = ["models", "runs", "weights"]
    model_source = "models" if model_entries else "weights"
    model_source = prompt_choice("Model source", model_options, default=model_source)
    args.model_name = None
    args.run = None
    args.weights = None
    if model_source == "models":
        if not model_entries:
            print("[ERROR] No model files found in workspace/models.")
            return False
        labels = [x[0] for x in model_entries]
        print_numbered_options("models", labels)
        selected_label = prompt_choice(
            "Select model file from models",
            labels,
            default=labels[0],
            show_options=False,
        )
        selected = next((x for x in model_entries if x[0] == selected_label), None)
        if selected is None:
            print("[ERROR] Internal error: selected model not found.")
            return False
        args.model_name = selected[1]
    elif model_source == "runs":
        runs = find_run_directories(layout.runs)
        if not runs:
            print("[ERROR] No runs found in workspace/runs.")
            return False
        pretty: list[str] = []
        opts: list[str] = []
        for i, rd in enumerate(runs, start=1):
            rel = os.path.relpath(rd, layout.root)
            pretty.append(f"{i}. {rel}")
            opts.append(str(i))
        print("[INFO] Available runs:")
        for row in pretty:
            print(f"  {row}")
        args.run = prompt_choice("Select run index", opts, default=opts[0], show_options=False)
    else:
        args.weights = prompt_text("Weights path", default="models").strip()

    inferred_imgsz = None
    inferred_imgsz_source = FALLBACK_IMGSZ_SOURCE
    try:
        mpath, _mname, _msrc = resolve_model(args, layout)
        inferred_imgsz, inferred_imgsz_source = infer_img_size_with_source_safe(mpath)
    except Exception:
        inferred_imgsz = None
        inferred_imgsz_source = FALLBACK_IMGSZ_SOURCE

    args.data_mode = prompt_choice("Data mode", list(DATA_MODES), default=args.data_mode)
    if args.data_mode == "folder":
        args.source_dir = prompt_text(
            "Source directory or archive",
            default=str(getattr(args, "source", None) or args.source_dir or "datasets"),
        ).strip()
        args.roi_pre_detect = prompt_yes_no("Enable ROI pre-detect", default=bool(args.roi_pre_detect))
        if args.roi_pre_detect:
            args.roi_weights = prompt_text("ROI weights (empty = main model)", default=str(args.roi_weights or "")).strip() or None
            args.roi_conf = float(prompt_text("ROI conf", default=str(args.roi_conf)).strip() or str(args.roi_conf))
            args.roi_policy = prompt_choice("ROI policy", list(ROI_POLICIES), default=args.roi_policy)
            args.roi_pad_px = int(prompt_text("ROI pad px", default=str(args.roi_pad_px)).strip() or str(args.roi_pad_px))
            args.roi_on_empty = prompt_choice("ROI on empty", list(ON_EMPTY_MODES), default=args.roi_on_empty)
            args.roi_class_ids = prompt_text("ROI class ids CSV (empty=all)", default=str(args.roi_class_ids or "")).strip() or None
    else:
        catalog = load_catalog(layout)
        ds_names = sorted(catalog.keys())
        if not ds_names:
            print("[ERROR] datasets_info.json has no datasets.")
            return False
        print_numbered_options("datasets", ds_names)
        args.dataset = prompt_choice("Select dataset", ds_names, default=ds_names[0], show_options=False)
        args.split = prompt_choice("Split", ["train", "val", "test"], default=args.split or "test")
        args.roi_pre_detect = False
        args.source_dir = None

    args.limit = int(prompt_text("Images limit (0=all)", default=str(args.limit)).strip() or str(args.limit))
    if args.img_size is None:
        if inferred_imgsz is not None:
            print(f"[INFO] Resolved input size: {inferred_imgsz} (source: {inferred_imgsz_source})")
        else:
            print(
                f"[WARN] Model input size not found. Using fallback {DEFAULT_INFERENCE_IMGSZ}. "
                "Set --img-size to override."
            )
    img_default = (
        args.img_size
        if args.img_size is not None
        else (inferred_imgsz if inferred_imgsz is not None else DEFAULT_INFERENCE_IMGSZ)
    )
    img_source = inferred_imgsz_source if inferred_imgsz is not None else FALLBACK_IMGSZ_SOURCE
    chosen = prompt_text("Input resolution (--img-size)", default=str(img_default)).strip() or str(img_default)
    args.img_size = int(chosen)
    args.img_size_source = img_source if str(args.img_size) == str(img_default) else "cli"
    args.conf = float(prompt_text("Inference conf", default=str(args.conf)).strip() or str(args.conf))
    args.device = prompt_device_selection(
        title="inference devices",
        default_device=str(args.device or default_device_value()),
    )
    args.half = prompt_yes_no("Use FP16 (--half)", default=bool(args.half))
    args.export_dataset = prompt_yes_no("Export YOLO autolabel dataset", default=bool(getattr(args, "export_dataset", True)))
    if args.export_dataset:
        args.export_label_conf_min = float(
            prompt_text("Export label conf min", default=str(getattr(args, "export_label_conf_min", 0.25))).strip()
            or str(getattr(args, "export_label_conf_min", 0.25))
        )
        args.export_label_conf_max = float(
            prompt_text("Export label conf max", default=str(getattr(args, "export_label_conf_max", 1.0))).strip()
            or str(getattr(args, "export_label_conf_max", 1.0))
        )
    args.export_visualize = prompt_yes_no(
        "Save prediction overlays",
        default=bool(args.export_dataset),
    )
    return True


def _ensure_device_available_or_exit(device: str | None) -> None:
    try:
        validate_device_available(device)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)


def _validate_non_interactive_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.data_mode == "folder":
        source_path = getattr(args, "source", None) or args.source_dir
        if getattr(args, "source", None) and args.source_dir:
            parser.error("Specify only one of --source or --source-dir.")
        if not source_path:
            parser.error("incomplete arguments: --source or --source-dir is required for --data-mode folder.")
        if not args.source_dir:
            args.source_dir = str(source_path)
    if args.data_mode == "dataset-split" and not args.dataset:
        parser.error("incomplete arguments: --dataset is required for --data-mode dataset-split.")
    if args.data_mode == "dataset-split" and not args.split:
        args.split = "test"
    if not args.model_name and not args.run and not args.weights:
        parser.error("incomplete arguments: specify --model-name, --run or --weights.")


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    request = make_command_request("inference", argv, interactive_allowed=is_interactive_allowed(argv))
    parser = build_inference_arg_parser()
    args = parser.parse_args(argv)
    args.device = resolve_device_request(args.device or default_device_value())

    try:
        workspace_root = resolve_workspace_root(args.workspace)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)
    layout = WorkspaceLayout(workspace_root)
    atexit.register(lambda wr=workspace_root: best_effort_prune_workspace_runs_detect(wr))
    os.makedirs(os.path.join(layout.root, "inference"), exist_ok=True)
    interactive_allowed = request.interactive_allowed
    interactive_used = False
    if len(argv) == 0 and interactive_allowed:
        if not sys.stdin.isatty():
            print(
                "[ERROR] Interactive inference mode requires a terminal (TTY). "
                "Run with explicit arguments in non-interactive environments.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if not _interactive_fill(args, layout):
            raise SystemExit(1)
        interactive_used = True
        request.interactive_used = True
        emit_replay(command_name="inference", parser=parser, args=args, stage="before launch")
    else:
        _validate_non_interactive_args(parser, args)
    validate_export_options(resolve_export_options(args), parser=parser)
    _ensure_device_available_or_exit(args.device)
    print(f"[INFO] Inference device: {device_display_name(args.device)}")

    code, exit_via_sysexit = run_inference_job(args, layout)
    if exit_via_sysexit:
        raise SystemExit(code)
    if code != 0:
        raise SystemExit(code)
    if interactive_used:
        emit_replay(command_name="inference", parser=parser, args=args, stage="after execution")


if __name__ == "__main__":
    main()
