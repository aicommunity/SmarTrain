from __future__ import annotations

import argparse
import atexit
import os
import sys

from smartrain.cli_support.cli_argparse import CliArgumentParser
from smartrain.cli_support.cli_replay import print_replay_command  # backward-compatible symbol for tests/mocks
from smartrain.cli_support.cli_prompts import print_numbered_options, prompt_choice, prompt_text, prompt_yes_no
from smartrain.cli_support.cli_contracts import emit_replay, make_command_request
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
from smartrain.services.inference_service import run_inference_job
from smartrain.services.inference_runtime_helpers import (
    DATA_MODES,
    ON_EMPTY_MODES,
    ROI_POLICIES,
    discover_model_entries,
    infer_img_size_from_model_context_safe,
    load_catalog,
    resolve_model,
)

# Test / integration imports (canonical model resolution).
_resolve_model = resolve_model

__all__ = [
    "build_inference_arg_parser",
    "main",
    "_resolve_model",
    "print_replay_command",
]


def build_inference_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Run object detection inference and save JSON report (empty call starts interactive mode)."
    )
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR}).")
    p.add_argument("--model-name", type=str, default=None, help="Promoted model directory name from workspace/models.")
    p.add_argument("--run", type=str, default=None, help="Run path or run index from workspace/runs list.")
    p.add_argument("--weights", type=str, default=None, help="Explicit model weights path (.pt/.onnx/.engine/.trt).")
    p.add_argument("--data-mode", choices=DATA_MODES, default="folder", help="Data source mode.")
    p.add_argument("--source-dir", type=str, default=None, help="Folder with images (recursive).")
    p.add_argument("--dataset", type=str, default=None, help="Dataset key from datasets/datasets_info.json.")
    p.add_argument("--split", choices=("train", "val", "test"), default="test", help="Dataset split for dataset-split mode.")
    p.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all).")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for inference model.")
    p.add_argument("--img-size", type=int, default=None, help="Inference input resolution (imgsz).")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Ultralytics device (cpu, 0, etc). Default: GPU 0 if available, otherwise cpu.",
    )
    p.add_argument("--half", action="store_true", help="Enable FP16 where supported.")
    p.add_argument("--perf-warmup-images", type=int, default=5, help="Warmup images excluded from steady perf statistics.")
    p.add_argument("--roi-pre-detect", action="store_true", help="Pre-detect ROI before inference (folder mode only).")
    p.add_argument("--roi-weights", type=str, default=None, help="ROI detector weights path (.pt/.onnx).")
    p.add_argument("--roi-conf", type=float, default=0.25, help="Confidence threshold for ROI detector.")
    p.add_argument("--roi-policy", choices=ROI_POLICIES, default="largest", help="ROI selection policy.")
    p.add_argument("--roi-pad-px", type=int, default=0, help="Padding in pixels around selected ROI.")
    p.add_argument("--roi-on-empty", choices=ON_EMPTY_MODES, default="full_image", help="Behavior when ROI detector has no detections.")
    p.add_argument("--roi-class-ids", type=str, default=None, help="CSV class ids for ROI detector (empty=all).")
    p.add_argument("--external-provider", type=str, default=None, help="External provider id for inference.")
    p.add_argument("--external-repo", type=str, default=None, help="Override external provider repository path.")
    p.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["detect", "segment", "classify", "detection", "segmentation", "classification"],
        help="Task type hint for task-aware backend routing (default: detection).",
    )
    return p


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
    try:
        mpath, _mname, _msrc = resolve_model(args, layout)
        inferred_imgsz = infer_img_size_from_model_context_safe(mpath)
    except Exception:
        inferred_imgsz = None

    args.data_mode = prompt_choice("Data mode", list(DATA_MODES), default=args.data_mode)
    if args.data_mode == "folder":
        args.source_dir = prompt_text("Source directory", default=args.source_dir or "datasets").strip()
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
        args.split = prompt_choice("Split", ["train", "val", "test"], default=args.split)
        args.roi_pre_detect = False
        args.source_dir = None

    args.limit = int(prompt_text("Images limit (0=all)", default=str(args.limit)).strip() or str(args.limit))
    img_default = inferred_imgsz if inferred_imgsz is not None else (args.img_size if args.img_size is not None else 640)
    args.img_size = int(prompt_text("Input resolution (--img-size)", default=str(img_default)).strip() or str(img_default))
    args.conf = float(prompt_text("Inference conf", default=str(args.conf)).strip() or str(args.conf))
    args.device = prompt_device_selection(
        title="inference devices",
        default_device=str(args.device or default_device_value()),
    )
    args.half = prompt_yes_no("Use FP16 (--half)", default=bool(args.half))
    return True


def _ensure_device_available_or_exit(device: str | None) -> None:
    try:
        validate_device_available(device)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)


def _validate_non_interactive_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.data_mode == "folder" and not args.source_dir:
        parser.error("incomplete arguments: --source-dir is required for --data-mode folder.")
    if args.data_mode == "dataset-split" and not args.dataset:
        parser.error("incomplete arguments: --dataset is required for --data-mode dataset-split.")
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
