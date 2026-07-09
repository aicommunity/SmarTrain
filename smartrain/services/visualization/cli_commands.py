from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import (
    print_numbered_options,
    prompt_choice,
    prompt_multi_choice_csv,
    prompt_text,
)
from smartrain.core.runtime.device_selector import default_device_value, prompt_device_selection
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.run_discovery import find_run_directories
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.services.datasets.dataset_cli_catalog import load_datasets_catalog
from smartrain.services.inference_runtime_helpers import discover_model_entries
from smartrain.services.visualization.contracts import VisRequest
from smartrain.services.visualization.pipeline import visualize_dataset, visualize_model, visualize_run
from smartrain.services.visualization.target_resolution import resolve_dataset_target, resolve_model_target, resolve_run_target


def build_vis_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Visualize dataset labels and model/run predictions.")
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR}).")
    p.add_argument("-y", "--non-interactive", "--nit", action="store_true", dest="non_interactive")
    p.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing rendered files.")
    sub = p.add_subparsers(dest="mode")

    p_ds = sub.add_parser("dataset", help="Visualize ground-truth labels for a dataset.")
    p_ds.add_argument("--dataset", type=str, default=None, help="Dataset key from datasets_info.json or dataset path.")
    p_ds.add_argument("--splits", type=str, default=None, help="CSV split names (default: all discovered in data.yaml).")

    p_model = sub.add_parser("model", help="Visualize GT + predictions for a model target.")
    p_model.add_argument("--model-name", type=str, default=None, help="Promoted model name from workspace/models.")
    p_model.add_argument("--weights", type=str, default=None, help="Explicit weights path.")
    p_model.add_argument("--run", type=str, default=None, help="Optional run for dataset/source resolution.")
    p_model.add_argument("--splits", type=str, default=None, help="CSV split names.")
    p_model.add_argument("--device", type=str, default=None, help="Inference device (cpu, 0, etc).")
    p_model.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions.")

    p_run = sub.add_parser("run", help="Visualize GT + predictions for a run target.")
    p_run.add_argument("--run", type=str, default=None, help="Run path or run index from workspace/runs.")
    p_run.add_argument("--splits", type=str, default=None, help="CSV split names.")
    p_run.add_argument("--device", type=str, default=None, help="Inference device (cpu, 0, etc).")
    p_run.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions.")
    return p


def _parse_splits(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    parts = tuple(x.strip() for x in str(raw).split(",") if x.strip())
    return parts or None


def _interactive_allowed(argv: list[str], args: argparse.Namespace) -> bool:
    return bool(is_interactive_allowed(argv) and sys.stdin.isatty() and not args.non_interactive)


def _prompt_vis_mode_if_missing(args: argparse.Namespace) -> str:
    modes = ["dataset", "model", "run"]
    choice = prompt_choice("What to visualize", modes, default="dataset")
    args.mode = choice
    return choice


def _prompt_splits_if_missing(args: argparse.Namespace) -> None:
    if args.splits:
        return
    selected = prompt_multi_choice_csv(
        "Splits",
        ["train", "val", "test"],
        default_values=["train", "val", "test"],
    )
    args.splits = ",".join(selected)


def _prompt_confidence(args: argparse.Namespace) -> None:
    default_conf = float(getattr(args, "conf", 0.25) if getattr(args, "conf", None) is not None else 0.25)
    while True:
        raw = prompt_text("Confidence threshold", default=str(default_conf)).strip()
        try:
            value = float(raw) if raw else default_conf
        except Exception:
            print(f"[ERROR] Invalid confidence value: {raw!r}")
            continue
        if value < 0.0 or value > 1.0:
            print(f"[ERROR] Confidence must be in [0, 1], got: {value}")
            continue
        args.conf = value
        return


def _pick_interactive_model(args: argparse.Namespace, layout: WorkspaceLayout) -> None:
    model_entries = discover_model_entries(layout)
    model_options = ["models", "weights"]
    model_source = "models" if model_entries else "weights"
    model_source = prompt_choice("Model source", model_options, default=model_source)
    args.model_name = None
    args.weights = None
    if model_source == "models":
        if not model_entries:
            raise RuntimeError("No model files found in workspace/models.")
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
            raise RuntimeError("Internal error: selected model not found.")
        args.model_name = selected[2]
        return
    args.weights = prompt_text("Weights path", default="models").strip() or "models"


def _pick_interactive_run(args: argparse.Namespace, layout: WorkspaceLayout) -> None:
    runs = find_run_directories(layout.runs)
    if not runs:
        raise RuntimeError("No runs found in workspace/runs.")
    pretty: list[str] = []
    opts: list[str] = []
    for i, run_dir in enumerate(runs, start=1):
        rel = os.path.relpath(run_dir, layout.root)
        pretty.append(f"{i}. {rel}")
        opts.append(str(i))
    print("[INFO] Available runs:")
    for row in pretty:
        print(f"  {row}")
    args.run = prompt_choice("Select run index", opts, default=opts[0], show_options=False)


def _fill_dataset_interactive(args: argparse.Namespace, layout: WorkspaceLayout) -> None:
    catalog = load_datasets_catalog(layout)
    if not args.dataset:
        if catalog:
            names = sorted(catalog.keys())
            print_numbered_options("datasets", names)
            args.dataset = prompt_choice("Dataset", names, default=names[0], show_options=False)
        else:
            args.dataset = prompt_text("Dataset path", default=str(Path(layout.datasets))).strip()
    _prompt_splits_if_missing(args)


def _fill_model_interactive(args: argparse.Namespace, layout: WorkspaceLayout) -> None:
    if not args.model_name and not args.weights:
        _pick_interactive_model(args, layout)
    if not args.run:
        if prompt_choice("Use explicit run for dataset resolution", ["no", "yes"], default="no") == "yes":
            run_args = argparse.Namespace(run=None)
            _pick_interactive_run(run_args, layout)
            args.run = run_args.run
    _prompt_splits_if_missing(args)
    if args.device is None:
        args.device = prompt_device_selection(
            title="visualization devices",
            default_device=default_device_value(),
        )
    _prompt_confidence(args)


def _fill_run_interactive(args: argparse.Namespace, layout: WorkspaceLayout) -> None:
    if not args.run:
        _pick_interactive_run(args, layout)
    _prompt_splits_if_missing(args)
    if args.device is None:
        args.device = prompt_device_selection(
            title="visualization devices",
            default_device=default_device_value(),
        )
    _prompt_confidence(args)


def _ensure_mode_defaults(args: argparse.Namespace) -> None:
    if args.mode == "dataset":
        args.dataset = getattr(args, "dataset", None)
        args.splits = getattr(args, "splits", None)
        return
    if args.mode == "model":
        args.model_name = getattr(args, "model_name", None)
        args.weights = getattr(args, "weights", None)
        args.run = getattr(args, "run", None)
        args.splits = getattr(args, "splits", None)
        args.device = getattr(args, "device", None)
        args.conf = getattr(args, "conf", 0.25)
        return
    if args.mode == "run":
        args.run = getattr(args, "run", None)
        args.splits = getattr(args, "splits", None)
        args.device = getattr(args, "device", None)
        args.conf = getattr(args, "conf", 0.25)


def _needs_interactive_fill(args: argparse.Namespace) -> bool:
    _ensure_mode_defaults(args)
    if args.mode == "dataset":
        return not args.dataset
    if args.mode == "model":
        return not args.model_name and not args.weights
    if args.mode == "run":
        return not args.run
    return False


def _build_request(args: argparse.Namespace, mode: str, workspace_root: Path) -> VisRequest:
    return VisRequest(
        mode=mode,  # type: ignore[arg-type]
        workspace_root=workspace_root,
        dataset=getattr(args, "dataset", None),
        model_name=getattr(args, "model_name", None),
        run_ref=getattr(args, "run", None),
        weights=getattr(args, "weights", None),
        splits=_parse_splits(getattr(args, "splits", None)),
        limit=(None if int(getattr(args, "limit", 0) or 0) <= 0 else int(getattr(args, "limit", 0))),
        conf=getattr(args, "conf", None),
        device=getattr(args, "device", None),
        overwrite=bool(getattr(args, "overwrite", False)),
        non_interactive=bool(getattr(args, "non_interactive", False)),
    )


def cmd_vis_dataset(args: argparse.Namespace, layout: WorkspaceLayout, req: VisRequest) -> int:
    target = resolve_dataset_target(layout, req)
    target["layout"] = layout
    return visualize_dataset(req, target)


def cmd_vis_model(args: argparse.Namespace, layout: WorkspaceLayout, req: VisRequest) -> int:
    target = resolve_model_target(layout, req)
    target["layout"] = layout
    return visualize_model(req, target)


def cmd_vis_run(args: argparse.Namespace, layout: WorkspaceLayout, req: VisRequest) -> int:
    target = resolve_run_target(layout, req)
    target["layout"] = layout
    if "model_path" not in target:
        # Reuse canonical run model path through inference resolver path.
        from smartrain.services.inference_runtime_helpers import resolve_model

        model_args = type("RunModelArgs", (), {"model_name": None, "run": req.run_ref, "weights": None})()
        model_path, _model_name, _source = resolve_model(model_args, layout)
        target["model_path"] = model_path
    return visualize_run(req, target)


def run_vis_cli(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    parser = build_vis_arg_parser()
    args = parser.parse_args(argv)
    interactive = _interactive_allowed(argv, args)
    try:
        workspace_root = Path(resolve_workspace_root(args.workspace))
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
    layout = WorkspaceLayout(str(workspace_root))
    if not args.mode:
        if interactive:
            print("[INFO] Interactive visualization mode (Enter = default).")
            _prompt_vis_mode_if_missing(args)
        else:
            parser.print_help()
            return 0
    _ensure_mode_defaults(args)
    if interactive and _needs_interactive_fill(args):
        print("[INFO] Interactive visualization mode (Enter = default).")
    try:
        if args.mode == "dataset":
            if interactive and _needs_interactive_fill(args):
                _fill_dataset_interactive(args, layout)
            req = _build_request(args, "dataset", workspace_root)
            return cmd_vis_dataset(args, layout, req)
        if args.mode == "model":
            if interactive and _needs_interactive_fill(args):
                _fill_model_interactive(args, layout)
            req = _build_request(args, "model", workspace_root)
            return cmd_vis_model(args, layout, req)
        if args.mode == "run":
            if interactive and _needs_interactive_fill(args):
                _fill_run_interactive(args, layout)
            req = _build_request(args, "run", workspace_root)
            return cmd_vis_run(args, layout, req)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    parser.error(f"Unknown mode: {args.mode}")
    return 2

