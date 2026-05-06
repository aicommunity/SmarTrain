#!/usr/bin/env python3
"""
Single entry point: commands from the workspace directory (SMART_TRAIN_WORKSPACE = cwd by default).
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Callable, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown

from smartrain.core.runtime.interactive_contract import INTERACTIVE_ALLOWED_ENV
from smartrain.core.training.train_backend_registry import default_train_provider
from smartrain.core.training.train_model_catalog import TrainModelCatalog
from smartrain.providers.core.global_index import list_provider_records
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace

app = typer.Typer(
    name="smartrain",
    add_completion=True,
    no_args_is_help=True,
    help="YOLO datasets, training, queue, run analysis. Work from the workspace root.",
)
console = Console()


def _print_en_quick_start() -> None:
    """Print formatted EN quick start in terminal."""
    quickstart_path = Path(__file__).resolve().parent.parent / "docs" / "getting-started" / "quickstart.md"
    fallback = (
        "# Quick start\n\n"
        "Run from workspace root:\n\n"
        "```bash\n"
        "smartrain deploy\n"
        "smartrain scan\n"
        "smartrain train --data my_dataset --model yolo11n.pt -y\n"
        "smartrain report dataset --dataset my_dataset -n 6 --languages en,ru\n"
        "smartrain analyze scan\n"
        "smartrain analyze all --report-languages en,ru\n"
        "```\n"
    )
    try:
        text = quickstart_path.read_text(encoding="utf-8")
    except OSError:
        text = fallback
    console.print(Markdown(text))

HELP_ANALYZE_GROUP = """Analyze training runs: summary tables, comparisons, PR curves, and inference speed.

Quick start:
  smartrain analyze
  smartrain analyze all
  smartrain analyze scan
  smartrain analyze export-table -o runs_summary.csv
  smartrain analyze compare --baseline runs/ds_a/2026-01-01_00-00-00 --others runs/ds_a/2026-01-02_00-00-00
  smartrain analyze inference-benchmark --runs-group-dir runs/ds_a --data-yaml datasets/ds_a/data.yaml
  smartrain analyze leaderboard -o analytics/leaderboard.csv

Common patterns:
  summary CSV: analyze export-table
  quality compare: analyze compare
  speed analysis: analyze inference-benchmark + analyze inference-plot
"""

HELP_QUEUE_GROUP = """Queue management for deferred training runs.

Quick examples:
  smartrain queue list
  smartrain queue add --cmd "smartrain train --data my_dataset -y"
  smartrain queue run --no-gui
"""

HELP_REGISTRY_GROUP = """Registry of runs and promoted models.

Quick examples:
  smartrain registry runs-list
  smartrain registry runs-info --run-dir runs/my_dataset/2026-01-01_00-00-00
  smartrain registry models-list
"""

HELP_REPORT_GROUP = """Dataset sample reports (multilingual Markdown, figures, optional PDF/ODT).

Default output folder: workspace `analytics/datasets-reports/<dataset>_<timestamp>/`.

Quick examples:
  smartrain report dataset --dataset my_dataset
  smartrain report dataset --dataset my_dataset -n 6 --languages en,ru
"""

HELP_MODEL_GROUP = """Model conversion tools.

Quick examples:
  smartrain model convert
  smartrain model convert --input models/best.pt --format onnx
  smartrain model convert --input runs/my_ds/2026-01-01_00-00-00/2026-01-01_00-00-00.pt --format tensorrt-engine --precision fp16
  smartrain model convert --input models/my_model.onnx --format tensorrt-trt
  smartrain model release --run runs/my_ds/2026-01-01_00-00-00

Interactive convert:
  - choose source model type: pt or onnx
  - select a file (or enter a manual path)
  - select one or multiple target models (onnx/engine/trt depending on source; CSV by numbers or values is supported, e.g. 1,3 or onnx,trt)
  - set batch/imgsz and other export parameters
  - run sources use canonical artifacts <run_dir>/<run_dir_name>.<ext>; legacy run layouts are canonized automatically

Artifacts:
  - tensorrt-engine: Ultralytics export to .engine
  - tensorrt-trt: trtexec export to .trt
"""

HELP_DEPS_GROUP = """Dependency management helpers.

Quick examples:
  smartrain deps sync-torch
"""

ARGPARSE_HELP_EXAMPLES: dict[str, str] = {
    "smartrain train": (
        "Examples:\n"
        "  smartrain train --data 2026-01-01_12-00-00-merged -y\n"
        "  smartrain train --data my_dataset --model yolo11n.pt --epochs 50\n"
        "  smartrain train --data my_dataset --batch 16 --img-size 1024\n"
    ),
    "smartrain cvat": (
        "Examples:\n"
        "  smartrain cvat import --cvat-zip task.zip --output-dir datasets/task_yolo\n"
        "  smartrain cvat export --dataset-dir datasets/task_yolo --zip-path task.cvat11.zip\n"
        "  smartrain cvat export --dataset-dir datasets/task_yolo --task-name task42 --names class_a,class_b\n"
    ),
    "smartrain sahi": (
        "Examples:\n"
        "  smartrain sahi --model models/best.pt --source images/\n"
        "  smartrain sahi --model models/best.pt --source image.jpg --output sahi_out\n"
        "  smartrain sahi --model models/best.pt --source images/ --slice-h 768 --slice-w 768\n"
    ),
    "smartrain heatmap": (
        "Examples:\n"
        "  smartrain heatmap --model models/best.pt --source image.jpg\n"
        "  smartrain heatmap --model models/best.pt --source image.jpg --output heatmap.png\n"
        "  smartrain heatmap --model models/best.pt --source image.jpg --colormap 12\n"
    ),
    "smartrain report dataset": (
        "Examples:\n"
        "  smartrain report dataset --dataset my_dataset\n"
        "  smartrain report dataset --dataset my_dataset -n 6 --languages en,ru\n"
        "  smartrain report dataset --workspace /data/ws --dataset my_dataset --no-odt\n"
    ),
    "smartrain inference": (
        "Examples:\n"
        "  smartrain inference --model-name my_promoted_model --data-mode folder --source-dir raw_images\n"
        "  smartrain inference --model-name my_promoted_model --data-mode dataset-split --dataset my_dataset --split test --limit 200\n"
        "  smartrain inference --run 1 --data-mode folder --source-dir samples --roi-pre-detect --roi-weights yolo11n.pt\n"
    ),
}


def _sync_workspace_env(cli_workspace: Optional[str]) -> None:
    w = (cli_workspace or "").strip()
    if w:
        os.environ[WORKSPACE_ENV_VAR] = str(Path(w).resolve())
    elif not (os.environ.get(WORKSPACE_ENV_VAR) or "").strip():
        os.environ[WORKSPACE_ENV_VAR] = str(Path.cwd().resolve())


@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    workspace: Annotated[
        Optional[str],
        typer.Option(
            "--workspace",
            envvar=WORKSPACE_ENV_VAR,
            help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR}, otherwise current directory)",
        ),
    ] = None,
) -> None:
    if getattr(ctx, "resilient_parsing", False):
        return
    _sync_workspace_env(workspace)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


@app.command("deploy")
def cmd_deploy(
    target: Annotated[
        Optional[str],
        typer.Argument(help="Where to expand (default current directory)"),
    ] = None,
) -> None:
    """Create workspace and empty datasets_info.json directories if missing."""
    root = os.path.abspath(os.path.expanduser(target or os.getcwd()))
    info = deploy_workspace(root)
    console.print(f"[blue]Deployment:[/blue] {info['root']}")
    for name in info["created_dirs"]:
        console.print(f"[green]+ directory[/green] {name}")
    for name in info["created_files"]:
        console.print(f"[green] + file[/green] {name}")
    for s in info["skipped"]:
        console.print(f"[yellow]∟ already exists:[/yellow] {s}")
    console.print("[green]Done.[/green]")


@app.command("info")
def cmd_info(
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            help="Training provider key for supported aliases.",
        ),
    ] = default_train_provider(),
) -> None:
    """Show product info and supported train model aliases."""
    try:
        catalog = TrainModelCatalog(provider=provider)
        aliases = tuple(a for a in catalog.supported_aliases() if _is_detection_model_alias(a))
    except ValueError as exc:
        typer.echo(f"[ERROR] {exc}")
        raise typer.Exit(2)
    typer.echo(f"Model source: {provider}")
    typer.echo("Supported train models:")
    for row in _format_columns(aliases):
        typer.echo(row)
    records = [r for r in list_provider_records() if str(r.get("install_state", "")) == "installed"]
    ext_ids = sorted({str(r.get("provider_id", "")).strip() for r in records if str(r.get("provider_id", "")).strip()})
    if ext_ids:
        typer.echo("")
        typer.echo("Supported train models (external providers):")
        for pid in ext_ids:
            typer.echo(f"Model source: {pid}")
            rec = next((x for x in records if str(x.get("provider_id", "")).strip() == pid), None)
            repo_path = str(rec.get("repo_path", "")).strip() if isinstance(rec, dict) else ""
            ext_catalog = TrainModelCatalog(provider=pid, provider_repo_path=repo_path or None)
            ext_base_aliases = tuple(a for a in ext_catalog.supported_aliases() if _is_detection_model_alias(a))
            ext_aliases = tuple(f"{pid}:{a}" for a in ext_base_aliases)
            for row in _format_columns(ext_aliases):
                typer.echo(row)
    typer.echo("")
    typer.echo("Installed external providers:")
    if not records:
        typer.echo("none")
    else:
        for rec in sorted(records, key=lambda x: str(x.get("provider_id", ""))):
            pid = str(rec.get("provider_id", ""))
            repo = str(rec.get("repo_path", ""))
            typer.echo(f"- {pid}: {repo}")


def _format_columns(items: tuple[str, ...], *, max_columns: int = 4) -> list[str]:
    if not items:
        return []
    width = shutil.get_terminal_size(fallback=(100, 20)).columns
    col_width = max(len(x) for x in items) + 2
    if col_width <= 0:
        return list(items)
    cols = max(1, min(max_columns, width // col_width))
    if cols <= 1:
        return list(items)
    rows_count = (len(items) + cols - 1) // cols
    lines: list[str] = []
    for row in range(rows_count):
        parts: list[str] = []
        for col in range(cols):
            idx = col * rows_count + row
            if idx >= len(items):
                continue
            cell = items[idx]
            if col < cols - 1:
                parts.append(cell.ljust(col_width))
            else:
                parts.append(cell)
        lines.append("".join(parts).rstrip())
    return lines


def _is_detection_model_alias(alias: str) -> bool:
    lowered = alias.lower()
    non_detection_markers = ("-seg", "-cls", "-pose", "-obb")
    return not any(marker in lowered for marker in non_detection_markers)


def _invoke_module_main(module: str, args: list[str]) -> None:
    m = importlib.import_module(module)
    fn = getattr(m, "main")
    # Never pass None: argparse would inspect sys.argv of the top-level command.
    fn(args)


@contextmanager
def _interactive_flag_env(allowed: bool):
    prev = os.environ.get(INTERACTIVE_ALLOWED_ENV)
    os.environ[INTERACTIVE_ALLOWED_ENV] = "1" if allowed else "0"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(INTERACTIVE_ALLOWED_ENV, None)
        else:
            os.environ[INTERACTIVE_ALLOWED_ENV] = prev


def _forward_argparse_command(
    ctx: typer.Context,
    *,
    module: str,
    build_parser: Callable[[], object] | None = None,
    prog: str | None = None,
    prepend_args: list[str] | None = None,
    empty_args_mode: str = "help",
) -> None:
    args = list(prepend_args or []) + list(ctx.args)
    non_interactive = any(tok in args for tok in ("-y", "--non-interactive"))
    if non_interactive:
        interactive_allowed = False
    elif len(args) == 0 and empty_args_mode in ("invoke", "invoke_if_tty_else_help"):
        interactive_allowed = True
    else:
        # Allow prompts when the user passed flags (e.g. smartrain test --run ...) from a TTY.
        interactive_allowed = bool(sys.stdin.isatty())
    def _enhance_parser_help(parser_obj: object) -> None:
        if prog is None:
            return
        examples = ARGPARSE_HELP_EXAMPLES.get(prog)
        if not examples:
            return
        if hasattr(parser_obj, "epilog"):
            existing = getattr(parser_obj, "epilog", None)
            setattr(parser_obj, "epilog", f"{existing}\n\n{examples}" if existing else examples)

    if not args:
        if empty_args_mode == "invoke":
            with _interactive_flag_env(interactive_allowed):
                _invoke_module_main(module, args)
            return
        if empty_args_mode == "invoke_if_tty_else_help":
            if sys.stdin.isatty():
                with _interactive_flag_env(interactive_allowed):
                    _invoke_module_main(module, args)
                return
        if build_parser:
            parser = build_parser()
            if prog is not None and hasattr(parser, "prog"):
                parser.prog = prog
            _enhance_parser_help(parser)
            if hasattr(parser, "print_help"):
                parser.print_help()
            raise typer.Exit(0)

    if build_parser and any(tok in ("--help", "-h") for tok in args):
        parser = build_parser()
        if prog is not None and hasattr(parser, "prog"):
            parser.prog = prog
        _enhance_parser_help(parser)
        try:
            parser.parse_args(args)
        except SystemExit as e:
            code = e.code
            if code is None:
                code = 0
            raise typer.Exit(code if isinstance(code, int) else 1)
    with _interactive_flag_env(interactive_allowed):
        _invoke_module_main(module, args)


@app.command(
    "scan",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_scan(ctx: typer.Context) -> None:
    """Scan sources and refresh dataset catalog.

    Examples:
      smartrain scan
      smartrain scan --datasets-list raw_data/datasets_list.txt
      smartrain scan --workspace /data/MarsSmarTrain

    Notes:
      - Synchronizes workspace raw_data and dataset metadata.
      - Optional: --repair-relative-paths / --repair-relative-paths-dry-run normalize stored paths under the workspace.
      - Use --help to inspect all low-level scan flags.
    """
    from smartrain.workflows.datasets.datasets_entry import build_datasets_json_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.datasets_entry",
        build_parser=build_datasets_json_arg_parser,
        prog="smartrain scan",
        empty_args_mode="invoke_if_tty_else_help",
    )


@app.command(
    "normalize-data-yaml",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_normalize_data_yaml(ctx: typer.Context) -> None:
    """Rewrite datasets/*/data.yaml to portable Ultralytics layout (no path key, relative splits).

    Examples:
      smartrain normalize-data-yaml
      smartrain normalize-data-yaml --workspace /data/MarsSmarTrain
      smartrain normalize-data-yaml --datasets-dir /data/MarsSmarTrain/datasets --dry-run
    """
    from smartrain.workflows.datasets.data_yaml_normalize import build_arg_parser

    parser = build_arg_parser()
    if getattr(ctx, "resilient_parsing", False):
        return
    args = parser.parse_args(list(ctx.args))
    from smartrain.workflows.datasets.data_yaml_normalize import run_normalize
    from smartrain.core.runtime.workspace_paths import WorkspaceLayout, resolve_workspace_root

    if args.datasets_dir:
        ddir = os.path.abspath(os.path.expanduser(args.datasets_dir))
    else:
        root = resolve_workspace_root(args.workspace)
        ddir = WorkspaceLayout(root).datasets
    raise typer.Exit(run_normalize(ddir, dry_run=bool(args.dry_run), as_json=bool(args.as_json)))


@app.command(
    "fusion",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_fusion(ctx: typer.Context) -> None:
    """Build a merged training dataset from source datasets.

    Examples:
      smartrain fusion --dataset ds_a --dataset ds_b --classes "class_a,class_b"
      smartrain fusion --dataset ds_a --dataset ds_b --exclude-classes "background,trash"
      smartrain fusion --dataset ds_a --dataset ds_b --output-name merged_ds
      smartrain fusion --workspace /data/MarsSmarTrain --dataset ds_a

    Notes:
      - Produces a new dataset directory under workspace datasets/.
      - Requires datasets/datasets_info.json and datasets/class_names.json in the selected workspace.
      - Check output data.yaml before training.
    """
    from smartrain.workflows.datasets.dataset_former import build_dataset_former_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_former",
        build_parser=build_dataset_former_arg_parser,
        prog="smartrain fusion",
        empty_args_mode="invoke_if_tty_else_help",
    )


@app.command(
    "train",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_train(ctx: typer.Context) -> None:
    """Train YOLO model and save run artifacts.

    Examples:
      smartrain train --data 2026-01-01_12-00-00-merged -y
      smartrain train --data my_dataset --model yolo11n.pt --epochs 50
      smartrain train --data my_dataset --batch 16 --img-size 1024
      smartrain train --workspace /data/MarsSmarTrain --data my_dataset

    Notes:
      - Writes outputs to workspace runs/.
      - Use smartrain analyze scan to inspect completed runs.
    """
    from smartrain.cli_apps.train_app import build_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.cli_apps.train_app",
        build_parser=build_arg_parser,
        prog="smartrain train",
        empty_args_mode="invoke",
    )


@app.command(
    "augment",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_augment(ctx: typer.Context) -> None:
    """Run offline augmentation and save as a new dataset.

    Examples:
      smartrain augment --dataset my_dataset --name my_dataset_aug
      smartrain augment --dataset my_dataset --count 2
      smartrain augment --workspace /data/MarsSmarTrain --dataset my_dataset
    """
    from smartrain.workflows.datasets.dataset_augment import build_augment_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_augment",
        build_parser=build_augment_arg_parser,
        prog="smartrain augment",
        empty_args_mode="invoke_if_tty_else_help",
    )


@app.command(
    "balance",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_balance(ctx: typer.Context) -> None:
    """Balance class distribution and write a new dataset.

    Examples:
      smartrain balance --dataset my_dataset --name my_dataset_balanced
      smartrain balance --dataset my_dataset --target-per-class 1000
      smartrain balance --workspace /data/MarsSmarTrain --dataset my_dataset
    """
    from smartrain.workflows.datasets.dataset_balance import build_balance_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_balance",
        build_parser=build_balance_arg_parser,
        prog="smartrain balance",
        empty_args_mode="invoke_if_tty_else_help",
    )


@app.command(
    "prune",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_prune(ctx: typer.Context) -> None:
    """Prune dataset: remove empty pairs or duplicates.

    Examples:
      smartrain prune empty --dataset my_dataset
      smartrain prune dedup --dataset my_dataset
      smartrain prune dedup --dataset my_dataset --allow-balanced-dedup
    """
    from smartrain.workflows.datasets.dataset_prune import (
        build_prune_arg_parser,
        build_prune_dedup_arg_parser,
        build_prune_empty_arg_parser,
    )

    parser = build_prune_arg_parser
    prog = "smartrain prune"
    if ctx.args and ctx.args[0] == "empty":
        parser = build_prune_empty_arg_parser
        prog = "smartrain prune empty"
    elif ctx.args and ctx.args[0] == "dedup":
        parser = build_prune_dedup_arg_parser
        prog = "smartrain prune dedup"

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_prune",
        build_parser=parser,
        prog=prog,
        empty_args_mode="invoke_if_tty_else_help",
    )


@app.command(
    "hash",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_hash(ctx: typer.Context) -> None:
    """Calculate or validate dataset hash.

    Examples:
      smartrain hash --dataset my_dataset
      smartrain hash /path/to/dataset --validate a1b2c3d4
      smartrain hash --workspace /data/MarsSmarTrain --dataset my_dataset

    Notes:
      - validate exit codes: 0 match, 1 mismatch, 2 error.
    """
    from smartrain.workflows.datasets.dataset_hash import build_hash_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_hash",
        build_parser=build_hash_arg_parser,
        prog="smartrain hash",
    )


@app.command(
    "stats",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_stats(ctx: typer.Context) -> None:
    """Show dataset statistics and compare runs.

    Examples:
      smartrain stats
      smartrain stats --dataset my_dataset
      smartrain stats compare --left ds_a --right ds_b
      smartrain stats --workspace /data/MarsSmarTrain
    """
    from smartrain.workflows.datasets.dataset_stats import build_stats_arg_parser, build_stats_compare_arg_parser

    parser = build_stats_compare_arg_parser if (ctx.args and ctx.args[0] == "compare") else build_stats_arg_parser
    prog = "smartrain stats compare" if (ctx.args and ctx.args[0] == "compare") else "smartrain stats"
    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_stats",
        build_parser=parser,
        prog=prog,
        empty_args_mode="invoke_if_tty_else_help",
    )


@app.command(
    "roi",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_roi(ctx: typer.Context) -> None:
    """Apply ROI crop and export a new dataset.

    Examples:
      smartrain roi --dataset my_dataset --name my_dataset_roi
      smartrain roi --dataset my_dataset --x1 0 --y1 100 --x2 1920 --y2 900
      smartrain roi --workspace /data/MarsSmarTrain --dataset my_dataset
    """
    from smartrain.workflows.datasets.dataset_roi_yolo import build_roi_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_roi_yolo",
        build_parser=build_roi_arg_parser,
        prog="smartrain roi",
        empty_args_mode="invoke_if_tty_else_help",
    )


@app.command(
    "test",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_test(ctx: typer.Context) -> None:
    """Complete missing test artifacts for runs/models."""
    from smartrain.cli_apps.test_app import build_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.cli_apps.test_app",
        build_parser=build_arg_parser,
        prog="smartrain test",
        empty_args_mode="invoke",
    )


@app.command(
    "inference",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_inference(ctx: typer.Context) -> None:
    """Run inference and save JSON report to workspace inference/."""
    from smartrain.cli_apps.inference_app import build_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.cli_apps.inference_app",
        build_parser=build_arg_parser,
        prog="smartrain inference",
        empty_args_mode="invoke",
    )


report_app = typer.Typer(
    help=HELP_REPORT_GROUP,
    invoke_without_command=True,
)


def _report_group_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(HELP_REPORT_GROUP)
        console.print("Run: [cyan]smartrain report dataset --help[/cyan]")
        raise typer.Exit(0)


@report_app.command(
    "dataset",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_report_dataset(ctx: typer.Context) -> None:
    """Multilingual per-class sample report (Markdown + PNG; PDF/ODT via pandoc or extras).

    Examples:
      smartrain report dataset --dataset my_dataset
      smartrain report dataset --dataset my_dataset -n 6 --languages en,ru
      smartrain report dataset --workspace /data/MarsSmarTrain --dataset my_dataset --no-pdf
    """
    from smartrain.workflows.datasets.dataset_report import build_report_dataset_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_report",
        build_parser=build_report_dataset_arg_parser,
        prog="smartrain report dataset",
        empty_args_mode="invoke_if_tty_else_help",
    )


app.add_typer(
    report_app,
    name="report",
    invoke_without_command=True,
    callback=_report_group_callback,
)


queue_app = typer.Typer(
    help=HELP_QUEUE_GROUP,
    invoke_without_command=True,
)


@queue_app.command("list", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_queue_list(ctx: typer.Context) -> None:
    """List queued training tasks and statuses."""
    _invoke_module_main("smartrain.workflows.queue.training_queue_cli", ["list", *list(ctx.args)])


@queue_app.command("add", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_queue_add(ctx: typer.Context) -> None:
    """Add a training command to queue."""
    _invoke_module_main("smartrain.workflows.queue.training_queue_cli", ["add", *list(ctx.args)])


@queue_app.command("remove", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_queue_remove(ctx: typer.Context) -> None:
    """Remove a queued task by id."""
    _invoke_module_main("smartrain.workflows.queue.training_queue_cli", ["remove", *list(ctx.args)])


@queue_app.command("clear", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_queue_clear(ctx: typer.Context) -> None:
    """Remove all tasks from queue."""
    _invoke_module_main("smartrain.workflows.queue.training_queue_cli", ["clear", *list(ctx.args)])


@queue_app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_queue_run_sub(ctx: typer.Context) -> None:
    """Run queue executor through queue subcommand."""
    _invoke_module_main("smartrain.workflows.queue.training_queue_cli", ["run", *list(ctx.args)])


def _queue_group_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        from smartrain.workflows.queue.training_queue_cli import build_queue_cli_arg_parser

        p = build_queue_cli_arg_parser()
        p.prog = "smartrain queue"
        p.print_help()
        raise typer.Exit(0)


app.add_typer(
    queue_app,
    name="queue",
    invoke_without_command=True,
    callback=_queue_group_callback,
)


@app.command(
    "queue-run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_queue_run(ctx: typer.Context) -> None:
    """Run queue executor as a top-level command.

    Examples:
      smartrain queue run --no-gui
      smartrain queue-run --no-gui
      smartrain queue-run --workspace /data/MarsSmarTrain
    """
    from smartrain.workflows.queue.training_queue import build_queue_run_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.queue.training_queue",
        build_parser=build_queue_run_arg_parser,
        prog="smartrain queue-run",
    )


registry_app = typer.Typer(
    help=HELP_REGISTRY_GROUP,
    invoke_without_command=True,
)


@registry_app.command("runs-list", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_registry_runs_list(ctx: typer.Context) -> None:
    """List runs in workspace runs/."""
    _invoke_module_main("smartrain.workflows.registry.registry_cli", ["runs-list", *list(ctx.args)])


@registry_app.command("runs-info", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_registry_runs_info(ctx: typer.Context) -> None:
    """Show training_info and key paths for a run."""
    _invoke_module_main("smartrain.workflows.registry.registry_cli", ["runs-info", *list(ctx.args)])


@registry_app.command("runs-metrics", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_registry_runs_metrics(ctx: typer.Context) -> None:
    """Show metrics files for a run."""
    _invoke_module_main("smartrain.workflows.registry.registry_cli", ["runs-metrics", *list(ctx.args)])


@registry_app.command("models-add", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_registry_models_add(ctx: typer.Context) -> None:
    """Promote model artifact from run into models/."""
    _invoke_module_main("smartrain.workflows.registry.registry_cli", ["models-add", *list(ctx.args)])


@registry_app.command("models-list", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_registry_models_list(ctx: typer.Context) -> None:
    """List promoted models in models/."""
    _invoke_module_main("smartrain.workflows.registry.registry_cli", ["models-list", *list(ctx.args)])


@registry_app.command("models-info", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_registry_models_info(ctx: typer.Context) -> None:
    """Show model manifest for a promoted model."""
    _invoke_module_main("smartrain.workflows.registry.registry_cli", ["models-info", *list(ctx.args)])


@registry_app.command("models-remove", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_registry_models_remove(ctx: typer.Context) -> None:
    """Remove a promoted model from models/."""
    _invoke_module_main("smartrain.workflows.registry.registry_cli", ["models-remove", *list(ctx.args)])


def _registry_group_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        from smartrain.workflows.registry.registry_cli import build_registry_arg_parser

        p = build_registry_arg_parser()
        p.prog = "smartrain registry"
        p.print_help()
        raise typer.Exit(0)


app.add_typer(
    registry_app,
    name="registry",
    invoke_without_command=True,
    callback=_registry_group_callback,
)


providers_app = typer.Typer(
    invoke_without_command=True,
    help="Install/uninstall/status for external providers.",
)


@providers_app.command("install", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_providers_install(ctx: typer.Context) -> None:
    _invoke_module_main("smartrain.providers.cli", ["install", *list(ctx.args)])


@providers_app.command("uninstall", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_providers_uninstall(ctx: typer.Context) -> None:
    _invoke_module_main("smartrain.providers.cli", ["uninstall", *list(ctx.args)])


@providers_app.command("status", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_providers_status(ctx: typer.Context) -> None:
    _invoke_module_main("smartrain.providers.cli", ["status", *list(ctx.args)])


@providers_app.command("doctor", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_providers_doctor(ctx: typer.Context) -> None:
    _invoke_module_main("smartrain.providers.cli", ["doctor", *list(ctx.args)])


def _providers_group_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        from smartrain.providers.cli import build_providers_arg_parser

        p = build_providers_arg_parser()
        p.prog = "smartrain providers"
        p.print_help()
        raise typer.Exit(0)


app.add_typer(
    providers_app,
    name="providers",
    invoke_without_command=True,
    callback=_providers_group_callback,
)


deps_app = typer.Typer(
    invoke_without_command=True,
    help=HELP_DEPS_GROUP,
)


@deps_app.command("sync-torch")
def cmd_deps_sync_torch() -> None:
    """Apply default torch policy: prefer CUDA 12.8, keep existing CUDA 13.x."""
    from smartrain.external_providers.installer import sync_torch_cuda_policy_current_env

    action, message = sync_torch_cuda_policy_current_env()
    prefix = "[SKIPPED]" if action == "skipped" else "[UPDATED]"
    typer.echo(f"{prefix} {message}")


def _deps_group_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo("Usage: smartrain deps [OPTIONS] COMMAND [ARGS]...")
        typer.echo("")
        typer.echo(HELP_DEPS_GROUP)
        raise typer.Exit(0)


app.add_typer(
    deps_app,
    name="deps",
    invoke_without_command=True,
    callback=_deps_group_callback,
)


analyze_app = typer.Typer(
    help=HELP_ANALYZE_GROUP,
    invoke_without_command=True,
)


@analyze_app.command(
    "all",
    short_help="Interactive full analysis orchestrator.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_analyze_all(ctx: typer.Context) -> None:
    """Run end-to-end analysis session and build report artifacts."""
    _invoke_module_main("smartrain.workflows.analyze.analyze_entry", ["all", *list(ctx.args)])


@analyze_app.command(
    "scan",
    short_help="List runs and basic metadata.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_analyze_scan(ctx: typer.Context) -> None:
    """List available runs and basic metadata.

    Examples:
      smartrain analyze scan
      smartrain analyze scan --models-root runs
      smartrain analyze scan --workspace /data/MarsSmarTrain
    """
    _invoke_module_main("smartrain.workflows.analyze.analyze_entry", ["scan", *list(ctx.args)])


@analyze_app.command(
    "export-table",
    short_help="Export summary CSV across runs.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_analyze_export_table(ctx: typer.Context) -> None:
    """Export summary table (CSV) across runs.

    Examples:
      smartrain analyze export-table -o runs_summary.csv
      smartrain analyze export-table --models-root runs --output-dir analytics
      smartrain analyze export-table --workspace /data/MarsSmarTrain
    """
    _invoke_module_main("smartrain.workflows.analyze.analyze_entry", ["export-table", *list(ctx.args)])


@analyze_app.command(
    "compare",
    short_help="Compare selected runs and generate artifacts.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_analyze_compare(ctx: typer.Context) -> None:
    """Compare selected runs and generate comparison artifacts.

    Examples:
      smartrain analyze compare --baseline runs/ds/2026-01-01_00-00-00 --others runs/ds/2026-01-02_00-00-00
      smartrain analyze compare --baseline runs/ds/2026-01-01_00-00-00 --others runs/ds/2026-01-02_00-00-00 -o compare.csv --out-png compare.png
      smartrain analyze compare --workspace /data/MarsSmarTrain --baseline runs/ds/2026-01-01_00-00-00 --others runs/ds/2026-01-02_00-00-00
    """
    _invoke_module_main("smartrain.workflows.analyze.analyze_entry", ["compare", *list(ctx.args)])


@analyze_app.command(
    "pr-curves",
    short_help="Build PR curves for a runs group.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_analyze_pr_curves(ctx: typer.Context) -> None:
    """Generate PR curves for all models in a runs group.

    Examples:
      smartrain analyze pr-curves --runs-group-dir runs/ds_a --data-yaml datasets/ds_a/data.yaml
      smartrain analyze pr-curves --runs-group-dir runs/ds_a --data-yaml datasets/ds_a/data.yaml --out-png analytics/pr.png
      smartrain analyze pr-curves --workspace /data/MarsSmarTrain --runs-group-dir runs/ds_a --data-yaml datasets/ds_a/data.yaml
    """
    _invoke_module_main("smartrain.workflows.analyze.analyze_entry", ["pr-curves", *list(ctx.args)])


@analyze_app.command(
    "inference-benchmark",
    short_help="Benchmark inference speed across models.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_analyze_inference_benchmark(ctx: typer.Context) -> None:
    """Benchmark inference speed across models in runs group.

    Examples:
      smartrain analyze inference-benchmark --runs-group-dir runs/ds_a --data-yaml datasets/ds_a/data.yaml
      smartrain analyze inference-benchmark --runs-group-dir runs/ds_a --data-yaml datasets/ds_a/data.yaml --split test --frames 200
      smartrain analyze inference-benchmark --runs-group-dir runs/ds_a --data-yaml datasets/ds_a/data.yaml --device 0 --half
    """
    _invoke_module_main("smartrain.workflows.analyze.analyze_entry", ["inference-benchmark", *list(ctx.args)])


@analyze_app.command(
    "inference-plot",
    short_help="Plot chart from benchmark CSV.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_analyze_inference_plot(ctx: typer.Context) -> None:
    """Build chart from inference benchmark CSV.

    Examples:
      smartrain analyze inference-plot --csv analytics/inference_tests/ds_a.csv
      smartrain analyze inference-plot --csv analytics/inference_tests/ds_a.csv --metric avg_total_fps
      smartrain analyze inference-plot --csv analytics/inference_tests/ds_a.csv --out-png analytics/ds_a_speed.png
    """
    _invoke_module_main("smartrain.workflows.analyze.analyze_entry", ["inference-plot", *list(ctx.args)])


@analyze_app.command(
    "test-metrics-plot",
    short_help="Plot test metrics from CSV files.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_analyze_test_metrics_plot(ctx: typer.Context) -> None:
    _invoke_module_main("smartrain.workflows.analyze.analyze_entry", ["test-metrics-plot", *list(ctx.args)])


@analyze_app.command(
    "leaderboard",
    short_help="Rank runs by composite score.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_analyze_leaderboard(ctx: typer.Context) -> None:
    """Build leaderboard CSV from run metrics.

    Examples:
      smartrain analyze leaderboard -o analytics/leaderboard.csv
      smartrain analyze leaderboard --quality-metric mAP50-95 --speed-metric avg_inference_fps
      smartrain analyze leaderboard --weight-quality 0.7 --weight-speed 0.2 --weight-stability 0.1
    """
    _invoke_module_main("smartrain.workflows.analyze.analyze_entry", ["leaderboard", *list(ctx.args)])


def _analyze_group_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        if sys.stdin.isatty():
            with _interactive_flag_env(True):
                _invoke_module_main("smartrain.workflows.analyze.analyze_entry", ["all"])
            raise typer.Exit(0)
        console.print(
            "[red][ERROR][/red] `smartrain analyze` без подкоманды требует интерактивный терминал (TTY). "
            "Используйте явную подкоманду, например `smartrain analyze scan`."
        )
        raise typer.Exit(2)


app.add_typer(
    analyze_app,
    name="analyze",
    invoke_without_command=True,
    callback=_analyze_group_callback,
)

model_app = typer.Typer(
    help=HELP_MODEL_GROUP,
    invoke_without_command=True,
)


@model_app.command(
    "convert",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_model_convert(ctx: typer.Context) -> None:
    """Convert `.pt`/`.onnx` models to ONNX and TensorRT.

    Examples:
      smartrain model convert --input models/best.pt --format onnx
      smartrain model convert --input runs/my_dataset/2026-01-01_00-00-00/2026-01-01_00-00-00.pt --format tensorrt-engine --precision fp16
      smartrain model convert --input models/my_model.onnx --format tensorrt-trt
      smartrain model convert
    """
    from smartrain.workflows.models.model_convert_cli import build_model_convert_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.models.model_convert_cli",
        build_parser=build_model_convert_arg_parser,
        prog="smartrain model convert",
        empty_args_mode="invoke_if_tty_else_help",
    )


@model_app.command(
    "release",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_model_release(ctx: typer.Context) -> None:
    """Release canonical run `.pt` into workspace models catalog.

    Examples:
      smartrain model release --run runs/my_dataset/2026-01-01_00-00-00
      smartrain model release --run 1
      smartrain model release
    """
    from smartrain.workflows.models.model_release_cli import build_model_release_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.models.model_release_cli",
        build_parser=build_model_release_arg_parser,
        prog="smartrain model release",
        empty_args_mode="invoke_if_tty_else_help",
    )


def _model_group_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(HELP_MODEL_GROUP)
        console.print("Run: [cyan]smartrain model convert --help[/cyan]")
        console.print("Run: [cyan]smartrain model release --help[/cyan]")
        raise typer.Exit(0)


app.add_typer(
    model_app,
    name="model",
    invoke_without_command=True,
    callback=_model_group_callback,
)


@app.command(
    "plot",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_plot(ctx: typer.Context) -> None:
    """Legacy analytics wrapper.

    Examples:
      smartrain plot scan
      smartrain plot compare --baseline runs/ds/2026-01-01_00-00-00 --others runs/ds/2026-01-02_00-00-00
      smartrain analyze compare --baseline runs/ds/2026-01-01_00-00-00 --others runs/ds/2026-01-02_00-00-00

    Notes:
      - Prefer `smartrain analyze ...` for new workflows.
    """
    from smartrain.workflows.analyze.analyze_entry import build_analyze_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.analyze.plot_creator",
        build_parser=build_analyze_arg_parser,
        prog="smartrain plot",
    )


@app.command(
    "migrate",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_migrate(ctx: typer.Context) -> None:
    """Canonical migration utilities.

    Examples:
      smartrain migrate canonical --mode dry-run
      smartrain migrate canonical --mode apply --continue-on-error
      smartrain migrate canonical --source-kind run --report analytics/migration-reports/run-only.json
    """
    from smartrain.workflows.migration.cli_migration import build_migration_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.migration.cli_migration",
        build_parser=build_migration_arg_parser,
        prog="smartrain migrate",
    )


@app.command(
    "migrate-models",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_migrate_models(ctx: typer.Context) -> None:
    """Migrate legacy models for analyze compatibility.

    Examples:
      smartrain migrate-models --models-root models
      smartrain migrate-models --workspace /data/MarsSmarTrain
      smartrain migrate-models --dry-run
    """
    from smartrain.workflows.migration.migrate_models_to_smartrain import build_migrate_models_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.migration.migrate_models_to_smartrain",
        build_parser=build_migrate_models_arg_parser,
        prog="smartrain migrate-models",
    )


@app.command(
    "cvat",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_cvat(ctx: typer.Context) -> None:
    """Convert datasets between CVAT 1.1 and YOLO formats.

    Examples:
      smartrain cvat import --cvat-zip task.zip --output-dir datasets/task_yolo
      smartrain cvat export --dataset-dir datasets/task_yolo --zip-path task.cvat11.zip
      smartrain cvat export --dataset-dir datasets/task_yolo --task-name task42 --names class_a,class_b
      smartrain cvat --help

    Notes:
      - Subcommands: import, export.
      - Use --tmp-dir to control temporary workspace.
    """
    from smartrain.workflows.datasets.cvat_cli import build_cvat_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.cvat_cli",
        build_parser=build_cvat_arg_parser,
        prog="smartrain cvat",
    )


@app.command(
    "clearml-upload",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_clearml_upload(ctx: typer.Context) -> None:
    """Upload completed run to ClearML.

    Examples:
      smartrain clearml-upload runs/my_dataset/2026-01-01_00-00-00
      smartrain clearml-upload runs/my_dataset/2026-01-01_00-00-00 --project Mars --task-name yolo-exp-1
      smartrain clearml-upload runs/my_dataset/2026-01-01_00-00-00 --no-images

    Notes:
      - Requires ClearML extras: pip install -e ".[clearml]".
    """
    from smartrain.workflows.analyze.clearml_upload import build_clearml_upload_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.analyze.clearml_upload",
        build_parser=build_clearml_upload_arg_parser,
        prog="smartrain clearml-upload",
    )


@app.command(
    "sahi",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_sahi(ctx: typer.Context) -> None:
    """Run tiled inference with SAHI for large images.

    Examples:
      smartrain sahi --model models/best.pt --source images/
      smartrain sahi --model models/best.pt --source image.jpg --output sahi_out
      smartrain sahi --model models/best.pt --source images/ --slice-h 768 --slice-w 768 --overlap-h 0.25 --overlap-w 0.25

    Notes:
      - Requires SAHI extras: pip install -e ".[sahi]".
    """
    from smartrain.workflows.inference.sahi_cli import build_sahi_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.inference.sahi_cli",
        build_parser=build_sahi_arg_parser,
        prog="smartrain sahi",
    )


@app.command(
    "heatmap",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_heatmap(ctx: typer.Context) -> None:
    """Generate heatmap visualization from image and model.

    Examples:
      smartrain heatmap --model models/best.pt --source image.jpg
      smartrain heatmap --model models/best.pt --source image.jpg --output heatmap.png
      smartrain heatmap --model models/best.pt --source image.jpg --colormap 12
    """
    from smartrain.workflows.inference.heatmap_cli import build_heatmap_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.inference.heatmap_cli",
        build_parser=build_heatmap_arg_parser,
        prog="smartrain heatmap",
    )


@app.command(
    "orient",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_orient(ctx: typer.Context) -> None:
    """Normalize image orientation into a new dataset.

    Examples:
      smartrain orient --dataset my_dataset --name my_dataset_oriented
      smartrain orient --dataset my_dataset --angles 0,90,180,270
      smartrain orient --workspace /data/MarsSmarTrain --dataset my_dataset
    """
    from smartrain.workflows.datasets.dataset_orient import build_orient_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_orient",
        build_parser=build_orient_arg_parser,
        prog="smartrain orient",
        empty_args_mode="invoke_if_tty_else_help",
    )


def main() -> None:
    if len(sys.argv) == 1:
        _print_en_quick_start()
        return
    app()


if __name__ == "__main__":
    main()
