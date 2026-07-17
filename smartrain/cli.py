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

from smartrain.cli_entrypoints.grouped_help import plain_sub_typer, plain_typer
from smartrain.cli_entrypoints.support.typer_non_interactive import (
    env_forces_non_interactive_cli,
    strip_typer_meta_non_interactive_flags,
)
from smartrain.core.runtime.completion_autoinstall import ensure_completion_auto_setup
from smartrain.core.runtime.interactive_contract import INTERACTIVE_ALLOWED_ENV
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace

app = plain_typer(
    name="smartrain",
    add_completion=True,
    help="YOLO datasets, training, queue, run analysis. Work from the workspace root.",
)


from smartrain.cli_entrypoints.cli_forwarding import (
    _format_columns,
    _forward_analyze_command,
    _forward_argparse_command,
    _interactive_flag_env,
    _invoke_module_main,
    _is_detection_model_alias,
)
from smartrain.cli_entrypoints.help_texts import (
    ARGPARSE_HELP_EXAMPLES,
    HELP_ANALYZE_GROUP,
    HELP_DATASET_GROUP,
    HELP_DEPS_GROUP,
    HELP_MODEL_GROUP,
    HELP_QUEUE_GROUP,
    HELP_REGISTRY_GROUP,
    _print_en_quick_start,
)

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
    ensure_completion_auto_setup(sys.argv[1:])
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
    typer.echo(f"Deployment: {info['root']}")
    for name in info["created_dirs"]:
        typer.echo(f"+ directory {name}")
    for name in info["created_files"]:
        typer.echo(f"+ file {name}")
    for s in info["skipped"]:
        typer.echo(f"already exists: {s}")
    typer.echo("Done.")


workspace_app = plain_sub_typer(help="Workspace coordination and status.")


@workspace_app.command(
    "peers",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_workspace_peers(ctx: typer.Context) -> None:
    """List active workspace sessions and lock files."""
    from smartrain.workflows.workspace.workspace_peers_cli import build_workspace_peers_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.workspace.workspace_peers_cli",
        build_parser=build_workspace_peers_arg_parser,
        prog="smartrain workspace peers",
        empty_args_mode="invoke",
    )


app.add_typer(
    workspace_app,
    name="workspace",
    help="Workspace coordination: active peers, locks, shared-folder safety.",
)


@app.command("quickstart")
def cmd_quickstart() -> None:
    """Print step-by-step getting-started workflow guide."""
    _print_en_quick_start()


@app.command("info")
def cmd_info(
    provider: Annotated[
        Optional[str],
        typer.Option(
            "--provider",
            help="Training provider key for supported aliases.",
        ),
    ] = None,
) -> None:
    """Show product info and supported train model aliases."""
    from smartrain.core.training.train_model_catalog import TrainModelCatalog
    from smartrain.core.training.ultralytics_model_alias_registry import default_train_provider
    from smartrain.providers.core.global_index import list_provider_records

    provider = (provider or default_train_provider()).strip()
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
    "sync",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_sync(ctx: typer.Context) -> None:
    """Safely pull missing artifacts from another workspace copy."""
    from smartrain.services.workspace.workspace_sync_service import build_sync_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.services.workspace.workspace_sync_service",
        build_parser=build_sync_arg_parser,
        prog="smartrain sync",
        empty_args_mode="invoke",
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
    from smartrain.services.datasets.data_yaml_normalize import build_arg_parser

    parser = build_arg_parser()
    if getattr(ctx, "resilient_parsing", False):
        return
    args = parser.parse_args(list(ctx.args))
    from smartrain.services.datasets.data_yaml_normalize import run_normalize
    from smartrain.core.runtime.workspace_paths import WorkspaceLayout, resolve_workspace_root

    if args.datasets_dir:
        ddir = os.path.abspath(os.path.expanduser(args.datasets_dir))
    else:
        root = resolve_workspace_root(args.workspace)
        ddir = WorkspaceLayout(root).datasets
    raise typer.Exit(run_normalize(ddir, dry_run=bool(args.dry_run), as_json=bool(args.as_json)))


@app.command(
    "merge",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_merge(ctx: typer.Context) -> None:
    """Build a merged training dataset from source datasets.

    Examples:
      smartrain merge --dataset ds_a --dataset ds_b --classes "class_a,class_b"
      smartrain merge --dataset ds_a --dataset ds_b --exclude-classes "background,trash"
      smartrain merge --dataset ds_a --dataset ds_b --output-name merged_ds
      smartrain merge --workspace /data/MarsSmarTrain --dataset ds_a

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
        prog="smartrain merge",
        empty_args_mode="invoke_if_tty_else_help",
        ensure_scan=True,
    )


@app.command(
    "fusion",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_fusion(ctx: typer.Context) -> None:
    """Deprecated alias for `smartrain merge`."""
    typer.echo("[DEPRECATION] smartrain fusion is deprecated; use smartrain merge.")
    cmd_merge(ctx)


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
    from smartrain.cli_entrypoints.train_app import build_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.cli_entrypoints.train_app",
        build_parser=build_arg_parser,
        prog="smartrain train",
        empty_args_mode="invoke",
        ensure_scan=True,
    )


@app.command(
    "augment",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_augment(ctx: typer.Context) -> None:
    """Run offline augmentation and save as a new dataset.

    Examples:
      smartrain augment --dataset my_dataset --output-name my_dataset_aug
      smartrain augment --dataset my_dataset --enable-flip --flip-sampling exhaustive
      smartrain augment --workspace /data/MarsSmarTrain --dataset my_dataset
    """
    from smartrain.workflows.datasets.dataset_augment import build_augment_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_augment",
        build_parser=build_augment_arg_parser,
        prog="smartrain augment",
        empty_args_mode="invoke_if_tty_else_help",
        ensure_scan=True,
    )


@app.command(
    "split",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_split(ctx: typer.Context) -> None:
    """Repartition one dataset into train/valid/test splits.

    Examples:
      smartrain split --dataset my_dataset --split-ratio 0.7,0.2,0.1
      smartrain split --dataset my_dataset --exclude-test --output-name my_dataset_resplit
      smartrain split --workspace /data/MarsSmarTrain --dataset my_dataset
    """
    from smartrain.workflows.datasets.dataset_split import build_split_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_split",
        build_parser=build_split_arg_parser,
        prog="smartrain split",
        empty_args_mode="invoke_if_tty_else_help",
        ensure_scan=True,
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
        ensure_scan=True,
    )


@app.command(
    "prune",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_prune(ctx: typer.Context) -> None:
    """Prune dataset: remove empty pairs, duplicates, unused classes, or small labels.

    Examples:
      smartrain prune empty --dataset my_dataset
      smartrain prune dedup --dataset my_dataset
      smartrain prune classes --dataset my_dataset
      smartrain prune size --dataset my_dataset
      smartrain prune size --dataset my_dataset --min-size 12x18 --size-mode and --no-drop-empty-images
      smartrain prune dedup --dataset my_dataset --allow-balanced-dedup
    """
    from smartrain.workflows.datasets.dataset_prune import (
        build_prune_arg_parser,
        build_prune_classes_arg_parser,
        build_prune_dedup_arg_parser,
        build_prune_empty_arg_parser,
        build_prune_size_arg_parser,
    )

    parser = build_prune_arg_parser
    prog = "smartrain prune"
    if ctx.args and ctx.args[0] == "empty":
        parser = build_prune_empty_arg_parser
        prog = "smartrain prune empty"
    elif ctx.args and ctx.args[0] == "dedup":
        parser = build_prune_dedup_arg_parser
        prog = "smartrain prune dedup"
    elif ctx.args and ctx.args[0] == "classes":
        parser = build_prune_classes_arg_parser
        prog = "smartrain prune classes"
    elif ctx.args and ctx.args[0] == "size":
        parser = build_prune_size_arg_parser
        prog = "smartrain prune size"

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_prune",
        build_parser=parser,
        prog=prog,
        empty_args_mode="invoke_if_tty_else_help",
        ensure_scan=True,
    )


@app.command(
    "filter",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_filter(ctx: typer.Context) -> None:
    """Filter edge-truncated bbox annotations into a new dataset.

    Examples:
      smartrain filter --dataset my_dataset
      smartrain filter --dataset my_dataset --stats-only
      smartrain filter --dataset my_dataset --dry-run
    """
    from smartrain.workflows.datasets.dataset_filter import build_filter_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_filter",
        build_parser=build_filter_arg_parser,
        prog="smartrain filter",
        empty_args_mode="invoke_if_tty_else_help",
        ensure_scan=True,
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
    from smartrain.services.datasets.dataset_hash import build_hash_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.services.datasets.dataset_hash",
        build_parser=build_hash_arg_parser,
        prog="smartrain hash",
        ensure_scan=True,
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
    from smartrain.services.datasets.dataset_stats import build_stats_arg_parser, build_stats_compare_arg_parser

    parser = build_stats_compare_arg_parser if (ctx.args and ctx.args[0] == "compare") else build_stats_arg_parser
    prog = "smartrain stats compare" if (ctx.args and ctx.args[0] == "compare") else "smartrain stats"
    _forward_argparse_command(
        ctx,
        module="smartrain.services.datasets.dataset_stats",
        build_parser=parser,
        prog=prog,
        empty_args_mode="invoke_if_tty_else_help",
        ensure_scan=True,
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
        ensure_scan=True,
    )


@app.command(
    "test",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_test(ctx: typer.Context) -> None:
    """Complete missing test artifacts for runs/models."""
    from smartrain.cli_entrypoints.test_app import build_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.cli_entrypoints.test_app",
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
    from smartrain.cli_entrypoints.inference_app import build_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.cli_entrypoints.inference_app",
        build_parser=build_arg_parser,
        prog="smartrain inference",
        empty_args_mode="invoke",
        ensure_scan=True,
    )


@app.command(
    "vis",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_vis(ctx: typer.Context) -> None:
    """Visualize dataset labels and model/run predictions."""
    from smartrain.workflows.visualization.vis_cli import build_vis_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.visualization.vis_cli",
        build_parser=build_vis_arg_parser,
        prog="smartrain vis",
        empty_args_mode="invoke",
        ensure_scan=True,
    )


queue_app = plain_sub_typer(
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


registry_app = plain_sub_typer(
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


providers_app = plain_sub_typer(
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


deps_app = plain_sub_typer(
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


@deps_app.command("doctor")
def cmd_deps_doctor(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Print full detail strings for each dependency row."),
    ] = False,
) -> None:
    """Check optional report-export dependencies (pandoc, weasyprint, fallbacks)."""
    from smartrain.services.deps.optional_extras import check_export_deps, ubuntu_weasyprint_apt_hint

    report = check_export_deps()
    typer.echo("[INFO] Export dependencies doctor:")
    for row in report.rows:
        status = "ok" if row.ok else "missing"
        if verbose or not row.ok:
            typer.echo(f"  - {row.name}: {status} ({row.detail})")
        else:
            typer.echo(f"  - {row.name}: {status}")
    if not report.export_ready:
        typer.echo("[INFO] Reinstall smartrain to restore bundled pandoc: pip install -e .")
    weasy_row = next((r for r in report.rows if r.name.startswith("weasyprint")), None)
    if weasy_row is not None and not weasy_row.ok:
        typer.echo("[INFO] Optional WeasyPrint PDF engine: smartrain deps install")
        hint = ubuntu_weasyprint_apt_hint()
        if hint:
            typer.echo(f"[INFO] Ubuntu/Debian WeasyPrint system libraries: {hint}")
    raise typer.Exit(0 if report.export_ready else 1)


@deps_app.command("install")
def cmd_deps_install(
    extra: Annotated[
        list[str],
        typer.Option("--extra", help="Optional extra to install (repeatable). Default: export."),
    ] = [],
    all_extras: Annotated[
        bool,
        typer.Option("--all-extras", help="Install all known optional extras."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print pip command without running it."),
    ] = False,
) -> None:
    """Install optional pip extras (default: export/weasyprint for enhanced PDF export)."""
    from smartrain.services.deps.optional_extras import (
        check_export_deps,
        install_optional_extras,
        known_optional_extras,
    )

    if all_extras:
        selected = list(known_optional_extras())
    elif extra:
        selected = list(extra)
    else:
        selected = ["export"]
    try:
        cmd = install_optional_extras(selected, dry_run=dry_run)
    except ValueError as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(2) from exc
    if dry_run:
        typer.echo(f"[INFO] Would run: {cmd}")
        raise typer.Exit(0)
    typer.echo(f"[OK] Installed extras: {', '.join(selected)}")
    report = check_export_deps()
    if report.export_ready:
        typer.echo("[OK] pandoc is available for report export.")
    else:
        typer.echo("[WARN] pandoc still unavailable; run: smartrain deps doctor --verbose", err=True)
        raise typer.Exit(1)


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


analyze_app = plain_sub_typer(
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
    _forward_analyze_command(ctx, "all", prog="smartrain analyze all")


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
    _forward_analyze_command(ctx, "scan", prog="smartrain analyze scan")


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
    _forward_analyze_command(ctx, "export-table", prog="smartrain analyze export-table")


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
    _forward_analyze_command(ctx, "compare", prog="smartrain analyze compare")


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
    _forward_analyze_command(ctx, "pr-curves", prog="smartrain analyze pr-curves")


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
    _forward_analyze_command(ctx, "inference-benchmark", prog="smartrain analyze inference-benchmark")


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
    _forward_analyze_command(ctx, "inference-plot", prog="smartrain analyze inference-plot")


@analyze_app.command(
    "test-metrics-plot",
    short_help="Plot test metrics from CSV files.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_analyze_test_metrics_plot(ctx: typer.Context) -> None:
    _forward_analyze_command(ctx, "test-metrics-plot", prog="smartrain analyze test-metrics-plot")


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
    _forward_analyze_command(ctx, "leaderboard", prog="smartrain analyze leaderboard")


def _analyze_group_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        if sys.stdin.isatty():
            with _interactive_flag_env(True):
                _invoke_module_main("smartrain.workflows.analyze.analyze_entry", ["all"])
            raise typer.Exit(0)
        typer.echo(
            "[ERROR] `smartrain analyze` without a subcommand requires an interactive terminal (TTY). "
            "Use an explicit subcommand, e.g. `smartrain analyze scan`."
        )
        raise typer.Exit(2)


app.add_typer(
    analyze_app,
    name="analyze",
    invoke_without_command=True,
    callback=_analyze_group_callback,
)

model_app = plain_sub_typer(
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


@model_app.command(
    "comment",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_model_comment(ctx: typer.Context) -> None:
    """Set or update a one-line comment for a released workspace model.

    Examples:
      smartrain model comment --release models/my_ds/2026-01-01_00-00-00_ultralytics_yolo11s_640px_100epochs_b16-abcd1234/detect_yolo11s_20260101_000000_640px_100epochs_b16.pt --comment "Production line 3"
      smartrain model comment --release 1 --comment "Updated note"
      smartrain model comment
    """
    from smartrain.workflows.models.model_comment_cli import build_model_comment_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.models.model_comment_cli",
        build_parser=build_model_comment_arg_parser,
        prog="smartrain model comment",
        empty_args_mode="invoke_if_tty_else_help",
    )


@model_app.command(
    "unrelease",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_model_unrelease(ctx: typer.Context) -> None:
    """Move a released model back into runs/ and remove it from the release catalog.

    Examples:
      smartrain model unrelease --release models/my_ds/run_id/models/detect_yolo11s_20260101_000000_640px_100epochs_b16.pt --yes
      smartrain model unrelease --release 1 --yes
      smartrain model unrelease
    """
    from smartrain.workflows.models.model_unrelease_cli import build_model_unrelease_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.models.model_unrelease_cli",
        build_parser=build_model_unrelease_arg_parser,
        prog="smartrain model unrelease",
        empty_args_mode="invoke_if_tty_else_help",
    )


@model_app.command(
    "rename",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_model_rename(ctx: typer.Context) -> None:
    """Rename a released workspace model and related artifacts.

    Examples:
      smartrain model rename --release models/my_ds/2026-01-01_00-00-00_ultralytics_yolo11s_640px_100epochs_b16-abcd1234/detect_yolo11s_20260101_000000_640px_100epochs_b16.pt --new-name my_detector_v2
      smartrain model rename --release 1 --new-name my_detector_v2
      smartrain model rename
    """
    from smartrain.workflows.models.model_rename_cli import build_model_rename_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.models.model_rename_cli",
        build_parser=build_model_rename_arg_parser,
        prog="smartrain model rename",
        empty_args_mode="invoke_if_tty_else_help",
    )


def _model_group_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(HELP_MODEL_GROUP)
        typer.echo("Run: smartrain model convert -- --help")
        typer.echo("Run: smartrain model release -- --help")
        typer.echo("Run: smartrain model unrelease -- --help")
        typer.echo("Run: smartrain model comment -- --help")
        typer.echo("Run: smartrain model rename -- --help")
        raise typer.Exit(0)


app.add_typer(
    model_app,
    name="model",
    invoke_without_command=True,
    callback=_model_group_callback,
)


dataset_app = plain_sub_typer(
    help=HELP_DATASET_GROUP,
    invoke_without_command=True,
)


@dataset_app.command(
    "report",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_dataset_report(ctx: typer.Context) -> None:
    """Multilingual per-class sample report (Markdown + PNG; PDF/ODT via pandoc or extras).

    Examples:
      smartrain dataset report --dataset my_dataset
      smartrain dataset report --dataset my_dataset -n 6 --languages en,ru
      smartrain dataset report --workspace /data/MarsSmarTrain --dataset my_dataset --no-pdf
    """
    from smartrain.services.datasets.dataset_report import build_report_dataset_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.services.datasets.dataset_report",
        build_parser=build_report_dataset_arg_parser,
        prog="smartrain dataset report",
        empty_args_mode="invoke_if_tty_else_help",
        ensure_scan=True,
    )


@dataset_app.command(
    "convert",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_dataset_convert(ctx: typer.Context) -> None:
    """Convert datasets between supported formats (CVAT 1.1, YOLO, CvsDclDet).

    Examples:
      smartrain dataset convert
      smartrain dataset convert --source task.zip --to yolo --output-dir datasets/task_yolo
      smartrain dataset convert --source datasets/task_yolo --to cvat11 --output-dir converted_raw_data/task --zip
      smartrain dataset convert --source raw_data/my_det.zip --to cvat11 --output-dir converted_raw_data/my_det
    """
    from smartrain.workflows.datasets.dataset_convert_cli import build_dataset_convert_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_convert_cli",
        build_parser=build_dataset_convert_arg_parser,
        prog="smartrain dataset convert",
        empty_args_mode="invoke_if_tty_else_help",
    )


@dataset_app.command(
    "rename",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_dataset_rename(ctx: typer.Context) -> None:
    """Rename a workspace dataset and update related references.

    Examples:
      smartrain dataset rename --dataset old_name --new-name new_name
      smartrain dataset rename --dataset old_name --new-name new_name --dry-run
      smartrain dataset rename
    """
    from smartrain.workflows.datasets.dataset_rename_cli import build_dataset_rename_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_rename_cli",
        build_parser=build_dataset_rename_arg_parser,
        prog="smartrain dataset rename",
        empty_args_mode="invoke_if_tty_else_help",
    )


def _dataset_group_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(HELP_DATASET_GROUP)
        typer.echo("Run: smartrain dataset report -- --help")
        typer.echo("Run: smartrain dataset rename -- --help")
        typer.echo("Run: smartrain dataset convert -- --help")
        raise typer.Exit(0)


app.add_typer(
    dataset_app,
    name="dataset",
    invoke_without_command=True,
    callback=_dataset_group_callback,
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
    """Unified migration utilities.

    Examples:
      smartrain migrate unified --mode dry-run
      smartrain migrate unified --mode apply --continue-on-error
      smartrain migrate unified --source-kind run --report analytics/migration-reports/run-only.json
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
        ensure_scan=True,
    )


@app.command(
    "rotate",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_rotate(ctx: typer.Context) -> None:
    """Rotate a dataset by a fixed angle (90/180/270 degrees clockwise).

    Examples:
      smartrain rotate
      smartrain rotate --dataset my_dataset --angle 90
      smartrain rotate --dataset my_dataset --angle 270 --output-name my_dataset_rot270
      smartrain rotate --workspace /data/MarsSmarTrain --dataset my_dataset --angle 180
    """
    from smartrain.workflows.datasets.dataset_rotate import build_rotate_arg_parser

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.datasets.dataset_rotate",
        build_parser=build_rotate_arg_parser,
        prog="smartrain rotate",
        empty_args_mode="invoke_if_tty_else_help",
        ensure_scan=True,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
