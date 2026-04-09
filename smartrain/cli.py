#!/usr/bin/env python3
"""
Единая точка входа: команды из каталога workspace (SMART_TRAIN_WORKSPACE = cwd по умолчанию).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from smartrain.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace

app = typer.Typer(
    name="smartrain",
    add_completion=True,
    help="Датасеты YOLO, обучение, очередь, анализ прогонов. Работайте из корня workspace.",
)
console = Console()


def _sync_workspace_env(cli_workspace: Optional[str]) -> None:
    w = (cli_workspace or "").strip()
    if w:
        os.environ[WORKSPACE_ENV_VAR] = str(Path(w).resolve())
    elif not (os.environ.get(WORKSPACE_ENV_VAR) or "").strip():
        os.environ[WORKSPACE_ENV_VAR] = str(Path.cwd().resolve())


@app.callback()
def _main_callback(
    ctx: typer.Context,
    workspace: Annotated[
        Optional[str],
        typer.Option(
            "--workspace",
            envvar=WORKSPACE_ENV_VAR,
            help=f"Корень workspace (иначе {WORKSPACE_ENV_VAR}, иначе текущий каталог)",
        ),
    ] = None,
) -> None:
    if getattr(ctx, "resilient_parsing", False):
        return
    _sync_workspace_env(workspace)


@app.command("deploy")
def cmd_deploy(
    target: Annotated[
        Optional[str],
        typer.Argument(help="Куда развернуть (по умолчанию текущий каталог)"),
    ] = None,
) -> None:
    """Создать каталоги workspace и пустые datasets_info.json при отсутствии."""
    root = os.path.abspath(os.path.expanduser(target or os.getcwd()))
    info = deploy_workspace(root)
    console.print(f"[blue]Развёртывание:[/blue] {info['root']}")
    for name in info["created_dirs"]:
        console.print(f"[green]+ каталог[/green] {name}")
    for name in info["created_files"]:
        console.print(f"[green]+ файл[/green] {name}")
    for s in info["skipped"]:
        console.print(f"[yellow]∟ уже есть:[/yellow] {s}")
    console.print("[green]Готово.[/green]")


def _call(module: str, attr: str, ctx: typer.Context) -> None:
    import importlib

    m = importlib.import_module(module)
    fn = getattr(m, attr)
    # Не передавать None: иначе argparse прочитает sys.argv (команда smartrain, а не подкоманда).
    fn(list(ctx.args))


def _call_with_args(module: str, attr: str, args: list[str]) -> None:
    import importlib

    m = importlib.import_module(module)
    fn = getattr(m, attr)
    fn(args)


def _ctx_has_help_flag(ctx: typer.Context) -> bool:
    return any(tok in ("--help", "-h") for tok in ctx.args)


def _dispatch_argparse_help(
    ctx: typer.Context,
    build_parser,
    prog: str,
) -> None:
    """Отдать флаги подкоманды в argparse: полная справка и для подкоманд (queue list --help)."""
    p = build_parser()
    p.prog = prog
    try:
        p.parse_args(list(ctx.args))
    except SystemExit as e:
        code = e.code
        if code is None:
            code = 0
        raise typer.Exit(code if isinstance(code, int) else 1)


@app.command(
    "scan",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_scan(ctx: typer.Context) -> None:
    """Сканирование raw_data и подготовка datasets + JSON-индексов."""
    if _ctx_has_help_flag(ctx):
        from smartrain.datasets_json_former import build_datasets_json_arg_parser

        _dispatch_argparse_help(ctx, build_datasets_json_arg_parser, "smartrain scan")
    _call("smartrain.datasets_json_former", "main", ctx)


@app.command(
    "fusion",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_fusion(ctx: typer.Context) -> None:
    """Сборка объединённого датасета в datasets (с выбором входных датасетов)."""
    if _ctx_has_help_flag(ctx):
        from smartrain.dataset_former import build_dataset_former_arg_parser

        _dispatch_argparse_help(ctx, build_dataset_former_arg_parser, "smartrain fusion")
    _call("smartrain.dataset_former", "main", ctx)


@app.command(
    "train",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_train(ctx: typer.Context) -> None:
    """Обучение YOLO (бывш. model_training_module). Справка: smartrain train --help."""
    if _ctx_has_help_flag(ctx):
        from smartrain.model_training_module import build_train_arg_parser

        _dispatch_argparse_help(ctx, build_train_arg_parser, "smartrain train")
    _call("smartrain.model_training_module", "main", ctx)


@app.command(
    "augment",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_augment(ctx: typer.Context) -> None:
    """Офлайн-аугментация датасета в новый datasets/<name>."""
    if _ctx_has_help_flag(ctx):
        from smartrain.dataset_augment import build_augment_arg_parser

        _dispatch_argparse_help(ctx, build_augment_arg_parser, "smartrain augment")
    _call("smartrain.dataset_augment", "main", ctx)


@app.command(
    "balance",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_balance(ctx: typer.Context) -> None:
    """Балансировка датасета в новый datasets/<name>."""
    if _ctx_has_help_flag(ctx):
        from smartrain.dataset_balance import build_balance_arg_parser

        _dispatch_argparse_help(ctx, build_balance_arg_parser, "smartrain balance")
    _call("smartrain.dataset_balance", "main", ctx)


@app.command(
    "hash",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_hash(ctx: typer.Context) -> None:
    """Хеш датасета (бывш. dataset_hash)."""
    if _ctx_has_help_flag(ctx):
        from smartrain.dataset_hash import build_hash_arg_parser

        _dispatch_argparse_help(ctx, build_hash_arg_parser, "smartrain hash")
    _call("smartrain.dataset_hash", "main", ctx)


@app.command(
    "stats",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_stats(ctx: typer.Context) -> None:
    """Статистика датасетов в datasets/: classes и datasets."""
    if _ctx_has_help_flag(ctx):
        from smartrain.dataset_stats import build_stats_arg_parser, build_stats_compare_arg_parser

        if ctx.args and ctx.args[0] == "compare":
            _dispatch_argparse_help(ctx, build_stats_compare_arg_parser, "smartrain stats compare")
        else:
            _dispatch_argparse_help(ctx, build_stats_arg_parser, "smartrain stats")
    _call("smartrain.dataset_stats", "main", ctx)


@app.command(
    "roi",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_roi(ctx: typer.Context) -> None:
    """ROI-кроп датасета (бывш. dataset_roi_yolo)."""
    if _ctx_has_help_flag(ctx):
        from smartrain.dataset_roi_yolo import build_roi_arg_parser

        _dispatch_argparse_help(ctx, build_roi_arg_parser, "smartrain roi")
    _call("smartrain.dataset_roi_yolo", "main", ctx)


queue_app = typer.Typer(help="Очередь: list | add | remove | clear | run.")


@queue_app.command("list", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_queue_list(ctx: typer.Context) -> None:
    _call_with_args("smartrain.training_queue_cli", "main", ["list", *list(ctx.args)])


@queue_app.command("add", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_queue_add(ctx: typer.Context) -> None:
    _call_with_args("smartrain.training_queue_cli", "main", ["add", *list(ctx.args)])


@queue_app.command("remove", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_queue_remove(ctx: typer.Context) -> None:
    _call_with_args("smartrain.training_queue_cli", "main", ["remove", *list(ctx.args)])


@queue_app.command("clear", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_queue_clear(ctx: typer.Context) -> None:
    _call_with_args("smartrain.training_queue_cli", "main", ["clear", *list(ctx.args)])


@queue_app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_queue_run_sub(ctx: typer.Context) -> None:
    _call_with_args("smartrain.training_queue_cli", "main", ["run", *list(ctx.args)])


app.add_typer(queue_app, name="queue")


@app.command(
    "queue-run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_queue_run(ctx: typer.Context) -> None:
    """Исполнитель очереди (бывш. training_queue)."""
    if _ctx_has_help_flag(ctx):
        from smartrain.training_queue import build_queue_run_arg_parser

        _dispatch_argparse_help(ctx, build_queue_run_arg_parser, "smartrain queue-run")
    _call("smartrain.training_queue", "main", ctx)


registry_app = typer.Typer(help="Реестр runs / models.")


@registry_app.command("runs-list", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_registry_runs_list(ctx: typer.Context) -> None:
    _call_with_args("smartrain.registry_cli", "main", ["runs-list", *list(ctx.args)])


@registry_app.command("runs-info", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_registry_runs_info(ctx: typer.Context) -> None:
    _call_with_args("smartrain.registry_cli", "main", ["runs-info", *list(ctx.args)])


@registry_app.command("runs-metrics", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_registry_runs_metrics(ctx: typer.Context) -> None:
    _call_with_args("smartrain.registry_cli", "main", ["runs-metrics", *list(ctx.args)])


@registry_app.command("models-add", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_registry_models_add(ctx: typer.Context) -> None:
    _call_with_args("smartrain.registry_cli", "main", ["models-add", *list(ctx.args)])


@registry_app.command("models-list", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_registry_models_list(ctx: typer.Context) -> None:
    _call_with_args("smartrain.registry_cli", "main", ["models-list", *list(ctx.args)])


@registry_app.command("models-info", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_registry_models_info(ctx: typer.Context) -> None:
    _call_with_args("smartrain.registry_cli", "main", ["models-info", *list(ctx.args)])


@registry_app.command("models-remove", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_registry_models_remove(ctx: typer.Context) -> None:
    _call_with_args("smartrain.registry_cli", "main", ["models-remove", *list(ctx.args)])


app.add_typer(registry_app, name="registry")


analyze_app = typer.Typer(help="Анализ прогонов: scan, export-table, compare, interactive.")


@analyze_app.command("scan", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_analyze_scan(ctx: typer.Context) -> None:
    _call_with_args("smartrain.results_analyzer", "main", ["scan", *list(ctx.args)])


@analyze_app.command("export-table", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_analyze_export_table(ctx: typer.Context) -> None:
    _call_with_args("smartrain.results_analyzer", "main", ["export-table", *list(ctx.args)])


@analyze_app.command("compare", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_analyze_compare(ctx: typer.Context) -> None:
    _call_with_args("smartrain.results_analyzer", "main", ["compare", *list(ctx.args)])


@analyze_app.command("interactive", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cmd_analyze_interactive(ctx: typer.Context) -> None:
    _call_with_args("smartrain.results_analyzer", "main", ["interactive", *list(ctx.args)])


app.add_typer(analyze_app, name="analyze")


@app.command(
    "plot",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_plot(ctx: typer.Context) -> None:
    """Устаревшая обёртка → analyze."""
    if _ctx_has_help_flag(ctx):
        from smartrain.results_analyzer import build_analyze_arg_parser

        _dispatch_argparse_help(ctx, build_analyze_arg_parser, "smartrain plot")
    _call("smartrain.plot_creator", "main", ctx)


@app.command(
    "migrate-models",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_migrate_models(ctx: typer.Context) -> None:
    """Миграция legacy моделей: добавление training_metadata.json для analyze."""
    if _ctx_has_help_flag(ctx):
        from smartrain.migrate_models_to_smartrain import build_migrate_models_arg_parser

        _dispatch_argparse_help(ctx, build_migrate_models_arg_parser, "smartrain migrate-models")
    _call("smartrain.migrate_models_to_smartrain", "main", ctx)


@app.command(
    "cvat",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_cvat(ctx: typer.Context) -> None:
    """Конвертация CVAT 1.1 (Images+bbox): import/export."""
    if _ctx_has_help_flag(ctx):
        from smartrain.cvat_cli import build_cvat_arg_parser

        _dispatch_argparse_help(ctx, build_cvat_arg_parser, "smartrain cvat")
    _call("smartrain.cvat_cli", "main", ctx)


@app.command(
    "clearml-upload",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_clearml_upload(ctx: typer.Context) -> None:
    """Загрузка готового прогона в ClearML (extras: pip install 'smartrain[clearml]')."""
    if _ctx_has_help_flag(ctx):
        from smartrain.clearml_upload import build_clearml_upload_arg_parser

        _dispatch_argparse_help(ctx, build_clearml_upload_arg_parser, "smartrain clearml-upload")
    _call("smartrain.clearml_upload", "main", ctx)


@app.command(
    "sahi",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_sahi(ctx: typer.Context) -> None:
    """Тайловый инференс SAHI (extras: pip install 'smartrain[sahi]')."""
    if _ctx_has_help_flag(ctx):
        from smartrain.sahi_cli import build_sahi_arg_parser

        _dispatch_argparse_help(ctx, build_sahi_arg_parser, "smartrain sahi")
    _call("smartrain.sahi_cli", "main", ctx)


@app.command(
    "heatmap",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_heatmap(ctx: typer.Context) -> None:
    """Визуализация heatmap (Ultralytics solutions)."""
    if _ctx_has_help_flag(ctx):
        from smartrain.heatmap_cli import build_heatmap_arg_parser

        _dispatch_argparse_help(ctx, build_heatmap_arg_parser, "smartrain heatmap")
    _call("smartrain.heatmap_cli", "main", ctx)


@app.command(
    "orient",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def cmd_orient(ctx: typer.Context) -> None:
    """Исправление поворотов 0/90/180/270 по эталонам (в новый датасет)."""
    if _ctx_has_help_flag(ctx):
        from smartrain.dataset_orient import build_orient_arg_parser

        _dispatch_argparse_help(ctx, build_orient_arg_parser, "smartrain orient")
    _call("smartrain.dataset_orient", "main", ctx)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
