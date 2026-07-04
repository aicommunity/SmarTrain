"""Plain-text grouped help formatter for Typer/Click CLI."""

from __future__ import annotations

import shutil
from typing import Any

import click
from typer.core import TyperGroup

from smartrain.cli_entrypoints.help_registry import (
    COMMAND_GROUPS,
    HELP_EPILOG,
    command_summary,
)


def _terminal_help_width(fallback: int = 120) -> int:
    return max(shutil.get_terminal_size(fallback=(fallback, 24)).columns, 80)


class WideHelpTyperGroup(TyperGroup):
    """Typer group that uses the full terminal width for help output."""

    def make_formatter(self) -> click.HelpFormatter:
        width = _terminal_help_width()
        return click.HelpFormatter(width=width, max_width=width)

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        opts: list[tuple[str, str]] = []
        for param in self.get_params(ctx):
            if param.param_type_name != "option":
                continue
            record = param.get_help_record(ctx)
            if record is not None:
                opts.append(record)
        if opts:
            formatter.write_paragraph()
            formatter.write("Options:\n")
            from click.formatting import term_len

            name_width = max(term_len(name) for name, _ in opts)
            for name, help_text in opts:
                formatter.write(f"  {name.ljust(name_width)}  {help_text}\n")
        self.format_commands(ctx, formatter)


class GroupedTyperGroup(WideHelpTyperGroup):
    """Typer group that renders commands in process-oriented sections."""

    def _write_command_line(
        self,
        formatter: click.HelpFormatter,
        name: str,
        summary: str,
        name_width: int,
    ) -> None:
        formatter.write(f"  {name.ljust(name_width)}  {summary}\n")

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        commands: dict[str, click.Command] = {
            name: cmd for name, cmd in self.commands.items() if not cmd.hidden
        }
        if not commands:
            return

        grouped_names: set[str] = set()
        for title, names in COMMAND_GROUPS:
            rows: list[tuple[str, str]] = []
            for name in names:
                cmd = commands.get(name)
                if cmd is None:
                    continue
                grouped_names.add(name)
                summary = command_summary(name) or cmd.get_short_help_str(limit=200) or ""
                rows.append((name, summary))
            if not rows:
                continue
            formatter.write_paragraph()
            formatter.write(f"{title}:\n")
            name_width = max(len(name) for name, _ in rows)
            for name, summary in rows:
                self._write_command_line(formatter, name, summary, name_width)

        remaining = sorted(set(commands.keys()) - grouped_names)
        if remaining:
            formatter.write_paragraph()
            formatter.write("Other:\n")
            rows = []
            for name in remaining:
                cmd = commands[name]
                summary = command_summary(name) or cmd.get_short_help_str(limit=200) or ""
                rows.append((name, summary))
            name_width = max(len(name) for name, _ in rows)
            for name, summary in rows:
                self._write_command_line(formatter, name, summary, name_width)

    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        epilog = self.epilog or HELP_EPILOG
        if epilog:
            formatter.write_paragraph()
            for line in epilog.splitlines():
                formatter.write(f"{line}\n")


def plain_typer(**kwargs: Any) -> Any:
    """Create a Typer app with grouped plain-text help (no Rich panels)."""
    import typer

    defaults: dict[str, Any] = {
        "cls": GroupedTyperGroup,
        "rich_markup_mode": None,
        "pretty_exceptions_enable": False,
    }
    defaults.update(kwargs)
    return typer.Typer(**defaults)


def plain_sub_typer(**kwargs: Any) -> Any:
    """Plain-text Typer sub-app (standard command list, no Rich)."""
    import typer

    defaults: dict[str, Any] = {
        "cls": WideHelpTyperGroup,
        "rich_markup_mode": None,
        "pretty_exceptions_enable": False,
    }
    defaults.update(kwargs)
    return typer.Typer(**defaults)
