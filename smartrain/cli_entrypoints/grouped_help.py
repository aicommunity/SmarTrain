"""Plain-text grouped help formatter for Typer/Click CLI."""

from __future__ import annotations

from typing import Any

import click
from typer.core import TyperGroup

from smartrain.cli_entrypoints.help_registry import (
    COMMAND_GROUPS,
    HELP_EPILOG,
    command_summary,
)


class GroupedTyperGroup(TyperGroup):
    """Typer group that renders commands in process-oriented sections."""

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        commands: dict[str, click.Command] = {
            name: cmd for name, cmd in self.commands.items() if not cmd.hidden
        }
        if not commands:
            return

        grouped_names: set[str] = set()
        for _title, names in COMMAND_GROUPS:
            rows: list[tuple[str, str]] = []
            for name in names:
                cmd = commands.get(name)
                if cmd is None:
                    continue
                grouped_names.add(name)
                summary = command_summary(name) or cmd.get_short_help_str(limit=120) or ""
                rows.append((name, summary))
            if not rows:
                continue
            formatter.write_paragraph()
            formatter.write_text(f"{_title}:")
            name_width = max(len(name) for name, _ in rows)
            for name, summary in rows:
                formatter.write_text(f"  {name.ljust(name_width)}  {summary}")

        remaining = sorted(set(commands.keys()) - grouped_names)
        if remaining:
            formatter.write_paragraph()
            formatter.write_text("Other:")
            rows = []
            for name in remaining:
                cmd = commands[name]
                summary = command_summary(name) or cmd.get_short_help_str(limit=120) or ""
                rows.append((name, summary))
            name_width = max(len(name) for name, _ in rows)
            for name, summary in rows:
                formatter.write_text(f"  {name.ljust(name_width)}  {summary}")

    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        epilog = self.epilog or HELP_EPILOG
        if epilog:
            formatter.write_paragraph()
            for line in epilog.splitlines():
                formatter.write_text(line)


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
        "rich_markup_mode": None,
        "pretty_exceptions_enable": False,
    }
    defaults.update(kwargs)
    return typer.Typer(**defaults)
