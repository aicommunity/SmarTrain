from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

from smartrain.cli_replay import build_non_interactive_command, print_replay_command


@dataclass(slots=True)
class CliCommandRequest:
    command_name: str
    argv: list[str]
    interactive_allowed: bool
    interactive_used: bool = False
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CliCommandResponse:
    status: str
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)


def make_command_request(
    command_name: str,
    argv: list[str] | None,
    *,
    interactive_allowed: bool,
) -> CliCommandRequest:
    return CliCommandRequest(
        command_name=command_name,
        argv=list(argv or []),
        interactive_allowed=bool(interactive_allowed),
    )


def emit_replay(
    *,
    command_name: str,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    stage: str,
) -> str:
    replay_cmd = build_non_interactive_command(command_name, parser, args)
    print_replay_command(stage, replay_cmd)
    return replay_cmd

