from __future__ import annotations

import argparse
import shlex
from typing import Any


def _pick_option(action: argparse.Action) -> str | None:
    if not action.option_strings:
        return None
    long_opts = [o for o in action.option_strings if o.startswith("--")]
    return long_opts[0] if long_opts else action.option_strings[0]


def build_non_interactive_command(
    command_name: str,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> str:
    parts: list[str] = ["smartrain", *command_name.split()]
    for action in parser._actions:
        if action.dest in ("help",):
            continue
        if action.dest == argparse.SUPPRESS:
            continue
        dest = str(action.dest)
        if not hasattr(args, dest):
            continue
        value: Any = getattr(args, dest)
        opt = _pick_option(action)

        if isinstance(action, argparse._StoreTrueAction):
            if bool(value):
                parts.append(opt or "")
            continue
        if isinstance(action, argparse._StoreFalseAction):
            if value is False:
                parts.append(opt or "")
            continue
        if isinstance(action, argparse.BooleanOptionalAction):
            # BooleanOptionalAction exposes both positive/negative options.
            # Emit only the explicit flag form, never " --flag True/False ".
            if value is True:
                if action.option_strings:
                    positive = next((o for o in action.option_strings if o.startswith("--no-") is False), None)
                    parts.append(positive or (opt or ""))
            elif value is False:
                if action.option_strings:
                    negative = next((o for o in action.option_strings if o.startswith("--no-")), None)
                    parts.append(negative or (opt or ""))
            continue
        if value is None:
            continue
        if isinstance(action, argparse._AppendAction):
            if not value:
                continue
            for item in value:
                if opt:
                    parts.extend([opt, str(item)])
                else:
                    parts.append(str(item))
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            if opt:
                for item in value:
                    parts.extend([opt, str(item)])
            else:
                parts.extend([str(x) for x in value])
            continue

        if opt:
            parts.extend([opt, str(value)])
        else:
            parts.append(str(value))

    safe_parts = [shlex.quote(p) for p in parts if p]
    return " ".join(safe_parts)


def print_replay_command(stage: str, command: str) -> None:
    _ = stage
    print("[INFO] Command for non-interactive retry:")
    print(command)
