"""Shared interactive CLI preamble helpers."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from smartrain.cli_entrypoints.support.cli_prompts import is_interactive_tty, prompt_choice


def should_run_interactive(
    args: argparse.Namespace,
    required_attrs: Sequence[str],
    *,
    force_non_interactive: bool = False,
) -> bool:
    """Return True when missing required args and an interactive TTY is available."""
    if force_non_interactive:
        return False
    if getattr(args, "nit", False) or getattr(args, "non_interactive", False):
        return False
    if not is_interactive_tty():
        return False
    for name in required_attrs:
        val = getattr(args, name, None)
        if val is None:
            return True
        if isinstance(val, str) and not val.strip():
            return True
        if isinstance(val, (list, tuple)) and len(val) == 0:
            return True
    return False


def prompt_dataset_choice(
    dataset_names: Sequence[str],
    *,
    label: str = "Dataset",
    default: str | None = None,
) -> str:
    """Prompt for a dataset name from a non-empty list."""
    names = [str(n) for n in dataset_names]
    if not names:
        raise ValueError("No datasets available for interactive choice")
    default_name = default if default in names else names[0]
    return str(prompt_choice(label, names, default=default_name))


def require_or_exit(message: str, *, code: int = 2) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(code)


def ensure_dataset_arg(
    args: argparse.Namespace,
    dataset_names: Sequence[str],
    *,
    attr: str = "dataset",
    label: str = "Dataset",
) -> None:
    """Fill ``args.<attr>`` interactively when empty and datasets exist."""
    current = getattr(args, attr, None)
    if isinstance(current, str) and current.strip():
        return
    if not dataset_names:
        require_or_exit("No datasets found in catalog")
    setattr(args, attr, prompt_dataset_choice(dataset_names, label=label))
