"""Typer-level non-interactive flags stripped before argparse module.main."""

from __future__ import annotations

import os

# Stripped from argv before invoking argparse subcommands (never passed to parsers).
TYPER_META_NON_INTERACTIVE_FLAGS: frozenset[str] = frozenset({"--nit", "--smartrain-replay"})

# Env: force Typer non-interactive (CI / wrappers) even without flags on argv.
FORCE_NON_INTERACTIVE_ENV = "SMART_TRAIN_FORCE_NON_INTERACTIVE"


def env_forces_non_interactive_cli() -> bool:
    v = os.environ.get(FORCE_NON_INTERACTIVE_ENV, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def strip_typer_meta_non_interactive_flags(argv: list[str]) -> tuple[list[str], bool]:
    """Remove Typer-only tokens; return (filtered_argv, True if any token was removed)."""
    if not argv:
        return [], False
    stripped = False
    out: list[str] = []
    for tok in argv:
        if tok in TYPER_META_NON_INTERACTIVE_FLAGS:
            stripped = True
            continue
        out.append(tok)
    return out, stripped
