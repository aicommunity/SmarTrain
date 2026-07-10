"""Tests for grouped CLI help registry and formatter."""

from __future__ import annotations

import subprocess
import sys

from smartrain.cli_entrypoints.help_registry import COMMAND_GROUPS, COMMANDS, command_summary


def test_command_registry_covers_all_top_level_groups() -> None:
    grouped = {name for _title, names in COMMAND_GROUPS for name in names}
    assert "train" in grouped
    assert "scan" in grouped
    assert "quickstart" in grouped
    assert grouped.issubset(set(COMMANDS.keys()))


def test_command_summary_returns_english_text() -> None:
    summary = command_summary("train")
    assert summary
    assert "Train" in summary or "train" in summary.lower()


def test_cli_import_is_fast() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import time; t=time.perf_counter(); import smartrain.cli; print(time.perf_counter()-t)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    elapsed = float(proc.stdout.strip())
    assert elapsed < 0.5, f"smartrain.cli import took {elapsed:.2f}s"
