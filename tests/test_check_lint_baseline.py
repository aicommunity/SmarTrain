"""Unit tests for CI lint baseline count parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci" / "check_lint_baseline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_lint_baseline", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extract_ruff_counts() -> None:
    mod = _load_module()
    assert mod._extract_count("Found 283 errors.\n", tool="ruff") == 283
    assert mod._extract_count("Found 1 error.\n", tool="ruff") == 1
    assert mod._extract_count("All checks passed!\n", tool="ruff") == 0
    assert mod._extract_count("weird output", tool="ruff") is None


def test_extract_mypy_counts() -> None:
    mod = _load_module()
    assert mod._extract_count("Found 373 errors in 80 files (checked 372 source files)\n", tool="mypy") == 373
    assert mod._extract_count("Found 1 error in 1 file (checked 1 source file)\n", tool="mypy") == 1
    assert mod._extract_count("Success: no issues found in 12 source files\n", tool="mypy") == 0
    assert mod._extract_count("Found 2 errors in 1 file (checked 1 source file)\n", tool="mypy") == 2
