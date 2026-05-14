from __future__ import annotations

import os

import pytest

from smartrain.cli_support.typer_non_interactive import (
    FORCE_NON_INTERACTIVE_ENV,
    env_forces_non_interactive_cli,
    strip_typer_meta_non_interactive_flags,
)


def test_strip_removes_nit_and_smartrain_replay() -> None:
    raw = ["test", "--run", "x", "--nit", "--device", "cpu"]
    filtered, stripped = strip_typer_meta_non_interactive_flags(raw)
    assert filtered == ["test", "--run", "x", "--device", "cpu"]
    assert stripped is True


def test_strip_no_meta_returns_same_list() -> None:
    raw = ["train", "--data", "ds", "-y"]
    filtered, stripped = strip_typer_meta_non_interactive_flags(raw)
    assert filtered == raw
    assert stripped is False


def test_env_forces_non_interactive_cli_true_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FORCE_NON_INTERACTIVE_ENV, raising=False)
    assert env_forces_non_interactive_cli() is False
    monkeypatch.setenv(FORCE_NON_INTERACTIVE_ENV, "1")
    assert env_forces_non_interactive_cli() is True
    monkeypatch.setenv(FORCE_NON_INTERACTIVE_ENV, "no")
    assert env_forces_non_interactive_cli() is False
