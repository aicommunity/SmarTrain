"""Общие фикстуры: корень репозитория и окружение для subprocess."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def subprocess_env(repo_root: Path) -> dict[str, str]:
    """PYTHONPATH для импорта smartrain из исходников. HOME не трогаем — иначе пропадает user-site (typer)."""
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not prev else f"{repo_root}{os.pathsep}{prev}"
    return env


@pytest.fixture
def subprocess_env_isolated_home(repo_root: Path, tmp_path: Path) -> dict[str, str]:
    """Для тестов, где Ultralytics не должна писать в реальный ~/.config."""
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not prev else f"{repo_root}{os.pathsep}{prev}"
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(fake_home)
    env["XDG_CONFIG_HOME"] = str(fake_home / ".config")
    return env
