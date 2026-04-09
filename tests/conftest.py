"""Common fixtures: repository root and environment for subprocess."""

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
    """PYTHONPATH for importing smartrain from sources. Do not touch HOME - otherwise the user-site (typer) disappears."""
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not prev else f"{repo_root}{os.pathsep}{prev}"
    return env


@pytest.fixture
def subprocess_env_isolated_home(repo_root: Path, tmp_path: Path) -> dict[str, str]:
    """For tests where Ultralytics does not need to write to the real ~/.config."""
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not prev else f"{repo_root}{os.pathsep}{prev}"
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(fake_home)
    env["XDG_CONFIG_HOME"] = str(fake_home / ".config")
    return env
