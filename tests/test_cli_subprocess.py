"""Интеграция CLI: python -m smartrain."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from smartrain.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace

CLI_HELP_CASES: list[tuple[str, list[str]]] = [
    ("deploy", ["--help"]),
    ("datasets-json", ["--", "--help"]),
    ("dataset-former", ["--", "--help"]),
    ("train", ["--", "--help"]),
    ("hash", ["--", "--help"]),
    ("roi", ["--", "--help"]),
    ("cvat", ["--", "--help"]),
    ("queue", ["list"]),
    ("queue-run", ["--", "--help"]),
    ("registry", ["runs-list"]),
    ("analyze", ["scan", "--", "--help"]),
    ("plot", ["compare", "--", "--help"]),
]


def _run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "smartrain", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("cmd,extra", CLI_HELP_CASES)
def test_smartrain_subcommand_smoke(
    cmd: str,
    extra: list[str],
    tmp_path: Path,
    subprocess_env: dict[str, str],
) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run([cmd, *extra], cwd=tmp_path, env=env)
    assert r.returncode in (0, 2), f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    assert r.stdout or r.stderr


def test_smartrain_top_level_help(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    r = _run(["--help"], cwd=tmp_path, env=subprocess_env)
    assert r.returncode in (0, 2)
    out = (r.stdout or "") + (r.stderr or "")
    assert "deploy" in out


def test_smartrain_deploy_twice(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    env = dict(subprocess_env)
    r1 = _run(["deploy"], cwd=tmp_path, env=env)
    assert r1.returncode in (0, 2)
    o1 = (r1.stdout or "") + (r1.stderr or "")
    assert "source_datasets" in o1 or "+ каталог" in o1
    r2 = _run(["deploy"], cwd=tmp_path, env=env)
    assert r2.returncode in (0, 2)
    o2 = (r2.stdout or "") + (r2.stderr or "")
    assert "уже есть" in o2
