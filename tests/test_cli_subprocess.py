"""CLI integration: python -m smartrain."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from smartrain.cli import app
from smartrain.results_analyzer import build_analyze_arg_parser
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace

CLI_HELP_CASES: list[tuple[str, list[str]]] = [
    ("deploy", ["--help"]),
    ("scan", ["--", "--help"]),
    ("fusion", ["--", "--help"]),
    ("train", ["--", "--help"]),
    ("augment", ["--", "--help"]),
    ("balance", ["--", "--help"]),
    ("hash", ["--", "--help"]),
    ("stats", ["--", "--help"]),
    ("stats", ["compare", "--help"]),
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
    assert "raw_data" in o1 or "+ directory" in o1
    r2 = _run(["deploy"], cwd=tmp_path, env=env)
    assert r2.returncode in (0, 2)
    o2 = (r2.stdout or "") + (r2.stderr or "")
    assert "already exists" in o2


def test_analyze_typer_subcommands_match_argparse() -> None:
    analyze_group = next(group for group in app.registered_groups if group.name == "analyze")
    typer_subcommands = {cmd.name for cmd in analyze_group.typer_instance.registered_commands}

    parser = build_analyze_arg_parser()
    subparsers_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    argparse_subcommands = set(subparsers_action.choices.keys())

    assert typer_subcommands == argparse_subcommands


@pytest.mark.parametrize("group_name", ["queue", "registry", "analyze"])
def test_group_without_subcommand_shows_help_and_exits_zero(
    group_name: str,
    subprocess_env: dict[str, str],
    tmp_path: Path,
) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run([group_name], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    assert "Usage:" in out
