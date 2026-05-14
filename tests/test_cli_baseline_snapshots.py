from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from smartrain.cli_support.cli_replay import build_non_interactive_command
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace


def _run(args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "smartrain", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_baseline_help_snapshots(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())

    cases: list[tuple[list[str], list[str]]] = [
        (["--help"], ["deploy", "train", "analyze", "inference"]),
        (["train", "--", "--help"], ["Examples:", "--data", "--model"]),
        (["test", "--", "--help"], ["--formats", "--weights", "--non-interactive", "--nit"]),
        (["inference", "--", "--help"], ["--data-mode", "--source-dir", "--model-name"]),
        (["analyze", "scan", "--", "--help"], ["--workspace", "--models-root"]),
        (["balance", "--", "--help"], ["--dataset", "--strategy", "--preset"]),
        (["augment", "--", "--help"], ["--dataset", "--output-name", "--enable-flip"]),
    ]

    for cmd, expected_fragments in cases:
        proc = _run(cmd, tmp_path, env)
        out = (proc.stdout or "") + (proc.stderr or "")
        assert proc.returncode in (0, 2), out
        for fragment in expected_fragments:
            assert fragment in out, f"missing {fragment!r} in output for {' '.join(cmd)}"


def test_replay_builder_baseline_for_boolean_optional_flags() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--non-interactive", "-y", action="store_true")

    args = parser.parse_args(["--no-perf", "--name", "sample", "-y"])
    replay = build_non_interactive_command("inference", parser, args)

    assert "--no-perf" in replay
    assert "--perf False" not in replay
    assert "--name sample" in replay
    assert "-y" in replay or "--non-interactive" in replay
    assert replay.rstrip().endswith("--nit")

