"""CLI integration: python -m smartrain."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from smartrain.cli import app
from smartrain.results_analyzer import build_analyze_arg_parser
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace

CLI_HELP_CASES: list[tuple[str, list[str]]] = [
    ("info", []),
    ("providers", ["status"]),
    ("providers", ["doctor"]),
    ("deploy", ["--help"]),
    ("scan", ["--", "--help"]),
    ("fusion", ["--", "--help"]),
    ("train", ["--", "--help"]),
    ("augment", ["--", "--help"]),
    ("balance", ["--", "--help"]),
    ("prune", ["empty", "--", "--help"]),
    ("prune", ["dedup", "--", "--help"]),
    ("hash", ["--", "--help"]),
    ("stats", ["--", "--help"]),
    ("stats", ["compare", "--help"]),
    ("roi", ["--", "--help"]),
    ("inference", ["--", "--help"]),
    ("cvat", ["--", "--help"]),
    ("queue", ["list"]),
    ("queue-run", ["--", "--help"]),
    ("registry", ["runs-list"]),
    ("analyze", ["scan", "--", "--help"]),
    ("plot", ["compare", "--", "--help"]),
]

NO_ARGS_HELP_CASES: list[str] = [
    "queue",
    "registry",
    "scan",
    "fusion",
    "augment",
    "balance",
    "prune",
    "hash",
    "stats",
    "roi",
    "queue-run",
    "plot",
    "migrate-models",
    "cvat",
    "clearml-upload",
    "sahi",
    "heatmap",
    "orient",
    "providers",
    "deps",
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


def test_smartrain_without_args_prints_quick_start(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    r = _run([], cwd=tmp_path, env=subprocess_env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    assert "Quick start" in out
    assert "smartrain report dataset" in out
    assert "usage:" not in out.lower()


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


@pytest.mark.parametrize("group_name", ["queue", "registry"])
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
    assert "usage:" in out.lower()


@pytest.mark.parametrize("cmd", NO_ARGS_HELP_CASES)
def test_command_without_args_shows_help(cmd: str, subprocess_env: dict[str, str], tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run([cmd], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    assert "usage:" in out.lower()


def test_analyze_no_subcommand_requires_tty(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(["analyze"], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 2, out
    assert "tty" in out.lower()


@pytest.mark.parametrize(
    "cmd,required_phrase",
    [
        (["analyze", "compare", "--help"], "usage:"),
        (["train", "--", "--help"], "Examples:"),
        (["cvat", "--", "--help"], "Examples:"),
        (["sahi", "--", "--help"], "Examples:"),
        (["heatmap", "--", "--help"], "Examples:"),
    ],
)
def test_key_help_commands_include_examples(
    cmd: list[str],
    required_phrase: str,
    subprocess_env: dict[str, str],
    tmp_path: Path,
) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(cmd, cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode in (0, 2), out
    assert required_phrase.lower() in out.lower()


def test_train_without_args_dispatches_to_interactive_flow(
    subprocess_env: dict[str, str],
    tmp_path: Path,
) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(["train"], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    # In non-TTY subprocesses interactive train reports this message;
    # this proves we call train main([]) instead of printing argparse help.
    assert "interactive train mode requires a terminal" in out.lower()


def test_fusion_missing_workspace_metadata_shows_friendly_error(
    subprocess_env: dict[str, str],
    tmp_path: Path,
) -> None:
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(["fusion", "--", "--workspace", str(tmp_path), "--dataset", "ds_a"], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    low = out.lower()
    assert r.returncode == 0, out
    assert "fusion metadata files were not found" in low
    assert "metadata directory" in low
    assert "datasets_info.json" in low
    assert "class_names.json" in low
    assert "traceback" not in low


def test_info_prints_supported_train_models(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(["info"], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    assert "Model source: ultralytics" in out
    assert "Supported train models:" in out
    assert "yolov8n" in out
    assert "-seg" not in out
    assert "-cls" not in out
    assert "-pose" not in out
    assert "-obb" not in out


def test_info_unknown_provider_returns_error(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(["info", "--provider", "unknown-provider"], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 2, out
    assert "Unknown training provider" in out


def test_info_lists_installed_external_providers(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    cfg = tmp_path / "cfg"
    idx = cfg / "smartrain" / "providers" / "index.json"
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text(
        """{
  "schema_version": 1,
  "updated_at": "2026-01-01T00:00:00+00:00",
  "providers": [
    {
      "provider_id": "dr-yolo",
      "display_name": "DR-YOLO",
      "repo_path": "/tmp/dr-yolo",
      "venv_path": "/tmp/dr-yolo/venv",
      "install_root": "/tmp",
      "install_state": "installed",
      "detected_capabilities": {"train": true, "infer": true},
      "repo_ref": {"remote_url": "https://example", "branch": "master", "commit": "abc"},
      "installed_at": "2026-01-01T00:00:00+00:00",
      "last_validated_at": "2026-01-01T00:00:00+00:00",
      "last_error": null
    }
  ]
}
""",
        encoding="utf-8",
    )
    env = dict(subprocess_env)
    env["XDG_CONFIG_HOME"] = str(cfg)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(["info"], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    assert "Installed external providers:" in out
    assert "dr-yolo" in out


def test_info_lists_external_provider_model_aliases(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    cfg = tmp_path / "cfg"
    idx = cfg / "smartrain" / "providers" / "index.json"
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text(
        """{
  "schema_version": 1,
  "updated_at": "2026-01-01T00:00:00+00:00",
  "providers": [
    {
      "provider_id": "dr-yolo",
      "display_name": "DR-YOLO",
      "repo_path": "/tmp/dr-yolo",
      "venv_path": "/tmp/dr-yolo/venv",
      "install_root": "/tmp",
      "install_state": "installed",
      "detected_capabilities": {"train": true, "infer": true},
      "repo_ref": {"remote_url": "https://example", "branch": "master", "commit": "abc"},
      "installed_at": "2026-01-01T00:00:00+00:00",
      "last_validated_at": "2026-01-01T00:00:00+00:00",
      "last_error": null
    }
  ]
}
""",
        encoding="utf-8",
    )
    env = dict(subprocess_env)
    env["XDG_CONFIG_HOME"] = str(cfg)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(["info"], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    assert "Supported train models (external providers):" in out
    assert "Model source: dr-yolo" in out
    assert "dr-yolo:" in out


def test_providers_doctor_reports_not_installed(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    cfg = tmp_path / "cfg"
    env = dict(subprocess_env)
    env["XDG_CONFIG_HOME"] = str(cfg)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(["providers", "doctor"], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert "Providers doctor" in out
    assert "not_installed" in out


def test_providers_doctor_verbose_includes_reason(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    cfg = tmp_path / "cfg"
    env = dict(subprocess_env)
    env["XDG_CONFIG_HOME"] = str(cfg)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(["providers", "doctor", "--verbose"], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert "Providers doctor" in out
    assert "reason: no record in global providers index" in out


def test_providers_install_enhanced_records_nested_repo_path(
    subprocess_env: dict[str, str],
    tmp_path: Path,
) -> None:
    pytest.skip("Temporarily disabled: flaky timeout due to heavy torch install in subprocess.")
    deploy_workspace(str(tmp_path))
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    fake_git = fake_bin / "git"
    fake_git.write_text(
        """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

args = sys.argv[1:]
if args[:1] == ["clone"]:
    dest = Path(args[-1])
    nested = dest / "yolov8-main-Ghost"
    nested.mkdir(parents=True, exist_ok=True)
    (dest / ".git").mkdir(parents=True, exist_ok=True)
    (nested / "train.py").write_text("print('train')\\n", encoding="utf-8")
    (nested / "detect.py").write_text("print('detect')\\n", encoding="utf-8")
    (nested / "requirements.txt").write_text("", encoding="utf-8")
    raise SystemExit(0)
if args[:2] == ["rev-parse", "HEAD"]:
    print("fakecommit")
    raise SystemExit(0)
raise SystemExit(0)
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)

    cfg = tmp_path / "cfg"
    env = dict(subprocess_env)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["XDG_CONFIG_HOME"] = str(cfg)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    env["SMART_TRAIN_SKIP_TORCH_POLICY"] = "1"
    install_target = tmp_path / "providers-root"
    r = _run(
        ["providers", "install", "--provider", "enhanced-yolov8", "--target", str(install_target), "-y"],
        cwd=tmp_path,
        env=env,
    )
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    assert "[INSTALLED] enhanced-yolov8" in out

    index_file = cfg / "smartrain" / "providers" / "index.json"
    payload = json.loads(index_file.read_text(encoding="utf-8"))
    rec = next(x for x in payload.get("providers", []) if x.get("provider_id") == "enhanced-yolov8")
    repo_path = Path(str(rec["repo_path"]))
    assert repo_path.name == "yolov8-main-Ghost"
    assert (repo_path / "train.py").is_file()
    assert (repo_path / "detect.py").is_file()


@pytest.mark.parametrize(
    "cmd,required_error,forbidden_phrase",
    [
        (["fusion", "--", "--workspace", "."], "incomplete arguments", "interactive"),
        (["augment", "--", "--workspace", "."], "incomplete arguments", "interactive augment mode"),
        (["balance", "--", "--workspace", "."], "incomplete arguments", "interactive balance mode"),
        (["orient", "--", "--workspace", "."], "incomplete arguments", "interactive"),
        (["roi", "--", "--workspace", "."], "incomplete arguments", "interactive roi mode"),
        (["inference", "--", "--workspace", ".", "--data-mode", "folder"], "incomplete arguments", "interactive inference mode"),
        (["stats", "compare", "--left", "foo"], "incomplete arguments", "interactive mode stats compare"),
    ],
)
def test_partial_args_do_not_trigger_interactive(
    cmd: list[str],
    required_error: str,
    forbidden_phrase: str,
    subprocess_env: dict[str, str],
    tmp_path: Path,
) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(cmd, cwd=tmp_path, env=env)
    out = ((r.stdout or "") + (r.stderr or "")).lower()
    assert required_error in out
    assert forbidden_phrase not in out
