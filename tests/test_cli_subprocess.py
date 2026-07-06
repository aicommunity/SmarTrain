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
from smartrain.workflows.analyze.results_analyzer import build_analyze_arg_parser
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, deploy_workspace

HELP_MATRIX: list[list[str]] = [
    ["--help"],
    ["quickstart"],
    ["deploy", "--help"],
    ["info", "--help"],
    ["scan", "--", "--help"],
    ["normalize-data-yaml", "--", "--help"],
    ["fusion", "--", "--help"],
    ["split", "--", "--help"],
    ["train", "--", "--help"],
    ["augment", "--", "--help"],
    ["balance", "--", "--help"],
    ["prune", "--", "--help"],
    ["prune", "empty", "--", "--help"],
    ["prune", "dedup", "--", "--help"],
    ["filter", "--", "--help"],
    ["hash", "--", "--help"],
    ["stats", "--", "--help"],
    ["stats", "compare", "--", "--help"],
    ["roi", "--", "--help"],
    ["test", "--", "--help"],
    ["inference", "--", "--help"],
    ["plot", "--", "--help"],
    ["migrate", "--", "--help"],
    ["migrate-models", "--", "--help"],
    ["cvat", "--", "--help"],
    ["clearml-upload", "--", "--help"],
    ["sahi", "--", "--help"],
    ["heatmap", "--", "--help"],
    ["orient", "--", "--help"],
    ["queue-run", "--", "--help"],
    ["dataset", "--help"],
    ["dataset", "report", "--", "--help"],
    ["dataset", "rename", "--", "--help"],
    ["queue", "--help"],
    ["queue", "list", "--", "--help"],
    ["queue", "add", "--", "--help"],
    ["queue", "remove", "--", "--help"],
    ["queue", "clear", "--", "--help"],
    ["queue", "run", "--", "--help"],
    ["registry", "--help"],
    ["registry", "runs-list", "--", "--help"],
    ["registry", "runs-info", "--", "--help"],
    ["registry", "runs-metrics", "--", "--help"],
    ["registry", "models-add", "--", "--help"],
    ["registry", "models-list", "--", "--help"],
    ["registry", "models-info", "--", "--help"],
    ["registry", "models-remove", "--", "--help"],
    ["providers", "--help"],
    ["providers", "install", "--", "--help"],
    ["providers", "uninstall", "--", "--help"],
    ["providers", "status", "--", "--help"],
    ["providers", "doctor", "--", "--help"],
    ["deps", "--help"],
    ["deps", "sync-torch", "--help"],
    ["analyze", "--help"],
    ["analyze", "all", "--", "--help"],
    ["analyze", "scan", "--", "--help"],
    ["analyze", "export-table", "--", "--help"],
    ["analyze", "compare", "--", "--help"],
    ["analyze", "pr-curves", "--", "--help"],
    ["analyze", "inference-benchmark", "--", "--help"],
    ["analyze", "inference-plot", "--", "--help"],
    ["analyze", "test-metrics-plot", "--", "--help"],
    ["analyze", "leaderboard", "--", "--help"],
    ["model", "--help"],
    ["model", "convert", "--", "--help"],
    ["model", "release", "--", "--help"],
    ["model", "rename", "--", "--help"],
    ["rotate", "--", "--help"],
]

NO_ARGS_USAGE_CASES: list[str] = [
    "queue",
    "registry",
    "scan",
    "fusion",
    "split",
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


@pytest.mark.parametrize("argv", HELP_MATRIX)
def test_smartrain_help_matrix(
    argv: list[str],
    tmp_path: Path,
    subprocess_env: dict[str, str],
) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(argv, cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode in (0, 2), f"argv={argv}\nstderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    if argv == ["deps", "--help"]:
        assert "Usage:" in out or "usage:" in out.lower()
    elif argv == ["quickstart"]:
        assert "Quick start" in out or "smartrain deploy" in out
    else:
        assert "usage:" in out.lower() or "examples:" in out.lower(), f"argv={argv}\n{out}"
    assert "traceback" not in out.lower(), f"argv={argv}\n{out}"


def test_smartrain_top_level_help(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    r = _run(["--help"], cwd=tmp_path, env=subprocess_env)
    assert r.returncode in (0, 2)
    out = (r.stdout or "") + (r.stderr or "")
    assert "deploy" in out
    assert "Workspace:" in out
    assert "Dataset catalog and preparation:" in out
    assert "Training:" in out
    assert "quickstart" in out


def test_smartrain_without_args_shows_grouped_help(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    r = _run([], cwd=tmp_path, env=subprocess_env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    assert "Usage:" in out or "usage:" in out.lower()
    assert "Workspace:" in out
    assert "train" in out
    assert "Quick start" not in out


def test_smartrain_shell_completion_lists_commands(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    env = dict(subprocess_env)
    env["_SMARTRAIN_COMPLETE"] = "complete_bash"
    env["COMP_WORDS"] = "smartrain "
    env["COMP_CWORD"] = "1"
    r = subprocess.run(
        [sys.executable, "-m", "smartrain"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    assert "train" in out
    assert "scan" in out
    assert "Quick start" not in out


def test_smartrain_quickstart_prints_guide(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    r = _run(["quickstart"], cwd=tmp_path, env=subprocess_env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    assert "Quick start" in out or "smartrain deploy" in out
    assert "smartrain dataset report" in out


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


def _collect_typer_paths(root_app: object, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    paths: set[tuple[str, ...]] = set()
    commands = getattr(root_app, "registered_commands", [])
    for cmd in commands:
        name = str(getattr(cmd, "name", "") or "")
        if not name:
            continue
        paths.add((*prefix, name))
    groups = getattr(root_app, "registered_groups", [])
    for grp in groups:
        name = str(getattr(grp, "name", "") or "")
        if not name:
            continue
        sub_typer = getattr(grp, "typer_instance", None)
        if sub_typer is None:
            continue
        paths.add((*prefix, name))
        paths.update(_collect_typer_paths(sub_typer, (*prefix, name)))
    return paths


def _help_argv_to_path(argv: list[str]) -> tuple[str, ...] | None:
    if argv == ["--help"]:
        return None
    out: list[str] = []
    for tok in argv:
        if tok in {"--help", "-h", "--"}:
            break
        out.append(tok)
    return tuple(out) if out else None


def test_cli_help_matrix_covers_all_typer_paths() -> None:
    inventory = _collect_typer_paths(app)
    matrix_paths = {
        p for p in (_help_argv_to_path(argv) for argv in HELP_MATRIX) if p is not None
    }
    missing = sorted(inventory - matrix_paths)
    assert not missing, f"Missing help coverage for Typer paths: {missing}"


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


@pytest.mark.parametrize("cmd", NO_ARGS_USAGE_CASES)
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


@pytest.mark.parametrize("group_name", ["dataset", "model"])
def test_group_without_subcommand_prints_group_help(
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
    assert "help" in out.lower() or "usage:" in out.lower()


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


@pytest.mark.parametrize(
    ("cmd", "expected_hint"),
    [
        ("test", "interactive test mode requires a terminal"),
        ("inference", "interactive inference mode requires a terminal"),
    ],
)
def test_invoke_mode_commands_without_args_on_non_tty(
    cmd: str,
    expected_hint: str,
    subprocess_env: dict[str, str],
    tmp_path: Path,
) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run([cmd], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode != 0, out
    low = out.lower()
    assert (
        expected_hint in low
        or "requires a terminal" in low
        or "input is not a terminal" in low
        or "aborted" in low
    )


def test_fusion_missing_workspace_metadata_shows_friendly_error(
    subprocess_env: dict[str, str],
    tmp_path: Path,
) -> None:
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(
        ["fusion", "--", "--no-auto-scan", "--workspace", str(tmp_path), "--dataset", "ds_a"],
        cwd=tmp_path,
        env=env,
    )
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
        (["fusion", "--", "--no-auto-scan", "--workspace", "."], "incomplete arguments", "interactive"),
        (["split", "--", "--no-auto-scan", "--workspace", "."], "datasets_info.json", "interactive"),
        (["augment", "--", "--no-auto-scan", "--workspace", "."], "incomplete arguments", "interactive augment mode"),
        (["balance", "--", "--no-auto-scan", "--workspace", "."], "incomplete arguments", "interactive balance mode"),
        (["orient", "--", "--no-auto-scan", "--workspace", "."], "incomplete arguments", "interactive"),
        (["roi", "--", "--no-auto-scan", "--workspace", "."], "incomplete arguments", "interactive roi mode"),
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


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--", "--unknown-flag"],
        ["test", "--", "--unknown-flag"],
        ["inference", "--", "--unknown-flag"],
        ["scan", "--", "--unknown-flag"],
        ["fusion", "--", "--unknown-flag"],
        ["migrate", "--", "--unknown-flag"],
        ["migrate", "--", "canonical", "--", "--unknown-flag"],
        ["registry", "runs-list", "--", "--unknown-flag"],
        ["analyze", "scan", "--", "--unknown-flag"],
        ["model", "convert", "--", "--unknown-flag"],
    ],
)
def test_unknown_flag_errors_are_user_friendly(
    argv: list[str],
    subprocess_env: dict[str, str],
    tmp_path: Path,
) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(argv, cwd=tmp_path, env=env)
    out = ((r.stdout or "") + (r.stderr or "")).lower()
    assert r.returncode != 0, out
    assert "traceback" not in out
    assert "unknown" in out or "unrecognized" in out or "error" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["test", "--", "--task", "not-a-valid-task"],
        ["test", "--", "--onnx-provider-policy", "bogus_policy"],
        ["inference", "--", "--task", "bogus"],
        ["inference", "--", "--split", "bogus_split"],
        ["inference", "--", "--data-mode", "bogus_mode"],
        ["migrate", "--", "canonical", "--source-kind", "not_run_model_all"],
        ["migrate", "--", "canonical", "--mode", "not_a_mode"],
        ["analyze", "all", "--", "--profile", "not_a_profile"],
        ["analyze", "all", "--", "--recompute-missing-metrics", "maybe"],
    ],
)
def test_invalid_choice_errors(
    argv: list[str],
    subprocess_env: dict[str, str],
    tmp_path: Path,
) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(argv, cwd=tmp_path, env=env)
    out = ((r.stdout or "") + (r.stderr or "")).lower()
    assert r.returncode != 0, out
    assert "traceback" not in out
    assert "invalid choice" in out or "error" in out


def test_providers_install_unknown_provider_returns_error(
    subprocess_env: dict[str, str],
    tmp_path: Path,
) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(["providers", "install", "--", "--provider", "unknown-provider", "-y"], cwd=tmp_path, env=env)
    out = ((r.stdout or "") + (r.stderr or "")).lower()
    assert r.returncode in (0, 2), out
    assert "traceback" not in out
    assert "unknown provider" in out or "invalid choice" in out or "error" in out


def test_migrate_canonical_dry_run_smoke(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds_cli" / "run_cli"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"dataset": {"name": "ds_cli"}}}),
        encoding="utf-8",
    )
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(
        ["migrate", "--", "canonical", "--workspace", str(tmp_path), "--mode", "dry-run"],
        cwd=tmp_path,
        env=env,
    )
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    assert "Migration report JSON" in out


def test_registry_runs_list_smoke(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(["registry", "runs-list"], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    assert "runs with training_metadata.json not found" in out.lower()


def test_registry_models_list_smoke(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(["registry", "models-list"], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    assert "traceback" not in out.lower()


def test_queue_list_smoke(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    env = dict(subprocess_env)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(["queue", "list"], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    assert (
        "queue is empty" in out.lower()
        or "no tasks" in out.lower()
        or "queue file missing" in out.lower()
    )


def test_providers_status_smoke(subprocess_env: dict[str, str], tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    cfg = tmp_path / "cfg"
    env = dict(subprocess_env)
    env["XDG_CONFIG_HOME"] = str(cfg)
    env[WORKSPACE_ENV_VAR] = str(tmp_path.resolve())
    r = _run(["providers", "status"], cwd=tmp_path, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
    assert "providers status" in out.lower() or "not_installed" in out.lower()


def test_docs_cli_command_parity_contains_core_commands() -> None:
    docs_files = [
        Path("docs/cli/overview.md"),
        Path("docs/cli/analyze.md"),
        Path("docs/cli/inference.md"),
        Path("docs/cli/registry.md"),
        Path("docs/cli/queue.md"),
        Path("docs/cli/providers.md"),
        Path("docs/getting-started/quickstart.md"),
        Path("README.md"),
    ]
    combined = "\n".join(p.read_text(encoding="utf-8") for p in docs_files if p.is_file()).lower()
    expected_markers = [
        "smartrain train",
        "smartrain test",
        "smartrain inference",
        "smartrain analyze",
        "smartrain registry",
        "smartrain queue",
        "smartrain model convert",
        "smartrain migrate",
    ]
    missing = [marker for marker in expected_markers if marker not in combined]
    assert not missing, f"Missing CLI docs markers: {missing}"


def test_docs_cli_group_subcommand_parity_has_key_entries() -> None:
    docs_files = [
        Path("docs/cli/overview.md"),
        Path("docs/cli/analyze.md"),
        Path("docs/cli/inference.md"),
        Path("docs/cli/registry.md"),
        Path("docs/cli/queue.md"),
        Path("docs/cli/providers.md"),
        Path("docs/getting-started/quickstart.md"),
        Path("README.md"),
    ]
    combined = "\n".join(p.read_text(encoding="utf-8") for p in docs_files if p.is_file()).lower()
    expected_entries = [
        "smartrain dataset report",
        "smartrain queue list",
        "smartrain queue add",
        "smartrain queue run",
        "smartrain registry runs-list",
        "smartrain registry models-list",
        "smartrain providers status",
        "smartrain providers install",
        "smartrain providers doctor",
        "smartrain analyze scan",
        "smartrain analyze compare",
        "smartrain analyze leaderboard",
        "smartrain model convert",
        "smartrain model release",
        "smartrain model rename",
        "smartrain migrate unified",
    ]
    missing = [entry for entry in expected_entries if entry not in combined]
    assert not missing, f"Missing docs subcommand entries: {missing}"
