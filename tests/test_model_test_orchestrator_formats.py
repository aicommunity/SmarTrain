from __future__ import annotations

from smartrain.cli_support.cli_replay import build_non_interactive_command
from smartrain.services.model_test_orchestrator import _formats_option_explicit_in_argv
from smartrain.workflows.testing.model_test_cli import build_model_test_arg_parser


def test_formats_explicit_in_argv_detects_long_and_equals_form() -> None:
    assert _formats_option_explicit_in_argv(["test", "--run", "x", "--formats", "pt,onnx"])
    assert _formats_option_explicit_in_argv(["--formats=pt"])
    assert not _formats_option_explicit_in_argv(["--run", "/tmp/r", "--data", "d.yaml"])
    assert not _formats_option_explicit_in_argv(None)
    assert not _formats_option_explicit_in_argv([])


def test_replay_test_command_appends_non_interactive_when_missing() -> None:
    parser = build_model_test_arg_parser()
    args = parser.parse_args(
        [
            "--run",
            "/tmp/run",
            "--data",
            "/tmp/ds/data.yaml",
            "--formats",
            "pt",
            "--device",
            "cpu",
        ]
    )
    cmd = build_non_interactive_command("test", parser, args)
    assert cmd.endswith("--non-interactive") or " --non-interactive" in cmd
    assert "--non-interactive" in cmd


def test_replay_test_command_does_not_duplicate_non_interactive() -> None:
    parser = build_model_test_arg_parser()
    args = parser.parse_args(
        [
            "--run",
            "/tmp/run",
            "--data",
            "/tmp/ds/data.yaml",
            "--formats",
            "pt",
            "-y",
            "--device",
            "cpu",
        ]
    )
    cmd = build_non_interactive_command("test", parser, args)
    assert cmd.count("--non-interactive") == 1
