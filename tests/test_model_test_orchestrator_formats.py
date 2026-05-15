from __future__ import annotations

from smartrain.cli_support.cli_replay import build_non_interactive_command
from smartrain.workflows.testing.model_test_cli import build_model_test_arg_parser


def test_model_test_parser_nit_sets_non_interactive() -> None:
    parser = build_model_test_arg_parser()
    args = parser.parse_args(["--run", "/tmp/run", "--data", "/tmp/d.yaml", "--nit"])
    assert args.non_interactive is True


def test_replay_test_command_appends_nit_when_missing() -> None:
    """Replay contract replaces orchestrator H2: full CLI with --formats still gets Typer --nit suffix."""
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
    assert cmd.rstrip().endswith("--nit")
    assert cmd.count("--nit") == 1
    assert "--formats" in cmd


def test_replay_legacy_formats_only_namespace_still_gets_nit_suffix() -> None:
    """Old replay lines with --formats but no Typer meta flag are regenerated with trailing --nit."""
    parser = build_model_test_arg_parser()
    args = parser.parse_args(
        [
            "--run",
            "/tmp/run",
            "--data",
            "/tmp/ds/data.yaml",
            "--formats",
            "pt,onnx",
            "--device",
            "cpu",
        ]
    )
    cmd = build_non_interactive_command("test", parser, args)
    assert "--formats" in cmd
    assert cmd.count("--nit") == 1


def test_replay_test_command_single_nit_with_existing_non_interactive_flags() -> None:
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
    assert cmd.count("--nit") == 1
