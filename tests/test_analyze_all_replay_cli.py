"""Replay string for `smartrain analyze all` uses shared cli_replay helpers."""

from __future__ import annotations

import argparse
import shlex

from smartrain.cli_entrypoints.support.typer_non_interactive import TYPER_META_NON_INTERACTIVE_FLAGS
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command
from smartrain.workflows.analyze.results_analyzer import build_analyze_arg_parser


def _p_all_subparser(root: argparse.ArgumentParser) -> argparse.ArgumentParser:
    for a in root._actions:
        if isinstance(a, argparse._SubParsersAction):
            return a.choices["all"]
    raise AssertionError("missing 'all' subparser")


def test_analyze_all_replay_includes_baseline_others_and_recompute_choice() -> None:
    root = build_analyze_arg_parser()
    p_all = _p_all_subparser(root)
    ns = root.parse_args(
        [
            "all",
            "--baseline",
            "/workspace/runs/ds/2026-01-01",
            "--others",
            "/workspace/runs/ds/2026-01-02",
            "/workspace/runs/ds/2026-01-03",
            "--profile",
            "full",
            "--report-languages",
            "en,ru",
            "--recompute-missing-metrics",
            "yes",
            "--scatter-x",
            "avg_inference_ms_per_frame",
            "--scatter-y",
            "mAP50-95",
            "--val-batch",
            "2",
            "--val-imgsz",
            "512",
            "--no-val-half",
            "--allow-cpu-fallback",
            "--data-yaml",
            "/workspace/datasets/ds/data.yaml",
        ]
    )
    cmd = build_non_interactive_command("analyze all", p_all, ns)
    assert cmd.rstrip().endswith("--nit")
    assert "analyze" in cmd
    assert "all" in cmd
    assert "/workspace/runs/ds/2026-01-01" in cmd
    assert "/workspace/runs/ds/2026-01-02" in cmd
    assert "/workspace/runs/ds/2026-01-03" in cmd
    assert "--profile" in cmd
    assert "full" in cmd
    assert "--recompute-missing-metrics" in cmd
    assert "yes" in cmd
    assert "/workspace/datasets/ds/data.yaml" in cmd

    tok = shlex.split(cmd)
    # Replay appends Typer-only `--nit`; argparse round-trip uses the module parser without Typer.
    tok = [t for t in tok if t not in TYPER_META_NON_INTERACTIVE_FLAGS]
    ns2 = root.parse_args(tok[tok.index("all") :])
    assert ns2.others == ns.others


def test_analyze_all_replay_nit_stripped_before_argparse() -> None:
    """Typer analyze forwarding must drop replay suffix ``--nit`` before argparse."""
    root = build_analyze_arg_parser()
    p_all = _p_all_subparser(root)
    ns = root.parse_args(
        [
            "all",
            "--baseline",
            "/workspace/runs/ds/2026-01-01",
            "--others",
            "/workspace/runs/ds/2026-01-02",
            "--profile",
            "full",
        ]
    )
    cmd = build_non_interactive_command("analyze all", p_all, ns)
    tok = shlex.split(cmd)
    assert tok[-1] == "--nit"

    # Simulate ``_forward_analyze_command``: prepend subcommand, strip Typer meta flags.
    from smartrain.cli_entrypoints.support.typer_non_interactive import strip_typer_meta_non_interactive_flags

    ctx_args = tok[tok.index("all") + 1 :]
    filtered, stripped = strip_typer_meta_non_interactive_flags(ctx_args)
    assert stripped is True
    assert "--nit" not in filtered

    ns2 = root.parse_args(["all", *filtered])
    assert ns2.baseline == ns.baseline
    assert ns2.others == ns.others
    assert ns2.profile == ns.profile
