from __future__ import annotations

from smartrain.cli_support.cli_replay import build_non_interactive_command
from smartrain.workflows.datasets.dataset_augment import build_augment_arg_parser
from smartrain.workflows.datasets.dataset_balance import build_balance_arg_parser
from smartrain.workflows.datasets.dataset_former import build_dataset_former_arg_parser
from smartrain.workflows.datasets.dataset_orient import build_orient_arg_parser
from smartrain.workflows.datasets.dataset_roi_yolo import build_roi_arg_parser
from smartrain.workflows.datasets.dataset_stats import build_stats_arg_parser, build_stats_compare_arg_parser
from smartrain.workflows.training.model_training_module import build_train_arg_parser


def test_replay_boolean_optional_action_does_not_emit_true_false_values() -> None:
    parser = build_dataset_former_arg_parser()
    args = parser.parse_args(
        [
            "--workspace",
            "/tmp/ws",
            "--output-name",
            "merged",
            "--dataset",
            "ds_a",
            "--include-partial-datasets",
        ]
    )
    cmd = build_non_interactive_command("fusion", parser, args)
    assert "--include-partial-datasets" in cmd
    assert "True" not in cmd
    assert "False" not in cmd


def test_replay_fusion_boolean_optional_false_emits_negative_flag() -> None:
    parser = build_dataset_former_arg_parser()
    args = parser.parse_args(
        [
            "--workspace",
            "/tmp/ws",
            "--output-name",
            "merged",
            "--dataset",
            "ds_a",
            "--no-include-partial-datasets",
        ]
    )
    cmd = build_non_interactive_command("fusion", parser, args)
    assert "--no-include-partial-datasets" in cmd
    assert "--include-partial-datasets" not in cmd
    assert "True" not in cmd and "False" not in cmd


def test_replay_other_interactive_commands_do_not_emit_python_bools() -> None:
    cases: list[tuple[str, object, list[str], list[str]]] = [
        (
            "train",
            build_train_arg_parser(),
            ["--workspace", "/tmp/ws", "--data", "ds", "--epochs", "1", "--batch", "2", "-y"],
            ["-y"],
        ),
        (
            "augment",
            build_augment_arg_parser(),
            ["--workspace", "/tmp/ws", "--dataset", "ds", "--enable-flip", "--disable-center-rotate", "--dry-run"],
            ["--enable-flip", "--disable-center-rotate", "--dry-run"],
        ),
        (
            "balance",
            build_balance_arg_parser(),
            ["--workspace", "/tmp/ws", "--dataset", "ds", "--dry-run", "--emit-balance-report"],
            ["--dry-run", "--emit-balance-report"],
        ),
        (
            "stats",
            build_stats_arg_parser(),
            ["--workspace", "/tmp/ws", "--dataset", "ds", "--balance-ready", "--class-desc"],
            ["--balance-ready", "--class-desc"],
        ),
        (
            "stats compare",
            build_stats_compare_arg_parser(),
            ["--workspace", "/tmp/ws", "--left", "a", "--right", "b", "--abs", "--export-json"],
            ["--abs", "--export-json"],
        ),
        (
            "roi",
            build_roi_arg_parser(),
            ["--workspace", "/tmp/ws", "--dataset-name", "ds", "--weights", "/tmp/m.pt", "--images-only"],
            ["--images-only"],
        ),
        (
            "orient",
            build_orient_arg_parser(),
            ["--workspace", "/tmp/ws", "--dataset", "ds", "--report-only", "--dry-run"],
            ["--report-only", "--dry-run"],
        ),
    ]

    for cmd_name, parser, argv, expected_flags in cases:
        args = parser.parse_args(argv)
        cmd = build_non_interactive_command(cmd_name, parser, args)
        assert " True" not in cmd and " False" not in cmd
        for flag in expected_flags:
            assert flag in cmd

