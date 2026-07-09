"""Train CLI argparse builders."""

from __future__ import annotations

import argparse

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR

MODEL_VERSION = "yolov8n"
EPOCHS = 50
BATCH = 16
IMG_SIZE = 640


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        description="Training models (without arguments, interactive mode starts)"
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Root workspace (otherwise {WORKSPACE_ENV_VAR}); runs in runs/, resolution --data by datasets",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="YAML profile smart-train (basic config). Can be mixed with --ultralytics_yaml; priority CLI > --ultralytics_yaml > --config",
    )
    parser.add_argument(
        "--ultralytics_yaml",
        type=str,
        default=None,
        help="External Ultralytics args.yaml; incompatible keys (data/project/name/exist_ok/...) are ignored with a warning",
    )
    parser.add_argument(
        "--base-run-args-yaml",
        type=str,
        default=None,
        help="Path to args.yaml of the base run (used as a source of defaults in interactive mode)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Directory with data.yaml (absolute/relative) or record name from datasets/datasets_info.json; "
        "with --workspace it is usually set explicitly (the data value from --ultralytics_yaml is not used)",
    )

    parser.add_argument(
        "--task",
        type=str,
        default=argparse.SUPPRESS,
        help="Ultralytics task: detect, segment, classify, pose, obb (default from profile or detect)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=argparse.SUPPRESS,
        help=(
            f"Model (default {MODEL_VERSION} or from profile --config). "
            "Specify full alias/weights including scale (n/s/m/l/x), e.g. yolo11x.pt."
        ),
    )
    parser.add_argument(
        "--pretrained-run",
        type=str,
        default=None,
        help="Use .pt weights from an existing run directory as initialization.",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Use weights from promoted model directory in workspace/models.",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        default=None,
        help="Use explicit .pt path as initialization weights.",
    )
    parser.add_argument(
        "--external-provider",
        type=str,
        default=None,
        help="Use external provider id for training (runs via isolated provider venv).",
    )
    parser.add_argument(
        "--external-repo",
        type=str,
        default=None,
        help="Override external provider repository path.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Epoches (default {EPOCHS} or from profile)",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Batch (default {BATCH} or from profile)",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=argparse.SUPPRESS,
        help=f"imgsz (default {IMG_SIZE} or from profile)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=argparse.SUPPRESS,
        help="Compute device for training (e.g. 0, 1, cpu). Default: GPU 0 if available, otherwise cpu.",
    )

    parser.add_argument(
        "--target-path",
        type=str,
        default=None,
        help="Base directory for runs (defaults to workspace/runs when using workspace)",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to the folder with the model",
    )

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Perform testing only without training",
    )

    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        dest="non_interactive",
        help="Do not ask for confirmation if there is an existing results folder (for queue and CI)",
    )

    parser.add_argument(
        "--val-imgsz",
        type=int,
        default=None,
        help="Image size for val/test (default as --img-size when training)",
    )
    parser.add_argument(
        "--val-conf",
        type=float,
        default=None,
        help="conf threshold for val() (Ultralytics)",
    )
    parser.add_argument(
        "--val-iou",
        type=float,
        default=None,
        help="IoU threshold for val() (Ultralytics)",
    )
    parser.add_argument(
        "--val-batch",
        type=int,
        default=None,
        help="Batch for val/test (by default: as a training batch; for --test-only it is taken from training_metadata.json if available)",
    )
    parser.add_argument(
        "--conf-rec-beta-recall",
        type=float,
        default=2.0,
        help="Beta for objective B (recall-priority F-beta) in confidence recommendations.",
    )
    parser.add_argument(
        "--conf-rec-beta-precision",
        type=float,
        default=0.5,
        help="Beta for objective C (precision-priority F-beta) in confidence recommendations.",
    )
    parser.add_argument(
        "--conf-rec-fallback",
        type=float,
        default=0.25,
        help="Fallback confidence value when recommendations cannot be computed.",
    )
    parser.add_argument(
        "--conf-rec-disable",
        action="store_true",
        help="Disable confidence threshold recommendation computation.",
    )

    parser.add_argument(
        "--weighted-sampling",
        action="store_true",
        help="Weighted image sampling (classes with fewer objects more often); ultralytics patch",
    )

    parser.add_argument(
        "--clearml",
        action="store_true",
        help="Logging hyperparameters in ClearML (need pip install clearml)",
    )
    parser.add_argument(
        "--clearml-project",
        type=str,
        default=None,
        help="ClearML project name (aka CLEARML_PROJECT or smartrain)",
    )

    return parser


def parse_args(argv=None):
    return build_train_arg_parser().parse_args(argv)


def build_train_resume_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="smartrain train resume",
        description="Resume an interrupted training run",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR})",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Absolute or workspace-relative run directory to resume",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        dest="non_interactive",
        help="Non-interactive mode. Requires --run-dir.",
    )
    parser.add_argument(
        "--test-batch",
        type=int,
        default=None,
        help="Start batch for resume test stage (training_complete_test_pending).",
    )
    parser.add_argument(
        "--test-batch-min",
        type=int,
        default=1,
        help="Minimum batch for OOM backoff in resume test stage.",
    )
    parser.add_argument(
        "--test-batch-backoff",
        type=int,
        default=2,
        help="OOM backoff divider for test batch (e.g. 2 means batch/2).",
    )
    return parser


def build_train_calc_confidence_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="smartrain train calc-confidence",
        description="Calculate confidence recommendations for existing runs",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR})",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Absolute or workspace-relative run directory. Can be used multiple times.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all discovered run directories.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        dest="non_interactive",
        help="Non-interactive mode. If no --run-dir is given, all runs are processed.",
    )
    parser.add_argument(
        "--val-batch",
        type=int,
        default=None,
        help="Batch size for val/test recompute in calc-confidence (default: 1).",
    )
    return parser


def parse_train_args(argv=None):
    return build_train_arg_parser().parse_args(argv)
