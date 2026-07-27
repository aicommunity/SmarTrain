from __future__ import annotations

import argparse

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR
from smartrain.services.inference_runtime_helpers import DATA_MODES, ON_EMPTY_MODES, ROI_POLICIES


def build_inference_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Run object detection inference and save JSON report (empty call starts interactive mode)."
    )
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR}).")
    p.add_argument("--model-name", type=str, default=None, help="Promoted model directory name from workspace/models.")
    p.add_argument("--run", type=str, default=None, help="Run path or run index from workspace/runs list.")
    p.add_argument("--weights", type=str, default=None, help="Explicit model weights path (.pt/.onnx/.engine/.trt).")
    p.add_argument("--data-mode", choices=DATA_MODES, default="folder", help="Data source mode.")
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="Folder or archive with images (.zip, .tar, .tar.gz, .tgz) for folder mode.",
    )
    p.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Folder or archive with images (.zip, .tar, .tar.gz, .tgz); alias for --source.",
    )
    p.add_argument("--dataset", type=str, default=None, help="Dataset key from datasets/datasets_info.json.")
    p.add_argument("--split", choices=("train", "val", "test"), default=None, help="Dataset split for dataset-split mode.")
    p.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all).")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for inference model.")
    p.add_argument(
        "--confidence-objective",
        type=str,
        choices=("A", "B", "C"),
        default=None,
        help=(
            "Opt-in: read recommended conf from confidence_recommendations JSON "
            "(objective A=F1, B=recall-priority F-β, C=precision-priority). "
            "Requires --confidence-recommendations or a run with recommendations. "
            "Default inference conf stays 0.25 when this flag is omitted."
        ),
    )
    p.add_argument(
        "--confidence-aggregation",
        type=str,
        choices=("macro", "micro"),
        default="macro",
        help="Aggregation for --confidence-objective (default: macro).",
    )
    p.add_argument(
        "--confidence-recommendations",
        type=str,
        default=None,
        help="Path to confidence_recommendations_{split}.json for --confidence-objective.",
    )
    p.add_argument("--img-size", type=int, default=None, help="Inference input resolution (imgsz).")
    p.add_argument("--device", type=str, default=None, help="Ultralytics device (cpu, 0, etc). Default: GPU 0 if available, otherwise cpu.")
    p.add_argument("--half", action="store_true", help="Enable FP16 where supported.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Local Ultralytics inference batch size (default: 8). Ignored for external providers.",
    )
    p.add_argument("--perf-warmup-images", type=int, default=5, help="Warmup images excluded from steady perf statistics.")
    p.add_argument("--roi-pre-detect", action="store_true", help="Pre-detect ROI before inference (folder mode only).")
    p.add_argument("--roi-weights", type=str, default=None, help="ROI detector weights path (.pt/.onnx).")
    p.add_argument("--roi-conf", type=float, default=0.25, help="Confidence threshold for ROI detector.")
    p.add_argument("--roi-policy", choices=ROI_POLICIES, default="largest", help="ROI selection policy.")
    p.add_argument("--roi-pad-px", type=int, default=0, help="Padding in pixels around selected ROI.")
    p.add_argument("--roi-on-empty", choices=ON_EMPTY_MODES, default="full_image", help="Behavior when ROI detector has no detections.")
    p.add_argument("--roi-class-ids", type=str, default=None, help="CSV class ids for ROI detector (empty=all).")
    p.add_argument("--external-provider", type=str, default=None, help="External provider id for inference.")
    p.add_argument("--external-repo", type=str, default=None, help="Override external provider repository path.")
    p.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["detect", "segment", "classify", "detection", "segmentation", "classification"],
        help="Task type hint for task-aware backend routing (default: detection).",
    )
    p.add_argument("--save-overlay", action="store_true", help="For segmentation runs, save polygon overlay images next to inference_results.json.")
    p.add_argument(
        "--export-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export YOLO autolabel dataset under <basename>_autolabeled/ (default: on).",
    )
    p.add_argument(
        "--export-label-conf-min",
        type=float,
        default=0.25,
        help="Minimum confidence for writing labels to autolabel dataset (default: 0.25).",
    )
    p.add_argument(
        "--export-label-conf-max",
        type=float,
        default=1.0,
        help="Maximum confidence for writing labels to autolabel dataset (default: 1.0).",
    )
    p.add_argument(
        "--export-visualize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save prediction overlay images to pred_overlays/ (default: on when --export-dataset, else off).",
    )
    p.add_argument(
        "--export-split-dirs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Split autolabel export into independent part_XXX/ sub-datasets "
            "(and mirror pred_overlays/). Default: on."
        ),
    )
    p.add_argument(
        "--export-files-per-dir",
        type=int,
        default=500,
        help=(
            "Max actually exported images per independent autolabel sub-dataset "
            "(after label conf filter). Default: 500. Used when --export-split-dirs is on."
        ),
    )
    p.add_argument(
        "--export-classes",
        type=str,
        default=None,
        help=(
            "Comma-separated class names or ids to keep when saving results "
            "(empty = all classes). Frames without selected classes are omitted."
        ),
    )
    return p

