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
    p.add_argument("--source-dir", type=str, default=None, help="Folder with images (recursive).")
    p.add_argument("--dataset", type=str, default=None, help="Dataset key from datasets/datasets_info.json.")
    p.add_argument("--split", choices=("train", "val", "test"), default="test", help="Dataset split for dataset-split mode.")
    p.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all).")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for inference model.")
    p.add_argument("--img-size", type=int, default=None, help="Inference input resolution (imgsz).")
    p.add_argument("--device", type=str, default=None, help="Ultralytics device (cpu, 0, etc). Default: GPU 0 if available, otherwise cpu.")
    p.add_argument("--half", action="store_true", help="Enable FP16 where supported.")
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
    return p

