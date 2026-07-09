"""Grouped plain-text CLI help registry (English)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandHelp:
    name: str
    summary: str
    description: str


COMMANDS: dict[str, CommandHelp] = {
    "deploy": CommandHelp(
        name="deploy",
        summary="Create workspace directories and index files if missing.",
        description=(
            "Initialize the workspace layout under the target directory: raw_data/, datasets/, "
            "runs/, analytics/, models/, inference/, tmp/, and empty datasets_info.json files."
        ),
    ),
    "info": CommandHelp(
        name="info",
        summary="Show supported train model aliases and installed providers.",
        description=(
            "Print detection model aliases available for training and list externally installed "
            "training providers from the user config index."
        ),
    ),
    "quickstart": CommandHelp(
        name="quickstart",
        summary="Print step-by-step getting-started workflow guide.",
        description=(
            "Show the end-to-end workflow from workspace setup through scan, train, reports, "
            "and run analysis. Useful for first-time orientation."
        ),
    ),
    "scan": CommandHelp(
        name="scan",
        summary="Discover raw_data sources and refresh datasets_info.json.",
        description=(
            "Scan workspace raw_data/ (or explicit source paths), detect supported dataset layouts, "
            "normalize metadata, and update datasets/ catalog files."
        ),
    ),
    "sync": CommandHelp(
        name="sync",
        summary="Safely sync missing workspace artifacts from another copy.",
        description=(
            "Copy only missing datasets/raw_data/runs/models from another workspace, skip conflicts, "
            "repair portable paths, and run final scan in target workspace."
        ),
    ),
    "normalize-data-yaml": CommandHelp(
        name="normalize-data-yaml",
        summary="Rewrite datasets/*/data.yaml to portable Ultralytics layout.",
        description=(
            "Remove absolute path keys and normalize train/valid/test splits to relative paths "
            "under each dataset directory."
        ),
    ),
    "merge": CommandHelp(
        name="merge",
        summary="Merge multiple source datasets into one training dataset.",
        description=(
            "Build a merged dataset from catalog entries with class filtering, remapping, and "
            "optional output naming under workspace datasets/."
        ),
    ),
    "fusion": CommandHelp(
        name="fusion",
        summary="Deprecated alias for merge.",
        description=(
            "Backward-compatible alias for `smartrain merge`. New workflows should use merge."
        ),
    ),
    "split": CommandHelp(
        name="split",
        summary="Repartition one dataset into train/valid/test splits.",
        description=(
            "Shuffle and split image/label pairs into new split folders while preserving class "
            "structure and updating data.yaml."
        ),
    ),
    "augment": CommandHelp(
        name="augment",
        summary="Run offline augmentation and write a new dataset.",
        description=(
            "Apply configurable geometric and photometric augmentations offline and register the "
            "result as a new workspace dataset."
        ),
    ),
    "balance": CommandHelp(
        name="balance",
        summary="Balance class distribution and write a new dataset.",
        description=(
            "Oversample, undersample, or rebalance selected classes to reduce skew before training."
        ),
    ),
    "prune": CommandHelp(
        name="prune",
        summary="Remove empty pairs, duplicates, or unused classes.",
        description=(
            "Subcommands empty, dedup, and classes clean dataset artifacts and optionally emit "
            "reports before writing a pruned copy."
        ),
    ),
    "hash": CommandHelp(
        name="hash",
        summary="Calculate or validate dataset content hash.",
        description=(
            "Compute stable hashes for dataset directories and compare against stored metadata for "
            "reproducibility checks."
        ),
    ),
    "stats": CommandHelp(
        name="stats",
        summary="Show dataset statistics; compare datasets or runs.",
        description=(
            "Print per-class counts, split breakdowns, and optional charts. The compare subcommand "
            "contrasts two datasets side by side."
        ),
    ),
    "orient": CommandHelp(
        name="orient",
        summary="Normalize image EXIF orientation into a new dataset.",
        description=(
            "Read orientation metadata, rotate images and labels to upright, and export a corrected "
            "dataset copy."
        ),
    ),
    "rotate": CommandHelp(
        name="rotate",
        summary="Rotate dataset images and labels by 90/180/270 degrees.",
        description=(
            "Apply fixed clockwise rotation to all samples and write a new dataset with updated "
            "bounding boxes or polygons."
        ),
    ),
    "roi": CommandHelp(
        name="roi",
        summary="Apply ROI crop and export a new dataset.",
        description=(
            "Crop images to a region of interest using manual or model-guided boxes and regenerate "
            "labels in cropped coordinates."
        ),
    ),
    "dataset": CommandHelp(
        name="dataset",
        summary="Dataset catalog management: convert, reports, and rename.",
        description=(
            "Convert datasets between CVAT 1.1, YOLO, and CvsDclDet; generate multilingual per-class "
            "sample reports under analytics/datasets-reports/; rename workspace datasets with reference "
            "propagation across runs/, models/, queue.txt, analytics/, and dataset_passport.json. "
            "Subcommands: convert, report, rename."
        ),
    ),
    "train": CommandHelp(
        name="train",
        summary="Train YOLO and write run artifacts to runs/.",
        description=(
            "Launch Ultralytics training from a workspace dataset or profile. Writes weights, "
            "metrics, args.yaml, and training_metadata.json under runs/<dataset>/<timestamp>/."
        ),
    ),
    "queue": CommandHelp(
        name="queue",
        summary="Manage deferred training queue (list/add/remove/clear/run).",
        description=(
            "Persist training commands in workspace queue.txt, inspect worker status, and start the "
            "queue executor. Subcommands: list, add, remove, clear, run."
        ),
    ),
    "queue-run": CommandHelp(
        name="queue-run",
        summary="Run queue executor as a top-level command.",
        description=(
            "Start the background queue worker that executes pending training commands from "
            "queue.txt. Alias for smartrain queue run."
        ),
    ),
    "test": CommandHelp(
        name="test",
        summary="Complete missing test artifacts for runs/models.",
        description=(
            "Run validation and export tests (pt, onnx, engine, etc.) for a training run and fill "
            "gaps in test metrics and artifacts."
        ),
    ),
    "inference": CommandHelp(
        name="inference",
        summary="Run inference and save JSON report to workspace inference/.",
        description=(
            "Batch inference from folder, dataset split, or promoted model with optional ROI "
            "pre-detection and structured JSON output."
        ),
    ),
    "vis": CommandHelp(
        name="vis",
        summary="Visualize dataset labels and model/run predictions.",
        description=(
            "Generate rendered overlays with class names for datasets and for run/model targets "
            "using the training/validation/testing data of that target."
        ),
    ),
    "model": CommandHelp(
        name="model",
        summary="Convert, release, and rename workspace models.",
        description=(
            "Export to ONNX/TensorRT, promote run weights into models/, and rename released "
            "artifacts. Subcommands: convert, release, rename."
        ),
    ),
    "sahi": CommandHelp(
        name="sahi",
        summary="Run tiled SAHI inference for large images.",
        description=(
            "Slice large images, run detection on tiles, merge results, and save predictions for "
            "high-resolution inputs."
        ),
    ),
    "heatmap": CommandHelp(
        name="heatmap",
        summary="Generate heatmap visualization from image and model.",
        description=(
            "Render class activation-style heatmaps for a single image and trained weights."
        ),
    ),
    "analyze": CommandHelp(
        name="analyze",
        summary="Analyze training runs: tables, comparisons, PR curves, speed.",
        description=(
            "Scan runs/, export CSV summaries, compare quality, benchmark inference, and build "
            "leaderboards. Subcommands include all, scan, export-table, compare, pr-curves, "
            "inference-benchmark, inference-plot, test-metrics-plot, leaderboard."
        ),
    ),
    "plot": CommandHelp(
        name="plot",
        summary="Legacy analytics wrapper (prefer smartrain analyze).",
        description=(
            "Backward-compatible entry point that forwards to older analyze/plot workflows. "
            "New projects should use smartrain analyze instead."
        ),
    ),
    "registry": CommandHelp(
        name="registry",
        summary="Registry of runs and promoted models.",
        description=(
            "List and inspect training runs, query metrics, and manage promoted model entries. "
            "Subcommands: runs-list, runs-info, runs-metrics, models-add, models-list, "
            "models-info, models-remove."
        ),
    ),
    "migrate": CommandHelp(
        name="migrate",
        summary="Unified migration utilities for workspace artifacts.",
        description=(
            "Migrate legacy run layouts, metadata, and paths to current canonical formats with "
            "dry-run and apply modes."
        ),
    ),
    "migrate-models": CommandHelp(
        name="migrate-models",
        summary="Migrate legacy models for analyze compatibility.",
        description=(
            "Normalize older model files and metadata so they work with current analyze and "
            "registry tooling."
        ),
    ),
    "providers": CommandHelp(
        name="providers",
        summary="Install/uninstall/status for external training providers.",
        description=(
            "Manage optional external YOLO provider repositories. Subcommands: install, uninstall, "
            "status, doctor."
        ),
    ),
    "clearml-upload": CommandHelp(
        name="clearml-upload",
        summary="Upload completed run to ClearML.",
        description=(
            "Push run artifacts, metrics, and metadata to a configured ClearML project for "
            "experiment tracking."
        ),
    ),
    "deps": CommandHelp(
        name="deps",
        summary="Dependency management helpers.",
        description=(
            "Sync PyTorch/CUDA packages to the recommended policy. Subcommand: sync-torch."
        ),
    ),
}

COMMAND_GROUPS: list[tuple[str, list[str]]] = [
    (
        "Workspace",
        ["deploy", "info", "quickstart", "sync"],
    ),
    (
        "Dataset catalog and preparation",
        [
            "scan",
            "normalize-data-yaml",
            "merge",
            "fusion",
            "split",
            "augment",
            "balance",
            "prune",
            "hash",
            "stats",
            "orient",
            "rotate",
            "roi",
            "dataset",
        ],
    ),
    (
        "Training",
        ["train", "queue", "queue-run"],
    ),
    (
        "Model and inference",
        ["test", "inference", "vis", "model", "sahi", "heatmap"],
    ),
    (
        "Analytics",
        ["analyze", "plot"],
    ),
    (
        "Registry and artifacts",
        ["registry", "migrate", "migrate-models"],
    ),
    (
        "Integrations",
        ["providers", "clearml-upload", "deps"],
    ),
]

HELP_EPILOG = (
    "Tip: smartrain <command> -- --help   # detailed flags for argparse-backed commands\n"
    "Tip: smartrain quickstart             # step-by-step getting-started guide"
)


def command_summary(name: str) -> str:
    entry = COMMANDS.get(name)
    return entry.summary if entry is not None else ""


def command_description(name: str) -> str:
    entry = COMMANDS.get(name)
    return entry.description if entry is not None else ""
