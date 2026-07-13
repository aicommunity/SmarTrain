"""CLI help texts and argparse examples."""

from __future__ import annotations

from pathlib import Path

import typer


def _print_en_quick_start() -> None:
    """Print getting-started guide as plain text."""
    quickstart_path = Path(__file__).resolve().parent.parent.parent / "docs" / "getting-started" / "quickstart.md"
    fallback = (
        "Quick start\n\n"
        "Run from workspace root:\n\n"
        "  smartrain --install-completion\n"
        "  smartrain --show-completion\n"
        "  smartrain deploy\n"
        "  smartrain scan\n"
        "  smartrain train --data my_dataset --model yolo11n.pt -y\n"
        "  smartrain dataset report --dataset my_dataset -n 6 --languages en,ru\n"
        "  smartrain analyze scan\n"
        "  smartrain analyze all --report-languages en,ru\n"
    )
    try:
        text = quickstart_path.read_text(encoding="utf-8")
    except OSError:
        text = fallback
    typer.echo(text)

HELP_ANALYZE_GROUP = """Analyze training runs: summary tables, comparisons, PR curves, and inference speed.

Quick start:
  smartrain analyze
  smartrain analyze all
  smartrain analyze scan
  smartrain analyze export-table -o runs_summary.csv
  smartrain analyze compare --baseline runs/ds_a/2026-01-01_00-00-00 --others runs/ds_a/2026-01-02_00-00-00
  smartrain analyze inference-benchmark --runs-group-dir runs/ds_a --data-yaml datasets/ds_a/data.yaml
  smartrain analyze leaderboard -o analytics/leaderboard.csv

Common patterns:
  summary CSV: analyze export-table
  quality compare: analyze compare
  speed analysis: analyze inference-benchmark + analyze inference-plot
"""

HELP_QUEUE_GROUP = """Queue management for deferred training runs.

Quick examples:
  smartrain queue list
  smartrain queue add --cmd "smartrain train --data my_dataset -y"
  smartrain queue run --no-gui
"""

HELP_REGISTRY_GROUP = """Registry of runs and promoted models.

Quick examples:
  smartrain registry runs-list
  smartrain registry runs-info --run-dir runs/my_dataset/2026-01-01_00-00-00
  smartrain registry models-list
"""

HELP_DATASET_GROUP = """Dataset catalog management and sample reports.

Default report output: workspace `analytics/datasets-reports/<dataset>_<timestamp>/`.

Quick examples:
  smartrain dataset report --dataset my_dataset
  smartrain dataset report --dataset my_dataset -n 6 --languages en,ru
  smartrain dataset rename --dataset old_name --new-name new_name
  smartrain dataset rename --dataset old_name --new-name new_name --dry-run
  smartrain dataset rename
"""

HELP_MODEL_GROUP = """Model conversion tools.

Quick examples:
  smartrain model convert
  smartrain model convert --input models/best.pt --format onnx
  smartrain model convert --input runs/my_ds/2026-01-01_00-00-00/2026-01-01_00-00-00.pt --format tensorrt-engine --precision fp16
  smartrain model convert --input models/my_model.onnx --format tensorrt-trt
  smartrain model release --run runs/my_ds/2026-01-01_00-00-00
  smartrain model rename --release models/my_ds/detect_yolov8n_20260115_120000.pt --new-name my_detector_v2

Interactive convert:
  - choose source model type: pt or onnx
  - select a file (or enter a manual path)
  - select one or multiple target models (onnx/engine/trt depending on source; CSV by numbers or values is supported, e.g. 1,3 or onnx,trt)
  - set batch/imgsz and other export parameters
  - run sources use canonical artifacts <run_dir>/<run_dir_name>.<ext>; legacy run layouts are canonized automatically

Artifacts:
  - tensorrt-engine: Ultralytics export to .engine
  - tensorrt-trt: trtexec export to .trt
"""

HELP_DEPS_GROUP = """Dependency management helpers.

Quick examples:
  smartrain deps sync-torch
"""

ARGPARSE_HELP_EXAMPLES: dict[str, str] = {
    "smartrain train": (
        "Examples:\n"
        "  smartrain train --data 2026-01-01_12-00-00-merged -y\n"
        "  smartrain train --data my_dataset --model yolo11n.pt --epochs 50\n"
        "  smartrain train --data my_dataset --batch 16 --img-size 1024\n"
    ),
    "smartrain dataset convert": (
        "Examples:\n"
        "  smartrain dataset convert\n"
        "  smartrain dataset convert --source task.zip --to yolo --output-dir datasets/task_yolo\n"
        "  smartrain dataset convert --source datasets/task_yolo --to cvat11 --output-dir converted_raw_data/task --zip\n"
        "  smartrain dataset convert --source raw_data/my_det.zip --to cvat11 --output-dir converted_raw_data/my_det\n"
        "  smartrain dataset convert --source raw_data/my_det --to cvat11 --rename-classes white_line line --zip\n"
    ),
    "smartrain rotate": (
        "Examples:\n"
        "  smartrain rotate\n"
        "  smartrain rotate --dataset my_dataset --angle 90\n"
        "  smartrain rotate --dataset my_dataset --angle 270 --output-name my_dataset_rot270\n"
    ),
    "smartrain sahi": (
        "Examples:\n"
        "  smartrain sahi --model models/best.pt --source images/\n"
        "  smartrain sahi --model models/best.pt --source image.jpg --output sahi_out\n"
        "  smartrain sahi --model models/best.pt --source images/ --slice-h 768 --slice-w 768\n"
    ),
    "smartrain heatmap": (
        "Examples:\n"
        "  smartrain heatmap --model models/best.pt --source image.jpg\n"
        "  smartrain heatmap --model models/best.pt --source image.jpg --output heatmap.png\n"
        "  smartrain heatmap --model models/best.pt --source image.jpg --colormap 12\n"
    ),
    "smartrain filter": (
        "Examples:\n"
        "  smartrain filter --dataset my_dataset\n"
        "  smartrain filter --dataset my_dataset --stats-only\n"
        "  smartrain filter --dataset my_dataset --dry-run --baseline-inset-margin 0.01\n"
        "  smartrain filter --dataset my_dataset --edge-sides horizontal\n"
        "  smartrain filter --dataset my_dataset --no-edge-filter --size-filter\n"
        "  smartrain filter --dataset my_dataset --size-filter --size-baseline-mode stable --size-dims width\n"
    ),
    "smartrain dataset report": (
        "Examples:\n"
        "  smartrain dataset report --dataset my_dataset\n"
        "  smartrain dataset report --dataset my_dataset -n 6 --languages en,ru\n"
        "  smartrain dataset report --workspace /data/ws --dataset my_dataset --no-odt\n"
    ),
    "smartrain inference": (
        "Examples:\n"
        "  smartrain inference --model-name my_promoted_model --data-mode folder --source-dir raw_images\n"
        "  smartrain inference --model-name my_promoted_model --data-mode folder --source-dir raw_images --no-export-dataset\n"
        "  smartrain inference --model-name my_promoted_model --data-mode dataset-split --dataset my_dataset --split test --limit 200\n"
        "  smartrain inference --run 1 --data-mode folder --source-dir samples --roi-pre-detect --roi-weights yolo11n.pt\n"
    ),
}
