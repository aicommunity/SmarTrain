> Russian version: [ru/index.md](ru/index.md)

# Smart Train documentation

This section reflects the current codebase and is organized by scenarios: getting started, daily CLI workflows, reference formats, and architecture.

## 1) Getting started

- [Installation](getting-started/installation.md)
- [Docker (CUDA 12.8)](getting-started/docker.md)
- [Workspace and project directories](getting-started/workspace.md)
- [Quick workflow `scan -> merge -> train`](getting-started/quickstart.md)

## 2) CLI guide

- [Command overview](cli/overview.md)
- [Datasets: `scan`, `merge` (`fusion` alias), `augment`, `balance`, `orient`, `roi`, `hash`, `stats`, `dataset convert`](cli/datasets.md)
- [Training and evaluation: `train`, `clearml-upload`](cli/training.md)
- [External providers: `providers`, provider model aliases, train/inference integration](cli/providers.md)
- [Queue: `queue`, `queue-run`](cli/queue.md)
- [Analytics: `analyze` (including `pr-curves`, `inference-benchmark`, `inference-plot`, `test-metrics-plot`)](cli/analyze.md)
- [Model registry: `registry`](cli/registry.md)
- [Inference tools: `sahi`, `heatmap`](cli/inference-tools.md)

## 2.1) Provider profiles

- [Providers overview](providers/overview.md)
- [DR-YOLO provider](providers/dr-yolo.md)
- [LEAF-YOLO provider](providers/leaf-yolo.md)
- [MFEL-YOLO provider](providers/mfel-yolo.md)
- [MP-YOLO provider](providers/mp-yolo.md)
- [SSDM-YOLO provider](providers/ssdm-yolo.md)
- [Enhanced YOLOv8 provider](providers/enhanced-yolov8.md)

## 3) Reference

- [API and Modules](reference/api.md)
- [Data formats and file contracts](reference/data-formats.md)
- [`training_metadata.json` format](reference/training-metadata.md)

## 4) Development

- [Architecture and diagrams](development/architecture.md)
- [Testing](development/testing.md)
- [Project extensibility](development/extension-guide.md)
- [Developing new external providers](development/provider-development.md)
- [Project audit 2026-07-26](audit/2026-07-26-project-audit.md)

## 5) Migration

- [Legacy commands and compatibility](migration/legacy-commands.md)

## Notes

- Project root overview: [README.md](../README.md).
- Legacy flat documents in `docs/*.md` are kept for backward compatibility and linked from the new sections.

## How to use this documentation

- For quick start: `getting-started -> cli`.
- For production pipeline support: `cli -> reference`.
- For code changes: `development -> reference -> migration`.

