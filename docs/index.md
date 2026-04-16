> Russian version: [ru/index.md](ru/index.md)

# Smart Train documentation

This section reflects the current codebase and is organized by scenarios: getting started, daily CLI workflows, reference formats, and architecture.

## 1) Getting started

- [Installation](getting-started/installation.md)
- [Workspace and project directories](getting-started/workspace.md)
- [Quick workflow `scan -> fusion -> train`](getting-started/quickstart.md)

## 2) CLI guide

- [Command overview](cli/overview.md)
- [Datasets: `scan`, `fusion`, `augment`, `balance`, `orient`, `roi`, `hash`, `stats`](cli/datasets.md)
- [Training and evaluation: `train`, `clearml-upload`](cli/training.md)
- [Queue: `queue`, `queue-run`](cli/queue.md)
- [Analytics: `analyze` (including `pr-curves`, `inference-benchmark`, `inference-plot`, `test-metrics-plot`)](cli/analyze.md)
- [Model registry: `registry`](cli/registry.md)
- [CVAT 1.1: `cvat`](cli/cvat.md)
- [Inference tools: `sahi`, `heatmap`](cli/inference-tools.md)

## 3) Reference

- [API and Modules](reference/api.md)
- [Data formats and file contracts](reference/data-formats.md)
- [`training_metadata.json` format](reference/training-metadata.md)

## 4) Development

- [Architecture and diagrams](development/architecture.md)
- [Testing](development/testing.md)
- [Project extensibility](development/extension-guide.md)

## 5) Migration

- [Legacy commands and compatibility](migration/legacy-commands.md)

## Notes

- Project root overview: [README.md](../README.md).
- Legacy flat documents in `docs/*.md` are kept for backward compatibility and linked from the new sections.

## How to use this documentation

- For quick start: `getting-started -> cli`.
- For production pipeline support: `cli -> reference`.
- For code changes: `development -> reference -> migration`.

