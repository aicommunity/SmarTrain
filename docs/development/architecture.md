> Russian version: [../ru/development/architecture.md](../ru/development/architecture.md)

# Architecture and diagrams

This document captures real code flows and helps you quickly locate where to make changes.

Sources of truth for this section: `smartrain/cli.py`, `smartrain/workflows/training/model_training_module.py`, `smartrain/workflows/analyze/results_analyzer.py`, `smartrain/training_queue.py`, `smartrain/core/runtime/workspace_paths.py`, `smartrain/providers/cli.py`, `smartrain/providers/core/global_index.py`.

## 1) Top-level architecture

```mermaid
flowchart TD
  cli[smartrain CLI] --> datasetPipeline[Dataset_pipeline]
  cli --> trainingFlow[Training_pipeline]
  cli --> queueFlow[Queue_pipeline]
  cli --> analyticsFlow[Analytics_and_registry]
  datasetPipeline --> workspaceFs[File_workspace]
  trainingFlow --> workspaceFs
  queueFlow --> workspaceFs
  analyticsFlow --> workspaceFs
```

What it shows: four core system flows connected through the file-based workspace.
How to read: start at `smartrain CLI`, then follow subsystem links to filesystem state.
Practical takeaway: changes to file contracts affect multiple commands at once.

## 2) Sequence `smartrain train`

```mermaid
sequenceDiagram
  participant User
  participant CLI as cli.py
  participant Train as model_training_module.py
  participant Profile as core/training/train_profile.py
  participant YOLO as ultralytics.YOLO
  User->>CLI: smartrain train ...
  CLI->>Train: main(argv)
  Train->>Profile: merge parameters
  Train->>Train: resolve dataset and runtime data.yaml
  Train->>YOLO: train()
  Train->>YOLO: val()
  Train->>Train: write metrics and training_metadata.json
```

What it shows: the complete path from the CLI call to the learning artifacts.
How to read: from top to bottom along the time axis, with each call clarifying the context.
Practical takeaway: if parameters are wrong, check the profile-merge stage before `YOLO.train()` starts.

## 3) Life cycle of data in the workspace

```mermaid
flowchart TD
  rawData[raw_data and external sources] --> scan[scan]
  scan --> datasets[datasets and datasets_info.json]
  datasets --> fusion[fusion or augment or balance or roi]
  fusion --> train[train]
  train --> runs[runs]
  runs --> analyze[analyze]
  runs --> registry[registry `models-add`]
  analyze --> analytics[analytics]
  registry --> models[models]
```

What it shows: how data moves between major directories.
How to read: follow the flow left to right, from sources to final artifacts.
Practical takeaway: if the next pipeline stage fails, first verify `datasets_info.json` integrity and `fusion` outputs.

## 4) Queue task states

```mermaid
stateDiagram-v2
  [*] --> Waiting
  Waiting --> Running: executor picks task
  Running --> Done: return code 0
  Running --> Error: return code != 0
  Done --> [*]
  Error --> [*]
```

What it shows: row statuses from `queue.txt`.
How to read: transitions are driven by command execution results.
Practical takeaway: retry handling is not automated; restarts are manual.

Note on terms: The statuses in the diagram (`Waiting`, `Running`, `Done`, `Error`) correspond to the actual rows in `status.txt`.

## 5) Artifact Contracts

```mermaid
flowchart TD
  scanOut[datasets_info.json and class_names.json] --> fusionInput[fusion]
  fusionInput --> dataYaml[data.yaml]
  dataYaml --> trainRun[run directory]
  trainRun --> trainingMeta[training_metadata.json]
  trainRun --> metricsCsv[test_metrics.csv]
  trainRun --> registryManifest[model_manifest.json]
  trainRun --> analyzeTable[analyze export-table csv]
```

What it shows: dependencies between contract files.
How to read: each arrow means one artifact becomes input to the next step.
Bottom line: any changes to the `training_metadata.json` flow require re-validating `analyze` and `registry`.

## 6) `scan/fusion` path for ZIP and CVAT 1.1 sources

```mermaid
sequenceDiagram
  participant User
  participant Scan as datasets_json_former.py
  participant Access as dataset_access.py
  participant CVAT as cvat11_converter.py
  participant Fusion as dataset_former.py
  User->>Scan: smartrain scan
  Scan->>Access: resolve source root
  Access-->>Scan: internal structure ID cvat11 (CVAT for images 1.1) or zip
  User->>Fusion: smartrain fusion ...
  Fusion->>Access: iter image and label buckets
  Access->>CVAT: temporary labels from annotations.xml
  CVAT-->>Fusion: yolo-compatible label stream
  Fusion-->>User: merged dataset
```

What it shows: how CVAT/zip sources are reduced to a single stream for merge.
How to read: conversion happens during data access; no separate import step is required.
Practical takeaway: `cvat import` is not always needed, because `fusion` can work natively with CVAT for images 1.1 (internal ID `cvat11`).
