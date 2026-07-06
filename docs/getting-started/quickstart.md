> Russian version: [../ru/getting-started/quickstart.md](../ru/getting-started/quickstart.md)

# Quick start

This guide assumes `smartrain` is already installed and you run commands from the workspace root.

Run `smartrain` or `smartrain --help` for a grouped command reference. This guide is also available as `smartrain quickstart`.

## Main path: train on your dataset and build reports

1. **Create a workspace directory (if it does not exist yet)**
   `deploy` should be run from inside this workspace directory.

Launch mode: non-interactive.

```bash
mkdir -p /path/to/my_workspace
cd /path/to/my_workspace
```

2. **Initialize workspace layout once**
   Creates standard folders: `raw_data/`, `datasets/`, `runs/`, `analytics/`, etc.

Launch mode: non-interactive.

```bash
smartrain deploy
```

3. **Add dataset sources and index them**
   `scan` discovers supported source layouts, normalizes metadata, and updates dataset index files.

Launch mode: non-interactive.

```bash
# Option A: sources already in workspace/raw_data/
smartrain scan

# Option B: pass explicit source paths
smartrain scan --dataset /data/datasets/my_dataset
```

Supported dataset layouts:

- YOLO split directories layout
- YOLO flat paired directories layout
- YOLO flat with subset subfolders
- YOLO nested split under images/labels
- Darknet YOLO dataset layout
- CVAT for images 1.1 layout

Supported annotations: YOLO bbox (`class_id cx cy w h`) and segmentation polygons.
For precise mapping to SmarTrain internal IDs and layout differences, see `docs/reference/data-formats.md`.

4. **Train model on selected dataset**
   Produces a run with weights, metrics, and `training_metadata.json` in `runs/...`.

Launch modes:

- Interactive (no args): `smartrain train`
- Non-interactive (minimal args):

```bash
smartrain train --data my_dataset --model yolo11n.pt -y
```

**Instance segmentation (optional):**

```bash
smartrain train --data my_seg_dataset --task segment --model yolo11s-seg.pt -y
smartrain test --run <run> --formats pt --task segment
smartrain inference --model <best.pt> --source images/ --task segment
```

5. **Build dataset report**
   Generates visual/text report for dataset quality and class coverage.

Launch modes:

- Interactive (no args): `smartrain dataset report`
- Non-interactive (minimal args):

```bash
smartrain dataset report --dataset my_dataset -n 6 --languages en,ru
```

6. **Build run analysis report**
   Builds final analytical report artifacts for runs.

Launch modes:

- Interactive (no args): `smartrain analyze`
- Non-interactive (minimal args):

```bash
smartrain analyze all --report-languages en,ru
```

`smartrain analyze scan` is optional and can be used as a separate pre-check, but it is not required before `smartrain analyze`/`smartrain analyze all`. A separate `smartrain test` run and test/val dataset splits are not required for the report; speed/PR stages degrade with warnings when splits are missing (`profile=full`).

## Optional steps

- **Compare different training runs**
  Launch mode: non-interactive.
  ```bash
  smartrain analyze compare --baseline /path/to/run_a --others /path/to/run_b /path/to/run_c
  ```
- **Dataset operations before training** (when needed): `fusion`, `augment`, `balance`, `prune`, `orient`, `roi`.

## Where outputs are saved

- Dataset reports: `analytics/datasets-reports/<dataset>_<timestamp>/`
- Analyze sessions: `analytics/analyze-reports/<session>/`
- Training runs: `runs/<dataset>/<run>/`
