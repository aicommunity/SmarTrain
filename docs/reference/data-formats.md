> Russian version: [../ru/reference/data-formats.md](../ru/reference/data-formats.md)

# Reference: data formats

## Basic artifacts

- `datasets/datasets_info.json` — dataset directory.
- `datasets/class_names.json` — normalization of class names.
- `datasets/datasets_scan_summary.json` - summary of changes since `scan`.
- `datasets/<dataset>/dataset_passport.json` — passport of transformations.
- `runs/.../training_metadata.json` — training metadata.
- `queue.txt` and `tmp/status.txt` — queue and its statuses.

## Directory key fields

- `datasets_info.json`: `classes`, `structure`, `elements_count`.
- Synchronization service fields: `dataset_hash`, `source_hash`, `source_ref`, `source_signature`, `modified`.
- Optional fields: `data_path`, `tags`, `roi_auto`.

## Supported dataset structures

- `split`
- `flat`
- `subset_flat`
- `nested_split`
- `darknet`
- `cvat11`

## Annotations

Basic YOLO bbox format:

`class_id center_x center_y width height`

Segmentation polygons in `class_id x1 y1 x2 y2 ...` format are also supported.

## Queue and statuses

- Queue file in workspace: `queue.txt` (one command per line).
- Status file: `tmp/status.txt`.
- Basic performer statuses: `Waiting to be completed`, `Running`, `Done`, `Error`.
