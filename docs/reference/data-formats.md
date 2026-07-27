> Russian version: [../ru/reference/data-formats.md](../ru/reference/data-formats.md)

# Reference: data formats

## Basic artifacts

- `datasets/datasets_info.json` — dataset directory.
- `datasets/class_names.json` — normalization of class names.
- `datasets/datasets_scan_summary.json` - summary of changes since `scan`.
- `datasets/<dataset>/dataset_passport.json` — passport of transformations (e.g. `augment`, `orient`, `rotate` with `fixed_rotation_cw` and `angle` 90/180/270).
- `runs/.../training_metadata.json` — training metadata.
- `queue.txt` and `tmp/status.txt` — queue and its statuses.

## Directory key fields

- `datasets_info.json`: `classes`, `structure`, `elements_count`.
- Synchronization service fields: `dataset_hash`, `source_hash`, `source_ref`, `source_signature`, `modified`.
- Optional fields: `data_path`, `tags`, `roi_auto`.

## Supported dataset structures (official naming + internal IDs)

`datasets_info.json` stores internal structure IDs in the `structure` field.
These IDs are stable project contracts and are intentionally not renamed.

- Official name: **YOLO split directories layout**
  - Internal ID: `split`
  - Typical shape: `<dataset>/<train|val|test>/<images|labels>/...`
  - Notes: standard YOLO split-style dataset organization.

- Official name: **YOLO flat paired directories layout**
  - Internal ID: `flat`
  - Typical shape: `<dataset>/images/*` and `<dataset>/labels/*`
  - Notes: standard YOLO paired directories without explicit split folders.

- Official name: **YOLO flat with subset subfolders** (SmarTrain term)
  - Internal ID: `subset_flat`
  - Closest common format: YOLO split-style organization
  - Key difference: subset folder names are arbitrary and not restricted to `train/val/test`.

- Official name: **YOLO nested split under images/labels** (SmarTrain term)
  - Internal ID: `nested_split`
  - Closest common format: Ultralytics YOLO split directories layout
  - Key difference: split folders are nested as `images/<split>` and `labels/<split>`.

- Official name: **Darknet YOLO dataset layout**
  - Internal ID: `darknet`
  - Typical shape: `obj.data`, `obj.names`, `train.txt`/`valid.txt`, `obj_<subset>_data/`
  - Notes: legacy Darknet-style detection dataset packaging.

- Official name: **CVAT for images 1.1 layout**
  - Internal ID: `cvat11`
  - Typical shape: `annotations.xml` + `images/` (CVAT for images 1.1 export)
  - Notes: in SmarTrain this format is consumed through the internal `cvat11` structure identifier.

- Official name: **CvsDclDet detection export layout** (SmarTrain term)
  - Internal ID: `cvsdcldet`
  - Typical shape: flat folder with paired `*.jpg` (or other image) + `*.json`; JSON contains `detections[]` with `class_name`, `x`, `y`, `width`, `height` in pixels
  - Notes: convert to CVAT 1.1 with `smartrain dataset convert --source <path> --to cvat11` (folder or `.zip`/`.tar`/`.tar.gz` archive); output is usually placed under `converted_raw_data/`.

## Terminology policy

- Use **official format names** in user-facing documentation whenever possible (for example, "CVAT for images 1.1 layout", "Darknet YOLO dataset layout").
- Use **internal IDs** (`split`, `flat`, `subset_flat`, `nested_split`, `darknet`, `cvat11`, `cvsdcldet`) only when referring to code behavior, metadata contracts, or `datasets_info.json`.
- When a term is SmarTrain-specific (`subset_flat`, `nested_split`), always provide:
  - closest common format;
  - one-line key difference from that common format.
- Do not rename internal IDs in docs as if they were official external standards; treat them as stable project contract values.
- In mixed context, use both forms on first mention: `CVAT for images 1.1 (internal ID: cvat11)`.

## Annotations

Basic YOLO bbox format:

`class_id center_x center_y width height`

Segmentation polygons in `class_id x1 y1 x2 y2 ...` format are also supported.

Example polygon label file (`labels/img001.txt`):

```
0 0.10 0.20 0.90 0.20 0.90 0.80 0.10 0.80
1 0.30 0.30 0.50 0.30 0.50 0.50 0.30 0.50
```

Coordinates are normalized to `[0, 1]` relative to image width/height.

## Queue and statuses

- Queue file in workspace: `queue.txt` (one command per line).
- Status file: `tmp/status.txt`.
- Basic performer statuses: `Waiting to be completed`, `Running`, `Done`, `Error`.
