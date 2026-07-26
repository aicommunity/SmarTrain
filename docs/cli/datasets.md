> Russian version: [../ru/cli/datasets.md](../ru/cli/datasets.md)

# CLI: datasets

## `scan`

Updates the dataset index and synchronizes sources in the workspace.

- Output files: `datasets_info.json`, `class_names.json`, `datasets_scan_summary.json`.
- Supports sources from `raw_data/`, `--dataset`, `--datasets-list`.
- Useful modes: `--mode refresh`, `--purge-processed-raw`.
- **`--strip-unused-classes`** (default **on**): for **newly added** datasets, after copy to `datasets/` and format normalization (e.g. CVAT 1.1 → YOLO), removes class names with zero label instances from `data.yaml` / `obj.names` and remaps annotation `class_id` values. Disable with `--no-strip-unused-classes`. Supports all scan structure IDs (`split`, `flat`, `darknet`, `cvat11`, `cvsdcldet`, …).

## `normalize-data-yaml`

Rewrites every `datasets/**/data.yaml`: drops `path`, makes split paths relative. Foreign absolute paths from another machine are mapped to `train/images`, `val/images`, etc. when those folders exist under the same dataset root.

Example: `smartrain normalize-data-yaml --workspace /path/to/workspace` or `--datasets-dir ... --dry-run`.

## `merge` (alias: `fusion`)

Collects a new dataset from several sources:

- selection of inputs: `--dataset` (repeatable) or `--datasets` (CSV);
- class management: `--classes`, `--exclude-classes`, `--merge-classes`, `--common-classes-only`;
- split: `--fusion-split train,val,test`;
- **`--strip-unused-classes`**: after merge, drop output classes with zero instances (remap `class_id` in `.txt`).
- Interactive mode (`smartrain merge` from TTY) supports merge setup: it asks for class list and lets you add `sources -> target` rules that map to repeatable `--merge-classes` flags in replay.

Example non-interactive equivalent:

```bash
smartrain merge --dataset ds_a --dataset ds_b --classes "class_ab,other" --merge-classes "class_a,class_b" class_ab
```

## `split`

Repartitions one existing dataset into `train`/`valid`/`test` without merging sources:

```bash
smartrain split --dataset my_dataset --split-ratio 0.7,0.2,0.1
smartrain split --dataset my_dataset --exclude-test --output-name my_dataset_resplit
```

- `--split-ratio train,val,test` — random repartition within each input bucket (default `0.8,0.1,0.1`; same algorithm as `merge --fusion-split`);
- `--exclude-test` — skip test buckets from the source input;
- writes a new dataset under `datasets/` with `data.yaml` and updates `datasets_info.json`.

## `prune`

```bash
smartrain prune empty --dataset my_dataset
smartrain prune dedup --dataset my_dataset
smartrain prune classes --dataset my_dataset
smartrain prune size --dataset my_dataset
smartrain prune size --dataset my_dataset --min-size 12x18
smartrain prune size --dataset my_dataset --size-mode and
smartrain prune size --dataset my_dataset --no-drop-empty-images
```

- **`prune empty`** — removes empty image/label pairs into `<dataset>_pruned`.
- **`prune dedup`** — removes duplicate images by content into `<dataset>_deduped`.
- **`prune classes`** — copies the dataset to `<dataset>_classes_pruned`, removes unused classes from metadata (`data.yaml`, `obj.names`), remaps `class_id` in annotations; image and label files are kept.
- **`prune size`** — copies the dataset to `<dataset>_size_pruned`, removes label instances where bbox size is below `NxM` pixels (`--min-size`, default `20x20`). Drop rule: `--size-mode or` (default) removes when **either** side is below threshold; `--size-mode and` removes only when **both** sides are below threshold. Then by default removes images where no labels remain. Use `--no-drop-empty-images` to keep such images with an empty label file.

## `filter`

Filter YOLO bbox annotations into a new dataset (default output `<dataset>_fltd`). Two independent modes (can be combined; disable either with `--no-edge-filter` / `--no-size-filter`):

```bash
smartrain filter --dataset my_dataset
smartrain filter --dataset my_dataset --stats-only
smartrain filter --dataset my_dataset --dry-run
smartrain filter --dataset my_dataset --drop-images
smartrain filter --dataset my_dataset --no-edge-filter --size-filter --classes startup_marker
smartrain filter --dataset my_dataset --size-filter --size-dims width --drop-images
smartrain filter --dataset my_dataset --no-edge-filter --size-filter --size-dims width --size-baseline-mode stable --drop-images
```

- **Pass 1** — per-class baseline width/height stats from bbox **fully inside** an inset zone (`--baseline-inset-margin`, default `0.01`; optional `--baseline-inset-margin-px`). Used by both edge and size filters.
- **Edge filter** (`--edge-filter`, default on) — drop near-edge bbox that are too small (absolute `--abs-min-width-px` / `--abs-min-height-px` and relative to class p-quantile `--rel-quantile` with `--rel-width-factor` / `--rel-height-factor`). Near-edge zone: `--filter-proximity-margin` (defaults to baseline inset margin); strict touch/OOB: `--edge-eps`. Limit affected edges with `--edge-sides` (`any`, `horizontal`, `vertical`, `up`, `down`, `left`, `right`; default `any`). With **`--empirical-bounds`** (default off): bbox touching the physical image border use image edges; inset bbox use a per-class percentile hull (`--empirical-percentile`, default `0.10` → p10–p90) built from inset-only samples (`--empirical-inset-only`, default on). Optional per-resolution hulls: `--empirical-by-format` (default on).
- **Size filter** (`--size-filter`, default off) — drop bbox **smaller than the class baseline** anywhere in the frame. Control checked dimensions with `--size-dims` (`any`, `width`, `height`; default `any`). `--classes` limits which classes are filtered; other classes pass through unchanged.
  - **`--size-baseline-mode inset`** (default) — typical size from inset-only samples; threshold = `--rel-width-factor` × quantile(`--rel-quantile`) of inset widths/heights (same as edge relative thresholds).
  - **`--size-baseline-mode stable`** — for classes with stable object size: bulk-trim over **all** samples (`--size-bulk-split-ratio` × median, default 0.5), typical = quantile within bulk (`--size-typical-quantile`, default 0.25), threshold = `--rel-width-factor` × typical. Optional per-resolution baselines: `--size-by-format`.
- Optional: `--min-visibility`, `--min-area-px`, `--max-aspect-ratio` (edge filter only).
- **`--drop-images`** — remove entire image+labels from train/val/test buckets; originals are archived under `_filter_audit/dropped_images/<split>/images|labels` (excluded from `data.yaml`, training, and stats).
- **`--prune-empty`** (default on) — after filtering, remove pairs that **had annotations** but none remain; archived to `_filter_audit/dropped_images/`.
- **`--drop-background`** (default off) — remove source images that never had annotations (no label file or empty label); archived to `_filter_audit/dropped_images/` when enabled.
- Partial label removal (image kept) — dropped bbox lines are written to `_filter_audit/removed_labels/<split>/labels/` with the same relative paths as in the main dataset.
- Audit paths and counts are recorded in `filter_manifest.json` → `stats_after.audit`.
- **`--stats-only`** / **`--dry-run`** — preview without writing output dataset; interactive mode (`smartrain filter` from TTY) shows preview table and replay command.
- Writes `filter_manifest.json` and `dataset_passport.json` in the output dataset.

## `augment`, `balance`, `orient`, `rotate`, `roi`

### Instance segmentation (YOLO polygons)

Labels use `class_id x1 y1 x2 y2 ...` (normalized polygon vertices). See [data formats](../reference/data-formats.md).

**Supported today:** `rotate`, `orient`, `roi --mode yolo_segment`, `report` (GT polygon preview), `merge` (class filter pass-through; `fusion` remains alias).

**Limitations (see also [tech debt register](../refactor/tech-debt-instance-segmentation.md)):**

- `augment` — geometric modes require polygon-aware labels; use `--label-type segment` when your dataset has polygons. Bbox copy-paste (`--enable-bbox-copy`) is **not** supported for polygon datasets.
- `balance` — head undersampling uses the **enclosing bbox** of each polygon (approximation).
- CVAT import/export — polygon support is being extended; bbox-only CVAT tasks remain bbox-only.
- Native ONNX/engine/TRT **model test** for segmentation is skipped by default; use PT test (`smartrain test --formats pt --task segment`).

**ROI example:**

```bash
smartrain roi --dataset my_seg --mode yolo_segment --weights yolo11s-seg.pt
```

- `augment` — autonomous augmentations with recording of a new dataset; flip sampling **`--flip-sampling`** (`probabilistic`/`exhaustive`); optional orthogonal ±90° **`--enable-orthogonal-rotate`**; conveyor effects can be toggled individually: **`--enable-conveyor-rotate`**, **`--enable-conveyor-scale`**, **`--enable-conveyor-blur`**, **`--enable-conveyor-shift`**, **`--enable-conveyor-noise`** with **`--conveyor-noise-types`**, **`--conveyor-noise-intensity`**, **`--conveyor-noise-selection`** (umbrella **`--enable-conveyor`** enables all five); interactive mode asks for each effect separately (noise defaults to off); **`--aug-class-aware-geo`** / **`--aug-total-bbox-cap-mult`** match `balance` hybrid-aug (same literature refs as above; standalone default for class-aware is **off** for backward compatibility). With a bbox cap, **`--aug-budget-tail-first`** (default **on**) processes train frames in descending tail priority `max_c (n_max/n_c)^γ` before spending slack on head-like frames; **`--aug-budget-tail-gamma`** (default `1.0`) sets γ; disable priority ordering with **`--no-aug-budget-tail-first`**;
- `balance` — class balancing; after balancing, `--eval-coverage` (default) can rebalance items across `train`/`val`/`test` so eval splits are non-empty when possible and rare classes appear in `val`/`test`; `--no-eval-coverage` turns this off;
  - class priority tuning: `--class-weight-multiplier "other:0.6,tear_up:1.1"` multiplies class weights after base weighting;
  - auto head-class dampening is enabled by default (`--auto-head-cap`): the tool computes recommended dampening multipliers for overrepresented classes from train statistics; disable via `--no-auto-head-cap`;
  - strategy **`hybrid-aug`**: same hybrid sampling as `hybrid`, then offline **`augment`** on the **train** split only. Augment presets: `--aug-preset geo-photo` (default: flip + photometric + center-rotate, anchor center) or `conveyor-lite` (adds all conveyor effects via `--enable-conveyor`). **`--aug-class-aware-geo`** (default **on**) lowers flip / photometric / conveyor rates on frequent-class frames so offline geo-photo does not amplify majority bbox mass (motivation: class-independent DA can worsen skew — **DODA**, [ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/54d2d38a56a74387d5916ee40e462295-Paper-Conference.pdf); per-class augmentation strength — **CUDA**, [arXiv:2302.05499](https://arxiv.org/abs/2302.05499)). **By default**, hybrid-aug uses constrained-growth tail-first mode: `--aug-total-bbox-cap-mult 1.10`, `--aug-budget-tail-first`, `--aug-budget-tail-gamma 1.0`, plus `--train-head-bbox-undersample median-factor --train-head-bbox-cap-mult 5.0`; eval splits also get conservative head trimming defaults `--eval-head-bbox-undersample median-factor --eval-head-bbox-cap-mult 8.0 --eval-head-bbox-min-count 30 --eval-head-bbox-max-remove-frac 0.35` (all can be overridden explicitly). **`--aug-total-bbox-cap-mult`** passes through to augment: optional hard cap so total train bbox count after augment stays ≤ `ceil(mult × baseline)` while keeping every baseline hybrid train frame (slack applies only to **extra** augmented images). With cap **> 0**, **`--aug-budget-tail-first`** / **`--aug-budget-tail-gamma`** are forwarded so scarce classes consume slack first (MVP: reorder train images only). **`--aug-enable-bbox-copy`** turns on bbox copy-paste (off by default). The intermediate dataset folder named like `[output-name]_balanced_aug__hybrid` is deleted from disk and from `datasets_info.json` after a successful augment unless `--keep-hybrid-intermediate` is set; `balance_manifest.json` records train/eval head-trimming settings and `post_augment` fields (`class_aware_geo`, `total_bbox_cap_mult`, `budget_tail_first`, `budget_tail_gamma`, train bbox sums before/after augment) when emitted;
  - optional **head bbox undersampling** after sampling: `--train-head-bbox-undersample median-factor` with `--train-head-bbox-cap-mult` (default `5.0`) drops excess YOLO label lines for classes above `floor(cap_mult * median bbox count per class)` using stratified round-robin; see `balance_manifest.json` key `head_bbox_undersample` when used;
  - background: Repeat Factor Sampling as in LVIS [arXiv:1908.03195](https://arxiv.org/abs/1908.03195); Class-Balanced effective-number weights [arXiv:1901.05555](https://arxiv.org/abs/1901.05555); long-tailed learning taxonomy [arXiv:2110.04596](https://arxiv.org/abs/2110.04596), detection/long-tail surveys [arXiv:2408.00483](https://arxiv.org/abs/2408.00483); combining rebalancing with offline augmentation follows common practice on skewed benchmarks (e.g. COCO-ZIPF-style studies such as [arXiv:2403.07113](https://arxiv.org/abs/2403.07113)).
- `orient` — frame rotation correction;
- `rotate` — fixed clockwise rotation of the whole dataset by `90`, `180`, or `270` degrees into `datasets/<name>_rot<angle>` (interactive by default);
- `roi` — crop according to the ROI-model.

#### `augment`: variants per frame

Each source frame in `--splits` (default `train`) always gets one **original copy**. Extra files are independent per augmentation type (not all combinations):

| Type | Default | Variants per frame | Randomness |
|------|---------|-------------------|------------|
| Flip (`--enable-flip`) | off | 0–1 (`probabilistic`) or all for `--flip` (`exhaustive`) | `--flip-prob`; `--flip-sampling probabilistic\|exhaustive` |
| Orthogonal ±90° (`--enable-orthogonal-rotate`) | off | 0–1 or both directions (`exhaustive`) | `--orthogonal-rotate-prob`, `--orthogonal-rotate-direction` |
| Photometric / conveyor | off | 0–1 combined file | class-aware geo optional |
| Center-rotate | **on** | up to `--rotate-copies` | random angle ±`--center-rotate-deg` |
| Bbox copy | off | up to `--bbox-copy-copies` | deterministic per seed |

Flip modes: `horizontal`, `vertical`, `both` (single H+V pass), `h-and-v` (two separate files). Unlike `smartrain rotate`, orthogonal augment adds optional per-frame ±90° variants only on selected splits.

Conveyor noise (`--enable-conveyor-noise`): types via **`--conveyor-noise-types`** (`gaussian`, `iso`, `shot`, `poisson-gaussian`, `multiplicative`, `impulse`; default `iso,shot,gaussian`), strength **`--conveyor-noise-intensity`** [0..1], selection **`--conveyor-noise-selection`** `random` (one type) or `stack` (all types).

All of the above commands form `dataset_passport.json` in the new dataset directory.

## `dataset convert`

Convert datasets between supported formats (CVAT for images 1.1, YOLO, CvsDclDet). Sources can come from the workspace catalog (`datasets/`), `raw_data/` (folders and archives), paths listed in `raw_data/datasets_list.txt`, or explicit external paths.

```bash
smartrain dataset convert
smartrain dataset convert --source /path/to/export.zip --to yolo --output-dir datasets/task_yolo
smartrain dataset convert --source datasets/task_yolo --to cvat11 --output-dir converted_raw_data/task --zip
smartrain dataset convert --source raw_data/my_det --to cvat11 --output-dir converted_raw_data/my_det
smartrain dataset convert --source raw_data/StartMarker14_PU50.zip --to cvat11 --output-dir converted_raw_data/StartMarker14_PU50 --zip
smartrain dataset convert --source /data/external/dataset.tar.gz --to cvat11 --output-dir converted_raw_data/external
smartrain dataset convert --source raw_data/my_det --to cvat11 --rename-classes white_line line --zip
smartrain dataset convert --dataset my_dataset --to cvat11 --output-dir converted_raw_data/my_dataset --zip
```

- **Interactive mode** (`smartrain dataset convert` from TTY): unified menu with `[datasets]`, `[raw_data]` (folders and `.zip`/`.tar`/`.tar.gz` archives), `[external]` from `datasets_list.txt`, and manual path entry; then detected format, target (`yolo`, `cvat11`), output path, optional CvsDclDet class rename, optional zip (default **off**) and folder deletion after zip (default **on** when zip is enabled).
- **`--source`**: directory or archive (`.zip`, `.tar`, `.tar.gz`, `.tgz`). Archives are extracted to a cache directory (`tmp/extracted_datasets/` under workspace or current directory), then structure is detected. `--source-dir` is a backward-compatible alias.
- **`--to`**: `yolo` (flat `images/` + `labels/` + `data.yaml`), `cvat11` (folder `annotations.xml` + `images/`).
- **`--zip` / `--no-zip`**: after folder output, optionally pack to zip (`.cvat11.zip` for CVAT folders, `.zip` for YOLO).
- **`--delete-after-zip` / `--no-delete-after-zip`**: remove output folder after zip (default: delete when `--zip`).
- Writes `dataset_passport.json` in folder outputs. Run `smartrain scan` separately to update `datasets_info.json`.

Migration from removed `smartrain cvat`:

| Old | New |
|-----|-----|
| `cvat import --cvat-zip X --output-dir Y` | `dataset convert --source X --to yolo --output-dir Y` |
| `cvat export --dataset-dir D --zip-path Z` | `dataset convert --source-dir D --to cvat11 --output-dir Z --zip` |
| `cvat from-cvsdcldet --source-dir S --output-dir O --zip` | `dataset convert --source S --to cvat11 --output-dir O --zip` |

## `dataset report`

Multilingual per-class sample report (Markdown + PNG; optional PDF/ODT):

```bash
smartrain dataset report --dataset my_dataset
smartrain dataset report --dataset my_dataset -n 6 --languages en,ru
smartrain dataset report
```

- Default output folder: `analytics/datasets-reports/<dataset>_<timestamp>/`.
- Interactive mode (`smartrain dataset report` from TTY): pick dataset and options, get a replay command.
- PDF/ODT via pandoc, WeasyPrint, `fpdf2`, or `odfpy` (see overview for dependency notes).

## `dataset rename`

Rename a dataset catalog entry and propagate references across the workspace:

```bash
smartrain dataset rename --dataset old_name --new-name new_name
smartrain dataset rename --dataset old_name --new-name new_name --dry-run
smartrain dataset rename
```

- Renames `datasets/<old>/` to `datasets/<new>/` and moves the key in `datasets_info.json`.
- Also renames `runs/<old>/` and `models/<old>/` when present.
- Updates references in `dataset_passport.json`, `training_metadata.json`, `args.yaml`, `queue.txt`, and analytics artifacts under `analytics/`.
- **`--dry-run`**: print the rename plan without applying changes.
- **`--move-data-path`**: required when the dataset uses a custom `data_path` outside `datasets/<name>/`.
- Interactive mode (`smartrain dataset rename` from TTY): pick a dataset from the catalog, enter a new name, preview the plan, and get a replay command.
- Does not modify `raw_data/` sources or `class_names.json`.

## `hash`

Checking and calculating the hash of the dataset structure:

```bash
smartrain hash --dataset my_dataset
smartrain hash /abs/path/to/dataset --validate a1b2c3d4
```

`--validate` exit codes: `0` match, `1` mismatch, `2` error.

## `stats`

Current behavior:

- `smartrain stats` launches the unified statistics mode (datasets + classes in one run).
- `smartrain stats compare` runs a dedicated two-dataset comparison mode.
- Legacy forms `smartrain stats classes` and `smartrain stats datasets` are accepted for compatibility and map to the same unified stats mode.
