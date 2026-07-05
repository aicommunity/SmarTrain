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

## `fusion`

Collects a new dataset from several sources:

- selection of inputs: `--dataset` (repeatable) or `--datasets` (CSV);
- class management: `--classes`, `--exclude-classes`, `--merge-classes`, `--common-classes-only`;
- split: `--fusion-split train,val,test`;
- **`--strip-unused-classes`**: after merge, drop output classes with zero instances (remap `class_id` in `.txt`).

## `split`

Repartitions one existing dataset into `train`/`valid`/`test` without merging sources:

```bash
smartrain split --dataset my_dataset --split-ratio 0.7,0.2,0.1
smartrain split --dataset my_dataset --exclude-test --output-name my_dataset_resplit
```

- `--split-ratio train,val,test` — random repartition within each input bucket (default `0.8,0.1,0.1`; same algorithm as `fusion --fusion-split`);
- `--exclude-test` — skip test buckets from the source input;
- writes a new dataset under `datasets/` with `data.yaml` and updates `datasets_info.json`.

## `prune`

```bash
smartrain prune empty --dataset my_dataset
smartrain prune dedup --dataset my_dataset
smartrain prune classes --dataset my_dataset
```

- **`prune empty`** — removes empty image/label pairs into `<dataset>_pruned`.
- **`prune dedup`** — removes duplicate images by content into `<dataset>_deduped`.
- **`prune classes`** — copies the dataset to `<dataset>_classes_pruned`, removes unused classes from metadata (`data.yaml`, `obj.names`), remaps `class_id` in annotations; image and label files are kept.

## `filter`

Filter edge-truncated YOLO bbox annotations into a new dataset (default output `<dataset>_fltd`):

```bash
smartrain filter --dataset my_dataset
smartrain filter --dataset my_dataset --stats-only
smartrain filter --dataset my_dataset --dry-run
smartrain filter --dataset my_dataset --drop-images
```

- **Pass 1** — per-class baseline width/height stats from bbox **fully inside** an inset zone (`--baseline-inset-margin`, default `0.01`; optional `--baseline-inset-margin-px`).
- **Pass 2** — drop near-edge bbox that are too small (absolute `--abs-min-width-px` / `--abs-min-height-px` and relative to class p-quantile `--rel-quantile` with `--rel-width-factor` / `--rel-height-factor`). Near-edge zone: `--filter-proximity-margin` (defaults to baseline inset margin); strict touch/OOB: `--edge-eps`. Limit affected edges with `--edge-sides` (`any`, `horizontal`, `vertical`, `up`, `down`, `left`, `right`; default `any`).
- Optional: `--min-visibility`, `--min-area-px`, `--max-aspect-ratio`; `--classes` to limit affected classes.
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

**Supported today:** `rotate`, `orient`, `roi --mode yolo_segment`, `report` (GT polygon preview), `fusion` (class filter pass-through).

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
  - background: long-tailed learning taxonomy [arXiv:2110.04596](https://arxiv.org/abs/2110.04596), detection/long-tail surveys [arXiv:2408.00483](https://arxiv.org/abs/2408.00483); combining rebalancing with offline augmentation follows common practice on skewed benchmarks (e.g. COCO-ZIPF-style studies such as [arXiv:2403.07113](https://arxiv.org/abs/2403.07113)).
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
