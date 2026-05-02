> Russian version: [../ru/cli/datasets.md](../ru/cli/datasets.md)

# CLI: datasets

## `scan`

Updates the dataset index and synchronizes sources in the workspace.

- Output files: `datasets_info.json`, `class_names.json`, `datasets_scan_summary.json`.
- Supports sources from `raw_data/`, `--dataset`, `--datasets-list`.
- Useful modes: `--mode refresh`, `--purge-processed-raw`.

## `normalize-data-yaml`

Rewrites every `datasets/**/data.yaml`: drops `path`, makes split paths relative. Foreign absolute paths from another machine are mapped to `train/images`, `val/images`, etc. when those folders exist under the same dataset root.

Example: `smartrain normalize-data-yaml --workspace /path/to/workspace` or `--datasets-dir ... --dry-run`.

## `fusion`

Collects a new dataset from several sources:

- selection of inputs: `--dataset` (repeatable) or `--datasets` (CSV);
- class management: `--classes`, `--exclude-classes`, `--merge-classes`, `--common-classes-only`;
- crash: `--fusion-split train,val,test`.

## `augment`, `balance`, `orient`, `roi`

- `augment` — autonomous augmentations with recording of a new dataset; **`--aug-class-aware-geo`** / **`--aug-total-bbox-cap-mult`** match `balance` hybrid-aug (same literature refs as above; standalone default for class-aware is **off** for backward compatibility); **`--aug-per-class-bbox-cap-mult`** limits **extra** bbox per class vs baseline train counts: class `c` can gain at most `ceil(mult × n_c) − n_c` bbox lines from augmented files (`n_c` = baseline train bbox count for that class); combines with the global cap when both are set;
- `balance` — class balancing; after balancing, `--eval-coverage` (default) can rebalance items across `train`/`val`/`test` so eval splits are non-empty when possible and rare classes appear in `val`/`test`; `--no-eval-coverage` turns this off;
  - class priority tuning: `--class-weight-multiplier "other:0.6,tear_up:1.1"` multiplies class weights after base weighting;
  - auto head-class dampening is enabled by default (`--auto-head-cap`): the tool computes recommended dampening multipliers for overrepresented classes from train statistics; disable via `--no-auto-head-cap`;
  - strategy **`hybrid-aug`**: same hybrid sampling as `hybrid`, then offline **`augment`** on the **train** split only. Augment presets: `--aug-preset geo-photo` (default: flip + photometric + center-rotate, anchor center) or `conveyor-lite` (adds conveyor noise). **`--aug-class-aware-geo`** (default **on**) lowers flip / photometric / conveyor rates on frequent-class frames so offline geo-photo does not amplify majority bbox mass (motivation: class-independent DA can worsen skew — **DODA**, [ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/54d2d38a56a74387d5916ee40e462295-Paper-Conference.pdf); per-class augmentation strength — **CUDA**, [arXiv:2302.05499](https://arxiv.org/abs/2302.05499)). **`--aug-total-bbox-cap-mult`** passes through to augment: optional hard cap so total train bbox count after augment stays ≤ `ceil(mult × baseline)` while keeping every baseline hybrid train frame (slack applies only to **extra** augmented images). **`--aug-per-class-bbox-cap-mult`** caps extra bbox **per class** vs baseline train counts (`ceil(mult × n_c) − n_c` per class). **`--aug-enable-bbox-copy`** turns on bbox copy-paste (off by default). The intermediate dataset folder named like `[output-name]_balanced_aug__hybrid` is deleted from disk and from `datasets_info.json` after a successful augment unless `--keep-hybrid-intermediate` is set; `balance_manifest.json` → **`post_augment`** records `class_aware_geo`, `total_bbox_cap_mult`, `per_class_bbox_cap_mult`, and train bbox sums before/after augment when emitted;
  - optional **head bbox undersampling** after sampling: `--train-head-bbox-undersample median-factor` with `--train-head-bbox-cap-mult` (default `5.0`) drops excess YOLO label lines for classes above `floor(cap_mult * median bbox count per class)` using stratified round-robin; see `balance_manifest.json` key `head_bbox_undersample` when used;
  - background: long-tailed learning taxonomy [arXiv:2110.04596](https://arxiv.org/abs/2110.04596), detection/long-tail surveys [arXiv:2408.00483](https://arxiv.org/abs/2408.00483); combining rebalancing with offline augmentation follows common practice on skewed benchmarks (e.g. COCO-ZIPF-style studies such as [arXiv:2403.07113](https://arxiv.org/abs/2403.07113)).
- `orient` — frame rotation correction;
- `roi` — crop according to the ROI-model.

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
