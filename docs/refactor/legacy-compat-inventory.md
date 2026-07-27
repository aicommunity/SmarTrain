# Legacy compatibility inventory

Map of on-disk and runtime backward-compatibility layers. Target canonical workspace layout is documented in [`run-layout.md`](./run-layout.md). Migration UX: `smartrain update` (see [`../cli/overview.md`](../cli/overview.md)).

**Removal gate:** do not delete a read-fallback until `smartrain update --check` reports zero residual findings for that category on shared workspaces, and release notes cover the break. Phased removal plan: [`legacy-fallback-removal.md`](./legacy-fallback-removal.md).

## Canonical target (workspace)

```text
datasets/<key>/data.yaml              # relative splits, no path:
runs/<dataset>/<run_id>/
  models/<weight_stem>.{pt,onnx,...}
  train-ultralytics/
  tests/…
  tmp/_runtime_data_*.yaml
  training_metadata.json              # paths.best_model = basename
models/<dataset>/<run_id>/            # R3 unified (= run tree + models/<stem>.json)
models/releases_manifest.json         # keys <dataset>/<weight_stem>
```

## A. Run layout and weights

| Symbol | File | Legacy | Canonical | Type | Auto in `update`? |
|--------|------|--------|-----------|------|-------------------|
| `resolve_run_model` / `_resolve_run_model_existing` | `smartrain/core/runtime/run_artifacts.py` | R3 root PT, R2 sibling, `<run>.pt`, `train*/weights/{best,last}.pt` | `models/<stem>.ext` | read | ask if multi-PT |
| `resolve_run_weights_stem` | same | folder-named PT, sole PT | metadata `detect_*…` | read | ask |
| `materialize_preferred_run_model` | same | move/copy to preferred | preferred + metadata | migrate | yes |
| `ensure_run_layout` / `_migrate_legacy_*` | same | `train/`, root `test*`, root metrics/YAML/`val-recs-*` | `train-ultralytics/`, `tests/`, `tmp/` | migrate | yes\* |
| `_looks_like_release_bundle_dir` | same | heuristics | explicit sidecar/manifest | guard | n/a |
| `_normalize_run_root` | same | path into `models/`/`tmp/`/`tests/` | bundle root | read | yes |

\*If canonical already exists and hash differs from legacy, treat as **ask** (do not silently delete).

## B. Release catalog / manifests

| Symbol | File | Legacy | Canonical | Auto? |
|--------|------|--------|-----------|-------|
| `release_dir_for_pt`, `find_release_pt_in_dir` | `smartrain/services/models/release_models_manifest.py` | R1, R2, R3 root, nested `models/*.pt` | R3 unified | ask on conflict |
| `get_comment_for_run_dir` | same | folder key, sidecar scan | `<dataset>/<weight_stem>` | yes (rewrite keys) |
| `load_manifest` | same | broken JSON → empty | fail / ask restore | ask |
| unrelease strip overlay | `smartrain/workflows/models/model_unrelease_cli.py` | root PT/onnx → `models/` | unified | yes |
| rename nested siblings | `smartrain/services/models/release_model_rename_service.py` | must rename next to PT | `pt.parent` | n/a (fixed) |
| absolute `artifacts.*` / `source_run` | release sidecar | abs paths | workspace-relative | yes |

## C. Test / train artifacts

| Symbol | File | Legacy | Canonical | Auto? |
|--------|------|--------|-----------|-------|
| `format_test_dir` et al. | `smartrain/core/testing/artifact_paths.py` | root `test*`, root metrics | `tests/…` | yes |
| `iter_ultralytics_artifact_source_dirs` | `smartrain/core/testing/ultralytics_artifact_resolver.py` | root `test/`, train-val | `tests/test-ultralytics` | yes / ask |
| `latest_test_metrics_path` | `smartrain/core/analyze/run_metrics_discovery.py` | root CSV | `tests/test_metrics*.csv` | yes |
| `results_csv_path` | `smartrain/services/analyze/metrics_reader.py` | `train/`, `train-ultralytics-*` | `train-ultralytics/` | yes |

## D. Metadata / datasets / YAML

| Area | File | Legacy | Canonical | Auto? |
|------|------|--------|-----------|-------|
| training_metadata paths | train metadata IO / `normalize_model_references_in_metadata` | `path_absolute`, `train/weights/best.pt` | `path_under_workspace`, basename | ask / yes |
| `normalize_data_yaml_mapping` | `smartrain/services/datasets/data_yaml_normalize.py` | `path:`, abs splits | relative, no `path` | yes / ask |
| data.yaml candidates | `smartrain/services/analyze/data_yaml.py` | multi sources | tmp + metadata | ask if >1 |
| runtime split fallbacks | `train_runtime_data_yaml_service.py` | val←train, test←val | explicit splits | **not in update** |
| datasets_info / passport | workspace_paths / passport | abs / external `data_path` | catalog-relative | ask |
| registry `model_manifest.json` | registry CLI | abs provenance | relative + task_type | yes / ask |

## E. Snapshots / env / naming (inventory only; not `update` apply v1)

- `.smartrain/canonical/` → `.smartrain/unified/`
- Env `SMARTTRAIN_CANONICAL_WRITE` → `SMARTTRAIN_UNIFIED_WRITE`
- Task/stem aliases: `det/cls/seg`, short `task_model_YYYYMMDD_HHMMSS`, format `best`→`pt`
- Task inference from stem (detection fallback)

Note: docs that mention `SMARTTRAIN_CANONICAL_READ` / `SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK` are outdated — those flags are not used in current runtime.

## F. CLI aliases (outside `update` v1)

| Old | New | Warning today |
|-----|-----|---------------|
| `fusion` | `merge` | yes |
| `plot` | `analyze` | no (docs only) |
| `queue-run` | `queue run` | no |
| `migrate canonical` | `migrate unified` | hidden alias |

Related tools (delegates / companions of `update`): `migrate unified`, `migrate-models`, `normalize-data-yaml`, `scripts/migrate_model_task_provenance.py`.

## See also

- [`legacy-fallback-removal.md`](./legacy-fallback-removal.md) — phased deletion after residual=0
- [`06-deprecation-and-alias-policy.md`](./06-deprecation-and-alias-policy.md)
- [`run-layout.md`](./run-layout.md)
