# Legacy fallback removal phases

Prerequisite: workspace can be brought to canonical shape with `smartrain update`. Use `smartrain update --check` (exit non-zero if residual findings) as a gate before deleting runtime readers.

Follow [`06-deprecation-and-alias-policy.md`](./06-deprecation-and-alias-policy.md): migration tooling exists, usage checks green, release notes published.

## Phases

### P0 — blockers before mass migration

- Unified release rename must keep siblings next to the PT (`models/`), not in the bundle root (fixed in rename service).
- Unify release-bundle detection (`_looks_like_release_bundle_dir` vs `is_workspace_release_bundle`) toward one marker: valid release sidecar + optional `releases_manifest` entry.

### P1 — after layout migrate residual = 0

Remove or narrow read preference for:

- root `test/`, `test_*`, root metrics CSVs / recommendations / `test_artifacts_manifest.json`
- legacy `train/` (keep only if migrate never completed)

Files: `artifact_paths.py`, `ultralytics_artifact_resolver.py`, `run_metrics_discovery.py`, parts of `ensure_run_layout` migrate paths (keep migrate code behind `update` only if needed).

### P2 — after release unify residual = 0

Remove resolvers for:

- R3 root-level `detect_*.pt`
- R1 / R2 sibling layouts in `resolve_run_model` / release helpers

Keep R3 unified (`…/<run_id>/models/<stem>.pt`) only.

### P3 — manifest comment fallbacks

Remove folder-name key fallback `<dataset>/<release_folder>` once all keys are `<dataset>/<weight_stem>`.

### P4 — CLI / env aliases

- Add deprecation warnings where missing (`plot`, `queue-run`)
- Remove after one release cycle: `fusion`, `plot`, `queue-run`, `migrate canonical`, `SMARTTRAIN_CANONICAL_WRITE`

### Out of scope for automatic deletion

- Runtime split semantic fallbacks (val←train / test←val)
- Hardcoded provider / Ultralytics alias lists (install-time fallbacks)
- Registry bundles (`model_manifest.json`) — separate contract

## Check command

```bash
smartrain update --check
smartrain update --check --only layout,releases,tests
```

Exit `0` only when the scanner finds no residual legacy items in the selected categories.
