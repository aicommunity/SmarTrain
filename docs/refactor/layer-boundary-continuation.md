# Layer boundary refactor — continuation

**Audit date:** 2026-05-15  
**Baseline register:** [tech-debt-layer-boundaries.md](./tech-debt-layer-boundaries.md)

## Conformance summary

| # | Goal | Status | Evidence |
|---|------|--------|----------|
| G1 | Thin `workflows/` CLI | Partial | `results_analyzer.py` ~25 LOC facade; `cli_commands.py` still holds argparse + commands |
| G2 | Analyze in `services/analyze/` | Met | 30+ modules under `services/analyze/` |
| G3 | Unambiguous names | Met | `TaskTypeLabel`, `CanonicalIdentity`, `ultralytics_model_alias_registry` |
| G4 | No new path fallbacks | Met | `resolve_run_model` replaces legacy rglob search |
| G5 | Class-based extension | Partial | `AnalyzeCommandRegistry`, `CapabilityRegistry` / `BackendRegistry` alias |
| G6 | `services` ⊄ `workflows` | Met | Guardrails + allowlist for transitional imports |
| G7 | `orchestrators` ⊄ `workflows` | Met | Gateway uses `adapters.canonical.read.metrics_csv` |
| G8 | `backends` ⊄ `workflows` | Partial | Allowlist until ultralytics implementations move |
| G9 | Train/test not monoliths in workflows | Partial | Train `*_service` moved; MTM / `model_test_backends` still large |
| G10 | Canonical metrics read | Met | `metrics_csv` adapter |

**Score:** 5 Met, 5 Partial, 0 Not met (mandatory continuation below).

## Remaining gaps (ordered)

### P0 — blocks layer invariants

| id | Gap | Files | Done when |
|----|-----|-------|-----------|
| LB-C1 | `backends/*_adapter.py` import `workflows.inference` | `ultralytics_adapter.py`, `external_provider_adapter.py` | Implement `backends/implementations/ultralytics/`; empty guardrails allowlist |
| LB-C2 | `services/analyze/*` imports `workflows.testing` / `workflows.datasets` | `metrics_reader.py`, `report_writer.py`, `ultralytics_test_artifacts.py` | Move contracts to `core/` or `adapters/`; drop allowlist entries |

### P1 — reduces confusion / debt

| id | Gap | Files | Done when |
|----|-----|-------|-----------|
| LB-C3 | `model_training_module.py` still ~2000 LOC | `workflows/training/` | Extract runners to `services/training/`; `train_entry` → thin CLI |
| LB-C4 | `model_test_backends.py` ~3000 LOC | `workflows/testing/` | Split into `services/testing/backends/` |
| LB-C5 | `cli_commands.py` imports `workflows.analyze` facade | `services/analyze/cli_commands.py` | `services/analyze/prompts.py` bridge; remove allowlist |
| LB-C6 | `report_writer.py` ~3400 LOC | `services/analyze/` | Internal split (sections/builders); separate TD |

### P2 — nice to have

| id | Gap | Notes |
|----|-----|-------|
| LB-C7 | Wave 7 datasets → `services/datasets/` | Defer until dataset CLI work |
| LB-C8 | `import-linter` in CI | Enforce layers mechanically |

## Proposed next waves

| Wave | Scope | Est. commits |
|------|-------|--------------|
| LB-C1 | Ultralytics backend implementations | 2 |
| LB-C2 | Decouple analyze from workflows.testing/datasets | 2–3 |
| LB-C3–C4 | Thin train/test execution modules | 3–4 |

## Out of scope (WONTFIX)

- **Wave 7 datasets** — no non-CLI consumer yet; keep CLI in `workflows/datasets/`.
- **`conf_rec_fallback` / CPU val fallback** — product parameters, not path legacy.
- **External mfel eval substitute rename** — breaking JSON metadata keys; document only unless version bump.

## Links

- [03-target-architecture.md](./03-target-architecture.md)
- [package-layout.md](../development/package-layout.md)
