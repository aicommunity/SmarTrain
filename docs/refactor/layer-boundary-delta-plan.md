# Layer-boundary delta plan (LB-D1 … LB-D8)

**Status:** D1–D7 done; **LB-D8 (MTM liquidation)** done  
**Baseline:** layer-boundary refactor closed 2026-05-16; register TD-LB-000…012 DONE.

## Execution order

1. **LB-D1** — Adapter hygiene (`analyze_runtime_api` → services) — **done**
2. **LB-D2** — `TrainRuntimeOps`; remove `get_training_module_api` from `train_service` — **done**
3. **LB-D3** — `train_cli_main`, `train_wiring` — **done** (superseded by LB-D8 MTM removal)
4. **LB-D4** — Dedupe `inference_cli` → `inference_runtime_helpers` — **done** (~235 LOC CLI)
5. **LB-D5.3** — Remove no-op `emit_legacy_read_deprecation_warnings` — **done**
6. **LB-D5.2** — Rename mfel / `external_infer_fallback` → `external_eval_substitute` — **done**
7. **LB-D5.1** — `model_adapter` requires `task_type` + migration script — **done** (`scripts/migrate_model_task_provenance.py`)
8. **LB-D6** — Split `format_runners.py` — **done** (facade ~50 LOC; `format_runners_support` / `_ultralytics` / `_native`)
9. **LB-D7** — Docs + register TD-LB-013…015 — **done**
10. **LB-D8** — Remove `model_training_module.py`; `train_entry` + `train_wiring` + `train_cli_callbacks` — **done** (2026-05-16)

## LOC targets

| File | Target |
|------|--------|
| ~~`workflows/training/model_training_module.py`~~ | removed (LB-D8) |
| `workflows/inference/inference_cli.py` | < 200 LOC |
| `services/testing/backends/format_runners.py` | facade < 100 LOC after D6 |

## Verification (each phase)

```bash
pytest tests/regression/test_layer_import_guardrails.py -q
pytest tests/test_train_*.py tests/test_training_metadata*.py -q  # after D2/D3
pytest tests/ -k inference -q  # after D4
pytest tests/adapters/canonical/read/ -q  # after D5.1
pytest tests/ -k "model_test or test_backend" -q  # after D6
pytest tests/ -q  # final
```
