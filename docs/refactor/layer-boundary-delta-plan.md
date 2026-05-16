# Layer-boundary delta plan (LB-D1 … LB-D7)

**Status:** in progress  
**Baseline:** layer-boundary refactor closed 2026-05-16; register TD-LB-000…012 DONE.

## Execution order

1. **LB-D1** — Adapter hygiene (`analyze_runtime_api` → services)
2. **LB-D2** — `TrainRuntimeOps`; remove `get_training_module_api` from `train_service`
3. **LB-D3** — `train_public_api`, `train_cli_main`, shrink `model_training_module` facade
4. **LB-D4** — Dedupe `inference_cli` → `inference_runtime_helpers`
5. **LB-D5.3** — Remove no-op `emit_legacy_read_deprecation_warnings`
6. **LB-D5.2** — Rename mfel / `external_infer_fallback` → `external_eval_substitute`
7. **LB-D5.1** — `model_adapter` requires `task_type` + migration script
8. **LB-D6** — Split `format_runners.py` (separate commits)
9. **LB-D7** — Docs + register TD-LB-013…015

## LOC targets

| File | Target |
|------|--------|
| `workflows/training/model_training_module.py` | < 80 LOC (facade) |
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
