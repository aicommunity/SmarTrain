# Tech debt: ultralytics test restoration

## Register

| id | status | context | deferred_decision | resolution |
|----|--------|---------|-------------------|------------|
| TD-001 | DONE | `model_training_module` post-train / `train_service` | Post-training `test_yolo` still uses `plots=False`; full plot bundle is produced by `smartrain test` and resume path. | Documented: intentional; train keeps fast smoke test. |

## Burn-down checklist

- [x] TD-001
- [x] No remaining `OPEN` rows in this register after final pass (2026-05-14).
