> Russian version: [../ru/providers/mfel-yolo.md](../ru/providers/mfel-yolo.md)

# MFEL-YOLO provider

## Subsystem

- Provider id: `mfel-yolo`
- Runtime family: MFEL fork with custom blocks on top of Ultralytics API
- Default aliases exposed by Smart Train:
  - `mfel-yolo` (preferred default)
  - `e_pan+` (when present in provider config directory)

## Wrapper implementation

- Train adapter:
  - `external_providers/adapters.py` -> `launchers/mfel_train_launcher.py`
- Inference adapter:
  - `external_providers/adapters.py` -> `launchers/mfel_infer_launcher.py`
- Validation fallback launcher (used when built-in test cannot import custom blocks):
  - `external_providers/launchers/mfel_val_launcher.py`

## MFEL-specific handling

- Custom symbol patching for missing fork symbols (for compatibility and checkpoint load paths).
- Training defaults tuned for stability in launcher:
  - `amp=False`, `optimizer=AdamW`, `pretrained=False`, tuned LR/warmup/weight decay.
- `mfel_val_launcher.py` writes `results.csv` for deterministic `test_metrics.csv` generation.

## Notes

- Provider alias validation is strict.
- Run artifacts are normalized to the common external-provider contract.
