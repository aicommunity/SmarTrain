> Russian version: [../ru/providers/mp-yolo.md](../ru/providers/mp-yolo.md)

# MP-YOLO provider

## Subsystem

- Provider id: `mp-yolo`
- Runtime family used by integration: Ultralytics-compatible YOLOv8 API in provider `venv`
- Default aliases exposed by Smart Train: `yolov8n/s/m/l/x`

## Wrapper implementation

- Train adapter:
  - `external_providers/adapters.py` -> `launchers/mp_train_launcher.py`
- Inference adapter:
  - `external_providers/adapters.py` -> `launchers/mp_infer_launcher.py`

## Notes

- `mp_*` launchers are generic wrappers reused by multiple providers.
- Smart Train normalizes run naming and artifact layout to the shared contract.
