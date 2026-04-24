> Russian version: [../ru/providers/enhanced-yolov8.md](../ru/providers/enhanced-yolov8.md)

# Enhanced YOLOv8 provider

## Subsystem

- Provider id: `enhanced-yolov8`
- Runtime family used by integration: Ultralytics-compatible YOLOv8 API in provider `venv`
- Default aliases exposed by Smart Train: `yolov8n/s/m/l/x`

## Wrapper implementation

- Installer resolves nested runnable repo path (`yolov8-main-Ghost` style layout).
- Train adapter:
  - `external_providers/adapters.py` -> `launchers/mp_train_launcher.py`
- Inference adapter:
  - `external_providers/adapters.py` -> `launchers/mp_infer_launcher.py`

## Notes

- Alias validation is strict and provider-scoped.
- Artifacts are normalized so downstream tools stay backend-agnostic.
