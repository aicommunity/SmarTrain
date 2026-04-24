> Russian version: [../ru/providers/leaf-yolo.md](../ru/providers/leaf-yolo.md)

# LEAF-YOLO provider

## Subsystem

- Provider id: `leaf-yolo`
- Runtime family used by integration: Ultralytics-compatible YOLOv8 API in provider `venv`
- Default aliases exposed by Smart Train: `yolov8n/s/m/l/x`

## Wrapper implementation

- Install/probe/index:
  - `external_providers/installer.py`
  - `external_providers/probe.py`
  - `provider_global_index.py`
- Train adapter:
  - `external_providers/adapters.py` -> `launchers/mp_train_launcher.py`
- Inference adapter:
  - `external_providers/adapters.py` -> `launchers/mp_infer_launcher.py`

## Notes

- Smart Train performs strict alias validation.
- Output layout is normalized to the shared run contract for all downstream commands.
