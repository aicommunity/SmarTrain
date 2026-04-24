> Russian version: [../ru/providers/ssdm-yolo.md](../ru/providers/ssdm-yolo.md)

# SSDM-YOLO provider

## Subsystem

- Provider id: `ssdm-yolo`
- Runtime family used by integration: Ultralytics-compatible YOLOv8 API in provider `venv`
- Default aliases exposed by Smart Train: `yolov8n/s/m/l/x`

## Wrapper implementation

- Installer handles SSDM repository layout specifics (archive/unpacked runnable root resolution).
- Train adapter:
  - `external_providers/adapters.py` -> `launchers/mp_train_launcher.py`
- Inference adapter:
  - `external_providers/adapters.py` -> `launchers/mp_infer_launcher.py`

## Notes

- Provider-specific repository root resolution is part of installer logic.
- Output artifacts are normalized to Smart Train contract after run.
