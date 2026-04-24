> English version: [../../providers/enhanced-yolov8.md](../../providers/enhanced-yolov8.md)

# Провайдер Enhanced YOLOv8

## Подсистема

- Provider id: `enhanced-yolov8`
- Runtime-семейство в интеграции: Ultralytics-совместимый YOLOv8 API в provider `venv`
- Алиасы по умолчанию: `yolov8n/s/m/l/x`

## Реализация обвязки

- Installer разрешает вложенный runnable-path (layout вида `yolov8-main-Ghost`).
- Train-адаптер:
  - `external_providers/adapters.py` -> `launchers/mp_train_launcher.py`
- Inference-адаптер:
  - `external_providers/adapters.py` -> `launchers/mp_infer_launcher.py`

## Примечания

- Валидация алиасов модели строгая и ограничена каталогом провайдера.
- Артефакты нормализуются, чтобы downstream-команды были backend-agnostic.
