> English version: [../../providers/ssdm-yolo.md](../../providers/ssdm-yolo.md)

# Провайдер SSDM-YOLO

## Подсистема

- Provider id: `ssdm-yolo`
- Runtime-семейство в интеграции: Ultralytics-совместимый YOLOv8 API в provider `venv`
- Алиасы по умолчанию: `yolov8n/s/m/l/x`

## Реализация обвязки

- Installer обрабатывает особенности структуры SSDM-репозитория (архив/распаковка, поиск runnable root).
- Train-адаптер:
  - `external_providers/adapters.py` -> `launchers/mp_train_launcher.py`
- Inference-адаптер:
  - `external_providers/adapters.py` -> `launchers/mp_infer_launcher.py`

## Примечания

- Провайдер-специфичное разрешение корня репозитория находится в installer-слое.
- После запуска артефакты приводятся к стандартному контракту Smart Train.
