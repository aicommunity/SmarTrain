> English version: [../../providers/mp-yolo.md](../../providers/mp-yolo.md)

# Провайдер MP-YOLO

## Подсистема

- Provider id: `mp-yolo`
- Runtime-семейство в интеграции: Ultralytics-совместимый YOLOv8 API в provider `venv`
- Алиасы по умолчанию: `yolov8n/s/m/l/x`

## Реализация обвязки

- Train-адаптер:
  - `external_providers/adapters.py` -> `launchers/mp_train_launcher.py`
- Inference-адаптер:
  - `external_providers/adapters.py` -> `launchers/mp_infer_launcher.py`

## Примечания

- `mp_*` launcher-скрипты используются как универсальная обвязка для нескольких провайдеров.
- Имена run и layout артефактов нормализуются к общему контракту.
