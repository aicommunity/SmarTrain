> English version: [../../providers/dr-yolo.md](../../providers/dr-yolo.md)

# Провайдер DR-YOLO

## Подсистема

- Provider id: `dr-yolo`
- Runtime-семейство в интеграции: Ultralytics-совместимый YOLOv8 API в provider `venv`
- Алиасы по умолчанию: `yolov8n/s/m/l/x`

## Реализация обвязки

- Установка/проверки/индекс:
  - `external_providers/installer.py`
  - `external_providers/probe.py`
  - `provider_global_index.py`
- Train-адаптер:
  - `external_providers/adapters.py` -> `launchers/mp_train_launcher.py`
- Inference-адаптер:
  - `external_providers/adapters.py` -> `launchers/mp_infer_launcher.py`

## Примечания

- Валидация алиасов модели строгая и провайдер-специфичная.
- Структура артефактов приводится к единому контракту `train/test/metadata`.
