> English version: [../../providers/leaf-yolo.md](../../providers/leaf-yolo.md)

# Провайдер LEAF-YOLO

## Подсистема

- Provider id: `leaf-yolo`
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

- Smart Train выполняет строгую проверку алиаса модели для провайдера.
- Выходные артефакты нормализуются к общему run-контракту.
