> English version: [../../providers/mfel-yolo.md](../../providers/mfel-yolo.md)

# Провайдер MFEL-YOLO

## Подсистема

- Provider id: `mfel-yolo`
- Runtime-семейство: MFEL fork с кастомными блоками поверх Ultralytics API
- Алиасы по умолчанию:
  - `mfel-yolo` (предпочтительный дефолт)
  - `e_pan+` (если конфиг присутствует в репозитории провайдера)

## Реализация обвязки

- Train-адаптер:
  - `external_providers/adapters.py` -> `launchers/mfel_train_launcher.py`
- Inference-адаптер:
  - `external_providers/adapters.py` -> `launchers/mfel_infer_launcher.py`
- Launcher для fallback-валидации (когда builtin test не может импортировать кастомные блоки):
  - `external_providers/launchers/mfel_val_launcher.py`

## Специфика MFEL

- Патчинг символов для отсутствующих сущностей форка (совместимость загрузки checkpoint).
- Стабилизирующие train-дефолты в launcher:
  - `amp=False`, `optimizer=AdamW`, `pretrained=False`, настроенные LR/warmup/weight decay.
- `mfel_val_launcher.py` сохраняет `results.csv`, чтобы `test_metrics.csv` содержал реальные метрики.

## Примечания

- Проверка алиасов модели строгая.
- Структура run-артефактов приводится к общему внешнему контракту.
