> English version: [../../cli/providers.md](../../cli/providers.md)

# CLI: внешние провайдеры

## `smartrain providers`

Управление внешними backend-провайдерами обучения в отдельных виртуальных окружениях.

```bash
smartrain providers status
smartrain providers install --all -y
smartrain providers doctor --verbose
smartrain providers uninstall --provider dr-yolo -y
```

Подкоманды:

- `install`: клонирование/установка выбранных провайдеров и запись в глобальный индекс.
- `uninstall`: удаление выбранных провайдеров и их записей в индексе.
- `status`: текущее состояние индекса (`installed`/`not_installed`) и пути репозиториев.
- `doctor`: проверки готовности (репозиторий, entrypoints, venv, runtime-зависимости).

## Алиасы моделей провайдера в `train`/`inference`

Используйте префиксный формат:

```bash
smartrain train --external-provider dr-yolo --model yolov8n
smartrain train --model dr-yolo:yolov8n
smartrain inference --weights dr-yolo:yolov8n --data-mode folder --source-dir images/
```

Правила:

- `provider:model` автоматически выставляет `--external-provider`.
- Для внешних провайдеров действует строгая проверка алиаса: неподдерживаемые модели отклоняются с понятной ошибкой.
- В интерактивном `train` список моделей включает алиасы установленных провайдеров и пункт `<manual>`.

## Поведение по умолчанию для внешних провайдеров

Если указан `--external-provider` и явные значения не переданы:

- модель по умолчанию берётся из каталога провайдера;
- применяются launcher-дефолты (`epochs=70`, `batch=8`, `img_size=640`);
- имя run-папки нормализуется и санитизируется:
  - `YYYY-MM-DD_HH-MM_<provider>_<model>_<epochs>epochs_b<batch>-<dataset_hash>`

## Контракт артефактов внешнего запуска

Внешние прогоны приводятся к тому же контракту, что и встроенное обучение:

- `train/weights/best.pt`
- `test/` артефакты
- `test_metrics.csv`
- `training_metadata.json`

Этот контракт обязателен для последующих команд (`analyze`, `registry`, `inference`).
