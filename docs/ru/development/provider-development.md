> English version: [../../development/provider-development.md](../../development/provider-development.md)

# Разработка новых внешних провайдеров

Документ описывает актуальный интеграционный контракт для добавления нового внешнего провайдера.

## 1. Добавить спецификацию провайдера

Файл: `smartrain/external_providers/registry.py`

Добавьте новый `ExternalProviderSpec` с:

- уникальным `id` (lowercase, kebab-case),
- названием,
- URL/branch репозитория,
- номинальными train/infer entrypoint-скриптами.

## 2. Реализовать установку и проверки готовности

Файлы:

- `smartrain/external_providers/installer.py`
- `smartrain/external_providers/probe.py`
- `smartrain/provider_global_index.py`

Требования:

- установка должна быть идемпотентной;
- `venv` провайдера должен быть изолированным;
- runtime-зависимости должны ставиться в provider `venv`;
- запись в индексе должна содержать валидные `repo_path` и `venv_path`.

## 3. Адаптер и launcher-обвязка

Файлы:

- `smartrain/external_providers/adapters.py`
- `smartrain/external_providers/launchers/*.py`
- `smartrain/external_providers/runner.py`

Правила:

- адаптер маппит аргументы Smart Train в аргументы launcher-скрипта провайдера;
- launcher запускается в provider `venv` и поддерживает детерминированный CLI-контракт;
- если у провайдера кастомные блоки/импорты, патчинг символов выполняется локально в launcher.

## 4. Каталог моделей и строгая валидация

Файл: `smartrain/core/training/train_model_catalog.py`

Добавьте алиасы провайдера в `_EXTERNAL_PROVIDER_FALLBACK_ALIASES` и, при необходимости, динамическое обнаружение.

Путь валидации:

- `train`: `train_cli_callbacks` / `train_interactive_helpers_service` через `is_supported_external_provider_model(...)`.
- `inference`: `inference_cli.py` через ту же строгую проверку по каталогу провайдера.

Если алиас не поддерживается, команда должна завершаться ошибкой с перечнем поддерживаемых алиасов.

## 5. Дефолты для внешних провайдеров

Текущее поведение при `--external-provider` и отсутствии явных аргументов:

- модель по умолчанию берётся из каталога провайдера;
- для отсутствующих значений применяются launcher-дефолты: `epochs=70`, `batch=8`, `img_size=640`.

Логика расположена в `services/training/train_interactive_helpers_service.py` (`apply_external_provider_defaults`).

## 6. Имена run и нормализация артефактов

Имя run должно быть path-safe и стабильным:

- `YYYY-MM-DD_HH-MM_<provider>_<model>_<epochs>epochs_b<batch>-<dataset_hash>`

Токен модели обязательно санитизируется из пути/имени файла, чтобы не создавать вложенные некорректные каталоги.

Обязательный нормализованный контракт артефактов:

- `<run_dir_name>.pt` в корне run
- `test/`
- `test_metrics.csv`
- `training_metadata.json`

Хелперы нормализации находятся в `services/training/train_model_resolution_service.py`.

## 7. Паттерн fallback как у MFEL (совместимость кастомных блоков)

Когда builtin `test_yolo` не может загрузить checkpoint провайдера:

- используйте provider-side launcher fallback (пример: `mfel_val_launcher.py`);
- формируйте машиночитаемый CSV с тестовыми метриками (`results.csv` и `test_metrics.csv` в run-root);
- синхронизируйте `training_metadata.json` со фактическим результатом fallback.

## 8. Обязательные обновления документации

Для каждого нового провайдера обновляйте:

- `docs/cli/providers.md`
- `docs/providers/overview.md`
- профильный файл `docs/providers/<provider-id>.md`
- русские зеркала в `docs/ru/...`

Также обновляйте индексы:

- `docs/index.md`
- `docs/ru/index.md`

## 9. Минимальный чеклист тестов

- unit-тесты адаптера: `tests/test_external_providers_adapters.py`
- unit-тесты раннера: `tests/test_external_providers_runner.py`
- CLI-поведение (`provider:model`, валидация): `tests/test_train_interactive.py`, `tests/test_inference_cli.py`
- тесты launcher-специфики (пример MFEL): `tests/test_mfel_launchers.py`
- e2e smoke: one-epoch `train` для провайдера с проверкой артефактов и метаданных.
