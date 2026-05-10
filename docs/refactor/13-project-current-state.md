# Актуальное состояние архитектуры SmarTrain (срез)

**Дата среза:** 2026-05-10  
**Назначение:** краткий narrative-snapshot структуры пакета, закрытых волн рефакторинга, границ слоёв и известных ограничений. Не заменяет детальные спецификации в `00`–`08`.

**Источник истины по статусу волн и PR:** [`10-implementation-checklist.md`](./10-implementation-checklist.md) (чекбоксы Phase A–F и заметки). Журнал компромиссов и история шагов: [`09-tech-debt.md`](./09-tech-debt.md). Базовый аудит соответствия целевой архитектуре: [`11-plan-conformance-audit.md`](./11-plan-conformance-audit.md). Волна закрытия P0/P1 разрывов: [`12-gap-closure-wave-p0-p1.md`](./12-gap-closure-wave-p0-p1.md).

---

## Статус ключевых волн (резюме)

| Трек | Содержание | Статус (по `10-implementation-checklist`) |
|------|------------|---------------------------------------------|
| Phase A | PR 6.4 canonical write, dual-write, интеграция в test path | Закрыто `[x]` |
| Phase B | PR 6.5 gateway (`load_metrics`, `resolve_task_context`, predictions API и потребители) | Закрыто `[x]` |
| Phase C | PR 6.6–6.7 migration CLI, cutover, регрессии | Закрыто `[x]` |
| Phase D | Волна 4: backend protocols, registry, UltralyticsAdapter, external provider | Закрыто `[x]` |
| Phase E | Волна 5 + analyze: task layer, cls/seg wiring, декомпозиция analyze, schema v2 / unified read | Закрыто `[x]` |
| Phase F | Волна 8: clean-code, guardrails, legacy removal по policy | Закрыто `[x]` |
| Root migration | Перенос модулей из корня `smartrain/` в доменные подпакеты | Закрыто (см. начало [`09-tech-debt.md`](./09-tech-debt.md), Historical Log) |
| Gap closure P0/P1 | Границы services, runtime neutrality, gateway-first analyze, thinning orchestrators | Закрыто (лог 2026-05-08/09 в `09-tech-debt`) |
| P0-tail / P2 | Запрет `services`→`workflows`, `workflow_adapters`, schema governance analyze | Закрыто (лог 2026-05-09 в `09-tech-debt`) |

---

## Структура пакета `smartrain`

**Корень пакета (entrypoints):**

- [`smartrain/cli.py`](../../smartrain/cli.py) — маршрутизация CLI.
- [`smartrain/__main__.py`](../../smartrain/__main__.py), [`smartrain/__init__.py`](../../smartrain/__init__.py).

**Основные подпакеты:**

| Подпакет | Роль |
|----------|------|
| [`smartrain/services/`](../../smartrain/services/) | Use-case сервисы (train / test / inference / analyze helpers); **не** импортируют `smartrain.workflows` напрямую |
| [`smartrain/workflows/`](../../smartrain/workflows/) | CLI модули, оркестраторы workflow, datasets, training, testing, inference, analyze, migration, queue, registry |
| [`smartrain/core/`](../../smartrain/core/) | Общие утилиты: `runtime/`, `training/`, [`workflow_adapters/`](../../smartrain/core/workflow_adapters/) |
| [`smartrain/backends/`](../../smartrain/backends/) | Контракты backend, registry, `ultralytics_adapter`, `external_provider_adapter`, `train_test_registry` |
| [`smartrain/tasks/`](../../smartrain/tasks/) | Task-aware контракты и адаптеры метрик (detection / classification / segmentation) |
| [`smartrain/domain/canonical/`](../../smartrain/domain/canonical/) | DTO и валидация канонической модели данных |
| [`smartrain/adapters/canonical/`](../../smartrain/adapters/canonical/) | Read/write адаптеры, legacy reader/mapper |
| [`smartrain/orchestrators/`](../../smartrain/orchestrators/) | [`canonical_gateway.py`](../../smartrain/orchestrators/canonical_gateway.py) — единая точка чтения/записи canonical |
| [`smartrain/canonical/`](../../smartrain/canonical/) | Схемы, policy, refs (облегчённые модули верхнего уровня домена canonical) |

---

## Границы слоёв и guardrails

- **Правило:** модули в `smartrain/services/*.py` не импортируют `smartrain.workflows.*`. Доступ к workflow-реализациям — через фасады [`smartrain/core/workflow_adapters/`](../../smartrain/core/workflow_adapters/):
  - `training_runtime_api.py`
  - `testing_runtime_api.py`
  - `inference_runtime_api.py`
  - `analyze_runtime_api.py`
- **Регрессия:** [`tests/regression/test_train_service_guardrails.py`](../../tests/regression/test_train_service_guardrails.py) (strict: новые прямые `services`→`workflows` импорты запрещены).

---

## Данные, canonical и миграция

- **Чтение:** [`orchestrators/canonical_gateway.py`](../../smartrain/orchestrators/canonical_gateway.py) — `load_target`, `load_metrics`, `resolve_task_context`, predictions API и др.
- **Запись:** [`adapters/canonical/write/`](../../smartrain/adapters/canonical/write/) — snapshot, manifest (provenance, хеши), [`dual_write.py`](../../smartrain/adapters/canonical/write/dual_write.py) (`canonical_only`, `dual_write_strict`, `dual_write_best_effort`).
- **Миграция:** [`workflows/migration/cli_migration.py`](../../smartrain/workflows/migration/cli_migration.py) и связанные модули; режимы dry-run / apply / report-only (см. тесты `tests/migration/`).
- **Cutover / policy:** аварийный legacy read — только при явных env-флагаx (см. [`06-deprecation-and-alias-policy.md`](./06-deprecation-and-alias-policy.md) и регрессии `tests/regression/test_canonical_cutover.py`).

---

## Задачи (task), метрики и inference

- Контракты: [`tasks/contracts.py`](../../smartrain/tasks/contracts.py); нормализация метрик: [`tasks/metrics.py`](../../smartrain/tasks/metrics.py) и подпакеты `detection/`, `classification/`, `segmentation/`.
- Gateway подключает task-aware нормализацию при загрузке метрик (см. Phase E в `10-implementation-checklist.md`).
- **Inference:** `--task` / hint пробрасывается в capability resolution и runtime backend; отчёты поддерживают task-aware outputs (в т.ч. cls/seg), с деградацией для внешних провайдеров при отсутствии полей.
- **Model test / `pt_uni`:** внутреннее сравнение **`pt_uni`** включено для **detection**, **classification** и **segmentation** (проброс `task_type` в Ultralytics `val`); контракт — [`14-pt-uni-compare-contract.md`](./14-pt-uni-compare-contract.md).

---

## Backends и новые фреймворки

- Интерфейсы и результаты выполнения: [`backends/contracts.py`](../../smartrain/backends/contracts.py).
- Реестр возможностей: [`backends/registry.py`](../../smartrain/backends/registry.py), маршрутизация train/test/infer: [`backends/train_test_registry.py`](../../smartrain/backends/train_test_registry.py).
- Эталон локального backend: [`backends/ultralytics_adapter.py`](../../smartrain/backends/ultralytics_adapter.py); внешние провайдеры: [`backends/external_provider_adapter.py`](../../smartrain/backends/external_provider_adapter.py).
- **Добавление нового фреймворка (ожидаемый путь):** реализация адаптера по контракту, регистрация в capability matrix, тесты, запись provenance (`task_type` / `backend_type` в manifest и training metadata), при необходимости расширение [`04-task-backend-capabilities.md`](./04-task-backend-capabilities.md).

---

## Analyze и отчёты

- Чтение метрик в canonical-режиме: преимущественно через gateway; legacy fallback — **явный**, policy-gated, с диагностикой (например `metrics_read_policy` в [`services/analyze_format_compare_service.py`](../../smartrain/services/analyze_format_compare_service.py)).
- **Schema governance:** [`workflows/analyze/analyze_schema_contracts.py`](../../smartrain/workflows/analyze/analyze_schema_contracts.py) — версии схем, валидация session manifest и format-compare index на write-path ([`analyze_report.py`](../../smartrain/workflows/analyze/analyze_report.py), finalize/compare).
- Крупные фасады (с делегированием в сервисы): [`workflows/analyze/results_analyzer.py`](../../smartrain/workflows/analyze/results_analyzer.py), множество `analyze_*_service.py`.

---

## Operational limits (резюме)

Краткий список; детали и ссылки на код — в разделе **Operational Limits** [`09-tech-debt.md`](./09-tech-debt.md).

- Внешние провайдеры могут не отдавать полные cls/seg поля; допустим деградированный контракт до расширения провайдеров.
- Внутренний **`pt_uni` compare** в model test — только **detection**.
- **Canonical model read** ([`adapters/canonical/read/model_adapter.py`](../../smartrain/adapters/canonical/read/model_adapter.py)): разрешение `task_type` — metadata → подсказка по имени файла → последний резерв `detection`; `backend_type` — metadata → подсказка по формату весов (`onnx` → onnxruntime, `engine`/`trt` → tensorrt, иначе ultralytics).
- Крупные модули [`model_training_module.py`](../../smartrain/workflows/training/model_training_module.py) и [`cli.py`](../../smartrain/cli.py) остаются композиционными «толстыми» входами; логика вынесена в соседние `*_service` модули.

---

## Следующий горизонт (backlog, без обязательства срока)

1. **Расширение `pt_uni` / internal compare на classification и segmentation:** спроектировать контракт метрик и артефактов сравнения (аналог текущего detection-only пути), затем снять guard в orchestrator и добавить регрессионные тесты.
2. **Дальнейшее утончение фасадов** при росте функциональности (по мере боли в ревью): точечный перенос оставшихся блоков из `results_analyzer` / `model_training_module` без изменения CLI.

---

## Связанные документы

- Refactor: [`00-scope.md`](./00-scope.md) … [`08-rollout-checklist.md`](./08-rollout-checklist.md), [`09-tech-debt.md`](./09-tech-debt.md), [`10-implementation-checklist.md`](./10-implementation-checklist.md), [`11-plan-conformance-audit.md`](./11-plan-conformance-audit.md), [`12-gap-closure-wave-p0-p1.md`](./12-gap-closure-wave-p0-p1.md).
- Детализированный план рефакторинга (Cursor): `.cursor/plans/рефакторинг_cli_и_ml_ядра_798a9741.plan.md` — операционный статус по чекбоксам вести в `10-implementation-checklist.md`.
- Обзор для разработчиков: [`../development/architecture.md`](../development/architecture.md), [`../development/extension-guide.md`](../development/extension-guide.md).
