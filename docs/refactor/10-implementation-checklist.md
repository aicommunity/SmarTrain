# Implementation checklist (операционный трек исполнения плана)

Purpose: отслеживать **фактическое** выполнение требований детализированного плана рефакторинга (`/home/user/.cursor/plans/рефакторинг_cli_и_ml_ядра_798a9741.plan.md`) и раздела «Статус исполнения и приоритетный roadmap» там же. Этот файл **не заменяет** полноту описания PR/волн в плане — только чекбоксы и короткие статус-заметки.

Companion docs:

- [`09-tech-debt.md`](./09-tech-debt.md) — компромиссы, хвосты, временные решения.
- План — источник **обязательных** DoD, тест-планов и артефактов по каждому PR.

Convention: `[ ]` не начато / не закрыто по критериям плана; `[x]` закрыто (есть код + тесты + док при необходимости). Дату закрытия и ссылку на коммит/PR добавляйте в строку «Notes».

---

## Wave 6 — сверка закрытых под-PR (код + тесты)

### PR 6.1 (спецификация)

- [x] **6.1** Документы `05b` / `05c` / `05d` присутствуют и задают контракт (дальнейшее расширение при write/migration не отменяет требований плана). Notes: `05b` краткий; при необходимости расширять примерами из шаблона PR 6.1 без ослабления полей.

### PR 6.2 (domain DTO + validation)

- [x] **6.2** Пакет `smartrain/domain/canonical/*` (models, types, validators, errors) + unit-тесты `tests/domain/canonical/`.

### PR 6.3 (read adapters)

- [x] **6.3** `smartrain/adapters/canonical/read/*`, `ReadAdapterFactory`, тесты `tests/adapters/canonical/read/`.

### PR 6.5 (частично — только read-path consumers)

- [x] **6.5-p1** `canonical_gateway.load_target` + валидация payload. Notes: `load_metrics` / `load_predictions` / `resolve_task_context` из шаблона PR 6.5 — **ещё не реализованы** (Phase B).
- [x] **6.5-p2** Частичная миграция consumers под флаг `SMARTTRAIN_CANONICAL_READ` (`model_test_cli`, `inference_cli`, `results_analyzer`) с политикой no-fallback там, где включено.
- [ ] **6.5-p3** Полное удаление source-specific веток в business-логике и расширение gateway — Phase B.

---

## Приоритетный порядок (фазы A–F)

Совпадает с таблицей roadmap в плане; выполнять по зависимостям из «Единый Execution Roadmap».

### Phase A — Wave 6 / PR 6.4 (canonical write + dual-write)

- [x] **6.4-A1** Слой записи: `write_canonical_snapshot`, `build_manifest`, layout `…/.smartrain/canonical/`, hash в manifest (`smartrain/adapters/canonical/write/*`). Notes: первый инкремент PR 6.4; расширение полей manifest/provenance по мере зрелости writer.
- [x] **6.4-A2** Режимы `run_dual_write` с `canonical_only` / `dual_write_strict` / `dual_write_best_effort` и `DualWriteReport` (canonical/legacy статусы, warnings, rollback_hint). Notes: strict не откатывает уже записанный canonical snapshot — hint фиксирует операционное действие.
- [x] **6.4-A3** Интеграция в production-путь: после успешного `persist_target_test_artifacts_state(..., status="ok")` вызывается `persist_canonical_snapshot` при `SMARTTRAIN_CANONICAL_WRITE=1` (`model_test_service.py` + `canonical_gateway.py`).
- [x] **6.4-A4** Тесты layout/manifest и dual-write (`tests/adapters/canonical/write/`). Notes: расширять по failure/rollback сценариям из полного тест-плана PR 6.4 по мере появления legacy writer hooks.

Notes:

### Phase B — Wave 6 / PR 6.5 (gateway + consumers)

- [x] **6.5-B1a** `canonical_gateway.resolve_task_context` → `TaskContext` (`domain/canonical/context.py`).
- [x] **6.5-B1b** `canonical_gateway.load_metrics` → список `CanonicalMetricsRef` (обнаружение CSV через `metrics_reader`, namespace `{task}/test_{fmt}`).
- [x] **6.5-B1c** `canonical_gateway.load_predictions` — заглушка API (пустой список до стандартизации prediction bundle).
- [x] **6.5-B1d** Довести PR 6.5 API: стабильный контракт prediction discovery, опции фильтрации split/format, документация в `05b`/CLI; golden-тесты из тест-плана PR 6.5 (базовый уровень + integration/contract coverage добавлены).
- [x] **6.5-B2a** `results_analyzer` canonical-run метрики (`_build_run_record_canonical`) теперь идут через `canonical_gateway.load_metrics` (не напрямую через `read_test_metrics_row`).
- [x] **6.5-B2b** Миграция оставшихся consumer-веток `model_test_cli` / `inference_cli` / `results_analyzer`: убрать **бизнес-**ветвления run vs model в пользу canonical path там, где это предусмотрено PR 6.5 (инвентаризация веток → удаление/изоляция).
- [x] **6.5-B3** Интеграционные тесты consumer parity и feature-flag rollout (если используется) по тест-плану PR 6.5.

Notes:

### Phase C — Wave 6 / PR 6.6–6.7 (migration + cutover)

- [x] **6.6-C1** `legacy` reader/mapper слой по путям из PR 6.6.
- [x] **6.6-C2** CLI миграции: `dry-run`, `apply`, `report-only`; машиночитаемый отчёт + summary.
- [x] **6.6-C3** Тесты: historical coverage, idempotency, safety (dry-run не пишет), reporting.
- [ ] **6.7-C4** Cutover: default `canonical_only`, удаление временных мостов по policy; regression `test_canonical_cutover`, `test_no_legacy_branch_usage` (или эквиваленты из плана).

Notes:
- 2026-05-05: Усилен reporting для PR 6.6: в report добавлены `operator_guidance` и per-item `rollback_hint`; добавлен regression test на наличие guidance для failed item.

### Phase D — Wave 4 (backend abstraction)

- [ ] **4-D1** Интерфейсы `TrainBackend` / `TestBackend` / `InferenceBackend` и общий контракт результатов (PR 4.1).
- [ ] **4-D2** Capability registry selection по матрице (PR 4.2); не сужать матрицу из [`04-task-backend-capabilities.md`](./04-task-backend-capabilities.md) без явного решения.
- [ ] **4-D3** **`UltralyticsAdapter`** как эталонная реализация (PR 4.3).
- [ ] **4-D4** Нормализация external providers под общий adapter contract (PR 4.4).

Notes:

### Phase E — Wave 5 + Analyze / Artifact v2 (волны 5, 3 analyze, 7)

- [ ] **5-E1** Task contracts/context; detection adapter extraction; metrics adapter framework (PR 5.1–5.3).
- [ ] **5-E2** Classification/segmentation readiness + consumer wiring (PR 5.4–5.5).
- [ ] **3-E3** Декомпозиция **`results_analyzer`** на слои args / interactive / service / backends / artifacts (доделать симметрию с train/test/inference по Волне 3).
- [ ] **7-E4** Schema v2 артефактов + миграция analyze на unified read; legacy reader контролируем и документирован (Волна 7).

Notes:

### Phase F — Wave 8 (clean-code hardening)

- [ ] **8-F1** Системная замена анти-паттернов из [`07-clean-code-rules.md`](./07-clean-code-rules.md).
- [ ] **8-F2** CI/review guardrails (линт-правила, чеклисты PR).
- [ ] **8-F3** Удаление legacy-веток после окончания deprecation windows из [`06-deprecation-and-alias-policy.md`](./06-deprecation-and-alias-policy.md).

Notes:

---

## Ранее выполненные блоки вне Phase A–F (сводка)

- [x] Частичная декомпозиция сервисов: `smartrain/services/*` для train/test/inference; `test_backend_dispatch` для PT/non-PT test paths.
- [x] `train_test_registry` и task-aware хуки для train/test backend id (ограничения — в `09-tech-debt.md`).

---

## Как обновлять этот файл

1. При закрытии подпункта — `[x]`, дата, PR/коммит, при отклонении от плана — запись в `09-tech-debt.md`.
2. Раз в спринт/волну — сверка с YAML-todo в файле плана (если используется) и с gate-критериями из плана.
