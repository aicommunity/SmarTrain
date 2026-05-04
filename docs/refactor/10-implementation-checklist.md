# Implementation checklist (операционный трек исполнения плана)

Purpose: отслеживать **фактическое** выполнение требований детализированного плана рефакторинга (`/home/user/.cursor/plans/рефакторинг_cli_и_ml_ядра_798a9741.plan.md`) и раздела «Статус исполнения и приоритетный roadmap» там же. Этот файл **не заменяет** полноту описания PR/волн в плане — только чекбоксы и короткие статус-заметки.

Companion docs:

- [`09-tech-debt.md`](./09-tech-debt.md) — компромиссы, хвосты, временные решения.
- План — источник **обязательных** DoD, тест-планов и артефактов по каждому PR.

Convention: `[ ]` не начато / не закрыто по критериям плана; `[x]` закрыто (есть код + тесты + док при необходимости). Дату закрытия и ссылку на коммит/PR добавляйте в строку «Notes».

---

## Приоритетный порядок (фазы A–F)

Совпадает с таблицей roadmap в плане; выполнять по зависимостям из «Единый Execution Roadmap».

### Phase A — Wave 6 / PR 6.4 (canonical write + dual-write)

- [ ] **6.4-A1** Реализовать слой записи по структуре из плана: `CanonicalWriter.write(...)`, `write_manifest`, layout, hash/provenance (см. шаблон PR 6.4).
- [ ] **6.4-A2** Реализовать режимы dual-write (`canonical_only`, `dual_write_strict`, `dual_write_best_effort`) и отчёт `DualWriteReport` с полями из плана.
- [ ] **6.4-A3** Интегрировать writer минимум в один production use-case (`train` / `test` / `inference`) с тестами из плана PR 6.4.
- [ ] **6.4-A4** Тесты: layout, dual-write consistency, failure handling, rollback guidance (по перечню PR 6.4).

Notes:

### Phase B — Wave 6 / PR 6.5 (gateway + consumers)

- [ ] **6.5-B1** Расширить `canonical_gateway`: помимо `load_target` — контракты уровня `load_metrics`, `load_predictions`, `resolve_task_context` (как в шаблоне PR 6.5; имена/API зафиксировать в коде и синхронизировать с планом при необходимости).
- [ ] **6.5-B2** Миграция `model_test_cli` / `inference_cli` / `results_analyzer`: убрать **бизнес-**ветвления run vs model в пользу canonical path там, где это предусмотрено PR 6.5 (инвентаризация веток → удаление/изоляция).
- [ ] **6.5-B3** Интеграционные тесты consumer parity и feature-flag rollout (если используется) по тест-плану PR 6.5.

Notes:

### Phase C — Wave 6 / PR 6.6–6.7 (migration + cutover)

- [ ] **6.6-C1** `legacy` reader/mapper слой по путям из PR 6.6.
- [ ] **6.6-C2** CLI миграции: `dry-run`, `apply`, `report-only`; машиночитаемый отчёт + summary.
- [ ] **6.6-C3** Тесты: historical coverage, idempotency, safety (dry-run не пишет), reporting.
- [ ] **6.7-C4** Cutover: default `canonical_only`, удаление временных мостей по policy; regression `test_canonical_cutover`, `test_no_legacy_branch_usage` (или эквиваленты из плана).

Notes:

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

## Уже реализовано (high-level, для связности с репозиторием)

Не снимает обязанность закрыть фазы выше по полному DoD плана.

- [x] Wave 6 ранняя реализация: domain canonical (`smartrain/domain/canonical/*`), read adapters, `ReadAdapterFactory`, `canonical_gateway.load_target`, staged consumers под `SMARTTRAIN_CANONICAL_READ`.
- [x] Частичная декомпозиция сервисов: `smartrain/services/*` для train/test/inference; `test_backend_dispatch` для PT/non-PT test paths.
- [x] `train_test_registry` и task-aware хуки для train/test backend id (см. `09-tech-debt.md` для ограничений).

---

## Как обновлять этот файл

1. При закрытии подпункта — `[x]`, дата, PR/коммит, при отклонении от плана — запись в `09-tech-debt.md`.
2. Раз в спринт/волну — сверка с YAML-todo в файле плана (если используется) и с gate-критериями из плана.
