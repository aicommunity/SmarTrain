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
- [x] **6.7-C4** Cutover: default `canonical_only`, удаление временных мостов по policy; regression `test_canonical_cutover`, `test_no_legacy_branch_usage` (или эквиваленты из плана).

Notes:
- 2026-05-05: Усилен reporting для PR 6.6: в report добавлены `operator_guidance` и per-item `rollback_hint`; добавлен regression test на наличие guidance для failed item.
- 2026-05-05: Выполнен cutover policy для consumer read-path: canonical включён по умолчанию, legacy разрешён только через явный аварийный policy-флаг `SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK=1` вместе с `SMARTTRAIN_CANONICAL_READ=0`; добавлены regression тесты `tests/regression/test_canonical_cutover.py` и `tests/regression/test_no_legacy_branch_usage.py`.

### Phase D — Wave 4 (backend abstraction)

- [x] **4-D1** Интерфейсы `TrainBackend` / `TestBackend` / `InferenceBackend` и общий контракт результатов (PR 4.1).
- [x] **4-D2** Capability registry selection по матрице (PR 4.2); не сужать матрицу из [`04-task-backend-capabilities.md`](./04-task-backend-capabilities.md) без явного решения.
- [x] **4-D3** **`UltralyticsAdapter`** как эталонная реализация (PR 4.3).
- [x] **4-D4** Нормализация external providers под общий adapter contract (PR 4.4).

Notes:
- 2026-05-05: Реализован PR 4.1 foundation: в `smartrain/backends/contracts.py` добавлены protocol-интерфейсы `TrainBackend`/`TestBackend`/`InferenceBackend` и общий `BackendExecutionResult`; `CapabilityRegistry` усилен строгой валидацией `require` и helper `resolve_backend_id`; добавлены contract-тесты `tests/test_backend_registry_capabilities.py`.
- 2026-05-05: Реализован PR 4.2 базового уровня: расширен `train_test_registry` до train/test/infer capability routing (`resolve_infer_backend`), добавлено покрытие `tests/test_train_test_registry.py`; `inference_service` подключён к capability resolver как policy-слой (с warning при runtime mismatch, без жёсткого падения).
- 2026-05-05: Реализован PR 4.3 базового уровня: добавлен `smartrain/backends/ultralytics_adapter.py` как reference adapter (capability contract + infer entrypoint), `inference_service` использует adapter для локального backend creation; добавлены тесты `tests/test_ultralytics_adapter.py`.
- 2026-05-05: Реализован PR 4.4 в текущем scope: `ExternalProviderAdapter` покрывает external inference и external train execution wiring; `inference_service` и `train_service` используют adapter contract (в `train_service` через совместимую инъекцию runner-функций для сохранения тестового/legacy поведения). Тесты: `tests/test_external_provider_adapter.py`, `tests/test_inference_cli.py`, `tests/test_train_interactive.py`.

### Phase E — Wave 5 + Analyze / Artifact v2 (волны 5, 3 analyze, 7)

- [x] **5-E1** Task contracts/context; detection adapter extraction; metrics adapter framework (PR 5.1–5.3).
- [ ] **5-E2** Classification/segmentation readiness + consumer wiring (PR 5.4–5.5).
- [x] **3-E3** Декомпозиция **`results_analyzer`** на слои args / interactive / service / backends / artifacts (доделать симметрию с train/test/inference по Волне 3).
- [x] **7-E4** Schema v2 артефактов + миграция analyze на unified read; legacy reader контролируем и документирован (Волна 7). Notes: `format-compare` (analyze) переведён на `canonical_gateway.load_metrics` с поддержкой `split` (test/val), чтобы значения метрик шли через canonical/unified чтение; добавленные/обновленные regression-тесты покрывают `format_compare` блок.

Notes:
- 2026-05-05: Закрыт 5-E1: добавлены `tasks/context.py` (`TaskExecutionContext`) и `tasks/metrics.py` (task-aware metrics adapter framework), расширен `tasks/__init__.py`, и подключена task-aware metrics normalization в `canonical_gateway.load_metrics` (namespace + primary metrics через task adapter). Добавлены/обновлены тесты `tests/test_task_contracts.py` и `tests/orchestrators/test_canonical_gateway_extensions.py`.
- 2026-05-05: Частичный прогресс по 5-E2: в `model_test_orchestrator` добавлен task-aware guard для internal `pt_uni` compare (только detection), cls/seg path теперь явно пропускает detection-only compare с информативным сообщением; добавлен regression тест в `tests/test_model_test_cli.py`.
- 2026-05-05: Дополнен consumer wiring по 5-E2 для inference runtime: в `inference_cli` добавлен `--task` (detect/classify/segment aliases), а в `services/inference_service` capability routing (`resolve_infer_backend`) теперь получает нормализованный `task_type` вместо жёсткого `"detection"`; добавлен regression `test_inference_passes_task_hint_to_capability_resolution`.
- 2026-05-05: Дополнен adapter contract слой для 5-E2: `UltralyticsAdapter.infer` и `ExternalProviderAdapter.infer` теперь заполняют `BackendExecutionResult.task_type` из task hint (`task_to_metadata_task_type`) вместо детект-only default; добавлены тесты `test_ultralytics_adapter_infer_uses_task_hint` и `test_external_provider_adapter_propagates_task_hint`.
- 2026-05-05: Дополнена task-aware целостность inference artifact path: `inference` report теперь несёт `task_type`, `wrap_inference_report_v2` формирует `task_type`/`v2.metrics.namespace` из payload, а `task_to_metadata_task_type` расширен для canonical значений (`detection`/`segmentation`) без регресса в alias-режиме.
- 2026-05-05: Продолжен 5-E2 (consumer wiring): `InferenceBackendRegistry` переведён с detection-only capability check на task-aware (`KNOWN_TASKS` + `task_type` в `create_local_backend`), а `UltralyticsAdapter` теперь пробрасывает нормализованный `task_type` до registry; расширен regression `test_ultralytics_adapter_infer_uses_task_hint` (проверка фактического wiring параметра).
- 2026-05-05: Продолжен 5-E2 (runtime output contract): `BackendPrediction` переведён на task-aware envelope (`task_type` + `outputs`) с backward-compatible accessor `detections`; `inference_service` теперь явно передаёт `task_type` в `backend.predict(...)`; добавлен regression `test_inference_passes_task_hint_to_runtime_backend_predict`.
- 2026-05-05: Продолжен 5-E2 (task-specific outputs): `UltralyticsBackend` формирует task-aware payload (`classification.top1/top_k`, `segments` для segmentation, `detections` для detection), а `inference_service` пишет универсальный `images[].task_outputs` и summary `task_outputs_total` (при сохранении legacy-поля `images[].detections` для совместимости). Добавлены regression тесты `test_inference_writes_classification_task_outputs` и `test_inference_writes_segmentation_task_outputs`.
- 2026-05-05: Продолжен 5-E2 для external inference consumer wiring: external report path выровнен по unified summary-контракту (`task_outputs_total=0`, `detections_total=0`, zero-count image counters, `images=[]`), добавлен regression в `test_inference_external_provider_parsed_from_prefixed_weights`.
- 2026-05-05: Продолжен 5-E2 для external runtime contract: external infer runner теперь может вернуть structured payload (`return_code` + `images[].task_outputs`), `inference_service` нормализует его в unified report (`images[].task_outputs` + task-aware summary counters); добавлен regression `test_inference_external_provider_accepts_task_outputs_payload`.
- 2026-05-05: Продолжен 5-E2 на provider-launcher уровне: `build_external_infer_spec`/`runner.run_external_infer` поддерживают `--result-json` и structured результат (`return_code` + `images`), `mp_infer_launcher` и `mfel_infer_launcher` пишут task-aware JSON payload (detections + `task_outputs`). Добавлен regression `test_run_external_infer_returns_structured_payload_when_result_json_written`.
- 2026-05-05: Дожат 5-E2 для external task routing: task hint (`--task`) теперь прокидывается по всей external infer цепочке (`inference_service` -> `ExternalProviderAdapter`/`ExternalProviderBackend` -> `run_external_infer` -> launcher), а launcher-structured payload теперь учитывает task type (`classification` top1/top-k, `segmentation` polygons, `detection` boxes). Добавлены regression тесты `test_run_external_infer_passes_task_to_launcher`, `test_dr_infer_adapter_passes_task_and_result_json`, `test_mfel_infer_structured_outputs_classification_top1`, `test_mp_infer_structured_outputs_segmentation_polygon`.
- 2026-05-05: Запущена декомпозиция `results_analyzer` (3-E3) с сохранением CLI-совместимости: вынесены service/helper-слои `services/analyze_artifacts.py` (session/output paths), `services/analyze_data_yaml.py` (data.yaml discovery), `services/analyze_table_service.py` (`scan`/`export-table` workflow); в `results_analyzer` оставлены thin-wrapper entrypoints.
- 2026-05-05: Продолжена декомпозиция `results_analyzer` (3-E3): compare-пайплайн (`cmd_compare`) переведён на выделенный service `services/analyze_compare_service.py` (`run_compare_workflow`) с сохранением интерактивного/CLI контракта и регрессий.
- 2026-05-05: Продолжена декомпозиция `results_analyzer` (3-E3): artifact-builders для analyze-сессий (`confidence_recommendations`, `speed_quality`) вынесены в `services/analyze_artifact_builders.py`; публичное поведение сохранено через thin-wrapper функции в `results_analyzer`.
- 2026-05-05: Продолжена декомпозиция `results_analyzer` (3-E3): вынесен `format_compare` builder в `services/analyze_format_compare_service.py`; `_write_format_compare_artifacts` в `results_analyzer` теперь thin-wrapper.
- 2026-05-05: Продолжена декомпозиция `results_analyzer` (3-E3): интерактивная orchestration-логика (`cmd_interactive` flow) вынесена в `services/analyze_interactive_service.py` (`run_interactive_workflow`), при этом CLI-контракт и сценарии quality/speed/full сохранены.
- 2026-05-05: Закрыт 3-E3 в текущем scope: `leaderboard` orchestration и speed fallback вынесены в `services/analyze_leaderboard_service.py`; `results_analyzer` оставлен thin facade над выделенными service/helper слоями.

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
