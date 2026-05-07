# Tech Debt Log

Operational execution status and PR-level checkboxes: [`10-implementation-checklist.md`](./10-implementation-checklist.md).

Purpose: keep a running list of refactor leftovers and intentional short-term compromises.

- 2026-05-06: Root-package migration plan started (`root-package-structure-migration`). Inventory lock fixed for Batch 1 candidates:
  - moved `smartrain/provider_install_state.py` -> `smartrain/providers/core/provider_install_state.py`
  - moved `smartrain/environment_profile.py` -> `smartrain/core/runtime/environment_profile.py`
  - moved `smartrain/inference_perf.py` -> `smartrain/workflows/inference/inference_perf.py`
  - moved `smartrain/unified_validator_core.py` -> `smartrain/workflows/testing/unified_validator_core.py`
  - moved `smartrain/unified_metrics_adapter.py` -> `smartrain/workflows/testing/unified_metrics_adapter.py`
- 2026-05-06: Batch 1 verification complete (targeted regression green: 62 passed). Residual debt for Batch 1: `no residual debt`.
- 2026-05-06: Batch 2 complete: removed unused root module `smartrain/label_pred_match.py` after zero-reference recheck. Full regression green (`582 passed, 1 skipped`). Residual debt for Batch 2: `no residual debt`.
- 2026-05-06: Batch 3 (domain migration slice) complete for provider domain:
  - moved `smartrain/provider_global_index.py` -> `smartrain/providers/core/global_index.py`
  - moved `smartrain/providers_cli.py` -> `smartrain/providers/cli.py`
  - updated CLI dispatch module-paths and all imports/tests to new provider package paths.
  - targeted regression green (`69 passed`).
  Residual debt for this Batch 3 slice: `no residual debt`.
- 2026-05-06: Batch 4 complete: documentation synchronized for package structure migration (`README.md`, `docs/development/architecture.md`) and final full regression green (`582 passed, 1 skipped`). Residual debt for Batch 4: `no residual debt`.
- 2026-05-06: Root-package migration next wave complete (balanced decomposition slice):
  - moved CLI support modules to `smartrain/cli_support/*` (`cli_argparse`, `cli_contracts`, `cli_prompts`, `cli_replay`)
  - moved dataset utility modules to `smartrain/workflows/datasets/*` (`dataset_cli_common`, `data_yaml_normalize`, `dataset_passport`, `dataset_hash`)
  - moved analyze utility modules to `smartrain/workflows/analyze/*` (`analyze_cache`, `analyze_models`, `compare_service`)
  - updated imports/tests and validated with full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Additional canonical-structure wave complete:
  - moved `smartrain/canonical_refs.py` -> `smartrain/canonical/refs.py`
  - moved `smartrain/deprecation_policy.py` -> `smartrain/canonical/policy.py`
  - moved `smartrain/artifact_schema_v2.py` -> `smartrain/canonical/schema.py`
  - updated canonical-related imports and tests; full regression green (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration next wave complete (queue/registry slice):
  - moved `smartrain/training_queue.py` -> `smartrain/workflows/queue/training_queue.py`
  - moved `smartrain/training_queue_cli.py` -> `smartrain/workflows/queue/training_queue_cli.py`
  - moved `smartrain/registry_cli.py` -> `smartrain/workflows/registry/registry_cli.py`
  - updated CLI dispatch/imports/tests to new module paths and preserved queue default file location under package root.
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration next wave complete (datasets access/augment/balance slice):
  - moved `smartrain/dataset_access.py` -> `smartrain/workflows/datasets/dataset_access.py`
  - moved `smartrain/dataset_augment.py` -> `smartrain/workflows/datasets/dataset_augment.py`
  - moved `smartrain/dataset_balance.py` -> `smartrain/workflows/datasets/dataset_balance.py`
  - updated CLI dispatch/imports/monkeypatch paths and validated with targeted (`57 passed`) + full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration next wave complete (datasets orient/prune/former/roi slice):
  - moved `smartrain/dataset_orient.py` -> `smartrain/workflows/datasets/dataset_orient.py`
  - moved `smartrain/dataset_prune.py` -> `smartrain/workflows/datasets/dataset_prune.py`
  - moved `smartrain/dataset_former.py` -> `smartrain/workflows/datasets/dataset_former.py`
  - moved `smartrain/dataset_roi_yolo.py` -> `smartrain/workflows/datasets/dataset_roi_yolo.py`
  - updated CLI dispatch/imports/test monkeypatch paths and validated with targeted (`43 passed`) + full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration next wave complete (utility CLI relocation slice):
  - moved `smartrain/cvat_cli.py` -> `smartrain/workflows/datasets/cvat_cli.py`
  - moved `smartrain/sahi_cli.py` -> `smartrain/workflows/inference/sahi_cli.py`
  - moved `smartrain/heatmap_cli.py` -> `smartrain/workflows/inference/heatmap_cli.py`
  - updated CLI dispatch/imports/tests and validated with targeted (`test_imports` + `test_cli_subprocess`) and full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration next wave complete (model tooling CLI relocation slice):
  - moved `smartrain/model_convert_cli.py` -> `smartrain/workflows/models/model_convert_cli.py`
  - moved `smartrain/model_release_cli.py` -> `smartrain/workflows/models/model_release_cli.py`
  - updated CLI dispatch/imports/tests and API docs references to new module paths.
  - validated with targeted (`test_model_convert_cli` + `test_imports` + `test_cli_subprocess`) and full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration next wave complete (migration tooling relocation slice):
  - moved `smartrain/cli_migration.py` -> `smartrain/workflows/migration/cli_migration.py`
  - moved `smartrain/migrate_models_to_smartrain.py` -> `smartrain/workflows/migration/migrate_models_to_smartrain.py`
  - updated CLI dispatch/imports/tests and API docs references to new module paths.
  - validated with targeted migration+CLI tests and full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration next wave complete (analyze utility relocation slice):
  - moved `smartrain/plot_creator.py` -> `smartrain/workflows/analyze/plot_creator.py`
  - moved `smartrain/clearml_upload.py` -> `smartrain/workflows/analyze/clearml_upload.py`
  - updated CLI dispatch/imports/tests and API docs references to new module paths.
  - validated with targeted (`test_imports` + `test_cli_subprocess`) and full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration next wave complete (training core utilities relocation slice):
  - moved `smartrain/train_backend_registry.py` -> `smartrain/core/training/train_backend_registry.py`
  - moved `smartrain/train_model_catalog.py` -> `smartrain/core/training/train_model_catalog.py`
  - moved `smartrain/train_model_resolver.py` -> `smartrain/core/training/train_model_resolver.py`
  - moved `smartrain/external_model_ref.py` -> `smartrain/core/training/external_model_ref.py`
  - moved `smartrain/train_profile.py` -> `smartrain/core/training/train_profile.py`
  - moved `smartrain/confidence_recommendation.py` -> `smartrain/core/training/confidence_recommendation.py`
  - updated imports/tests/docs references and validated with targeted (`89 passed`) + full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration next wave complete (runtime utility relocation slice):
  - moved `smartrain/device_selector.py` -> `smartrain/core/runtime/device_selector.py`
  - moved `smartrain/mpl_runtime.py` -> `smartrain/core/runtime/mpl_runtime.py`
  - moved `smartrain/path_portable.py` -> `smartrain/core/runtime/path_portable.py`
  - moved `smartrain/workspace_path_repair.py` -> `smartrain/core/runtime/workspace_path_repair.py`
  - moved `smartrain/ultralytics_ephemeral.py` -> `smartrain/core/runtime/ultralytics_ephemeral.py`
  - updated imports/tests references and validated with targeted (`68 passed`) + full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration next wave complete (model-test module relocation slice):
  - moved `smartrain/model_test_cli.py` -> `smartrain/workflows/testing/model_test_cli.py`
  - moved `smartrain/model_test_service.py` -> `smartrain/workflows/testing/model_test_service.py`
  - moved `smartrain/model_test_backends.py` -> `smartrain/workflows/testing/model_test_backends.py`
  - moved `smartrain/model_test_backend_runner.py` -> `smartrain/workflows/testing/model_test_backend_runner.py`
  - moved `smartrain/model_test_onnx_worker.py` -> `smartrain/workflows/testing/model_test_onnx_worker.py`
  - updated imports/tests/scripts/docs references and validated with targeted (`53 passed`) + full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration next wave complete (model support utility relocation slice):
  - moved `smartrain/model_context.py` -> `smartrain/workflows/models/model_context.py`
  - moved `smartrain/tensorrt_checks.py` -> `smartrain/workflows/models/tensorrt_checks.py`
  - updated imports/tests references and validated with targeted (`52 passed`) + full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration wave A complete (dataset helper relocation slice):
  - moved `smartrain/cvat11_converter.py` -> `smartrain/workflows/datasets/cvat11_converter.py`
  - moved `smartrain/yolo_labels.py` -> `smartrain/workflows/datasets/yolo_labels.py`
  - moved `smartrain/weighted_yolo_dataset.py` -> `smartrain/workflows/datasets/weighted_yolo_dataset.py`
  - updated imports/tests references and validated with targeted (`40 passed`) + full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration wave B complete (run lifecycle relocation slice):
  - moved `smartrain/run_discovery.py` -> `smartrain/core/runtime/run_discovery.py`
  - moved `smartrain/run_artifacts.py` -> `smartrain/core/runtime/run_artifacts.py`
  - moved `smartrain/run_bundle_copy.py` -> `smartrain/core/runtime/run_bundle_copy.py`
  - updated imports/tests references and validated with targeted (`72 passed`) + full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration wave C complete (workspace/metrics relocation slice):
  - moved `smartrain/metrics_reader.py` -> `smartrain/workflows/analyze/metrics_reader.py`
  - moved `smartrain/workspace_paths.py` -> `smartrain/core/runtime/workspace_paths.py`
  - updated imports/tests/docs references and validated with targeted suite + full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration wave D complete (inference domain relocation slice):
  - moved `smartrain/inference_backends.py` -> `smartrain/workflows/inference/inference_backends.py`
  - moved `smartrain/inference_cli.py` -> `smartrain/workflows/inference/inference_cli.py`
  - updated inference CLI app/service imports and downstream references in tests/docs.
  - validated with targeted inference+CLI suite and full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration wave E complete (training/analyze orchestrator relocation slice):
  - moved `smartrain/model_training_module.py` -> `smartrain/workflows/training/model_training_module.py`
  - moved `smartrain/train_resume.py` -> `smartrain/workflows/training/train_resume.py`
  - moved `smartrain/results_analyzer.py` -> `smartrain/workflows/analyze/results_analyzer.py`
  - moved `smartrain/datasets_json_former.py` -> `smartrain/workflows/datasets/datasets_json_former.py`
  - updated imports in `cli_apps`, services, tests, scripts and docs.
  - validated with targeted train/analyze/dataset/model-test suite and full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Root-package migration wave F complete (contracts/final cleanup slice):
  - moved `smartrain/interactive_contract.py` -> `smartrain/core/runtime/interactive_contract.py`
  - updated CLI/workflow imports and docs references for interactive contract access.
  - verified root package now contains only `__init__.py`, `__main__.py`, `cli.py`.
  - validated with targeted CLI/inference/train/model-test suite and full regression (`582 passed, 1 skipped`).
  Residual debt for this wave: `no residual debt`.
- 2026-05-06: Wave E-tail decomposition slice (large-module split follow-up) complete for utility boundaries:
  - extracted train interactive prompt helpers into `smartrain/workflows/training/train_options.py` and rewired `model_training_module` through compatibility wrappers (`_prompt_*`) to preserve monkeypatch behavior in tests.
  - extracted dataset scanning/parsing helpers (`find_*`, `load_obj_*`) into `smartrain/workflows/datasets/dataset_scan.py` and rewired `datasets_json_former` imports.
  - extracted analyze session-path resolver layer into `smartrain/workflows/analyze/analyze_compare_session_service.py` and rewired `results_analyzer` wrappers.
  - validated with targeted regression (`93 passed`).
  Residual debt for this slice: `no residual debt`.
- 2026-05-06: Wave E-tail decomposition slice (entrypoint boundary extraction) complete:
  - introduced workflow entry modules: `workflows/training/train_entry.py`, `workflows/analyze/analyze_entry.py`, `workflows/datasets/datasets_entry.py`.
  - switched CLI dispatch and CLI apps to call entry modules instead of directly coupling to large orchestrator modules.
  - validated with targeted CLI/workflow regression (`163 passed, 1 skipped`).
  Residual debt for this slice: `no residual debt`.
- 2026-05-06: Wave E-tail decomposition slice (analyze benchmark/plot service extraction) complete:
  - extracted inference benchmark/plot pipeline from `results_analyzer` into `smartrain/workflows/analyze/analyze_benchmark_service.py`.
  - moved run-scope/image-collection/path-resolver helpers (`resolve_selected_run_dirs`, `collect_split_images`, inference csv/png resolvers) into the same service module.
  - kept `results_analyzer` compatibility wrappers and delegated execution through explicit callback injection to preserve monkeypatchable command surface.
  - validated with targeted analyze/CLI regression (`134 passed`).
  Residual debt for this slice: `no residual debt`.
- 2026-05-06: Wave E-tail decomposition slice (analyze PR-curves service extraction) complete:
  - extracted PR-curves execution pipeline from `results_analyzer` into `smartrain/workflows/analyze/analyze_pr_curves_service.py`.
  - moved PR output resolver into the same service and switched `results_analyzer` to wrapper-based delegation with callback injection.
  - preserved command-level compatibility surface (`cmd_pr_curves`) used by workflow tests and monkeypatch-based orchestrator checks.
  - validated with targeted analyze/CLI regression (`134 passed`).
  Residual debt for this slice: `no residual debt`.
- 2026-05-07: Wave E-tail decomposition slice (analyze test-metrics service extraction) complete:
  - extracted `test-metrics-plot` orchestration into `smartrain/workflows/analyze/analyze_test_metrics_service.py`.
  - moved metrics-plot output resolver into the same service and switched `results_analyzer` to wrapper delegation.
  - preserved recompute/monkeypatch compatibility by delegating through callbacks for `latest_test_metrics_path` and `_recompute_run_test_metrics`.
  - validated with targeted analyze/CLI regression (`134 passed`).
  Residual debt for this slice: `no residual debt`.
- 2026-05-07: Wave E-tail decomposition slice (datasets_json_former report/io & normalize extraction) complete:
  - extracted scan summary writer + scan report printer + preserved-field merge logic into `smartrain/workflows/datasets/datasets_json_report_io.py`.
  - extracted `_normalize_path_for_data_path` into `smartrain/workflows/datasets/datasets_json_normalize_service.py`.
  - rewired `datasets_json_former.py` to delegate these boundaries via imports, keeping CLI behavior stable.
  - validated with targeted dataset former regression (`1 passed`) + import sanity (`test_imports`).
  Residual debt for this slice: `no residual debt`.
- 2026-05-07: Wave E-tail decomposition slice (datasets_json_former scan/index extraction) in progress -> completed for scan/index helpers:
  - added `smartrain/workflows/datasets/datasets_json_scan_index_service.py` and moved scan/index helpers there.
  - changed `datasets_json_former.py` scan/index helpers (`_sorted_diff`, `_run_scan_folder_roots`, `_load_datasets_list_file`, zip extraction, source signature, etc.) into thin delegating wrappers.
  - validated targeted dataset former regression (`1 passed`) + `test_imports` (`2 passed`).
  Residual debt for this slice: `no residual debt (scan/index helpers)`.
- 2026-05-07: Wave E-tail decomposition slice (datasets_json_former convert/normalize extraction) complete:
  - introduced `smartrain/workflows/datasets/datasets_json_convert_purge_service.py`.
  - introduced `smartrain/workflows/datasets/datasets_json_cvat11_normalize_service.py`.
  - delegated `_copy_source_to_training`, `_dataset_content_hash`, `_confirm_purge_processed_raw`, `_purge_raw_sources` into convert/purge service.
  - delegated the CVAT11-heavy part of `_ensure_training_ready_after_copy` into `datasets_json_cvat11_normalize_service.py` via callback injection.
  - validated targeted dataset former regression (`test_cvat11_native_dataset_former`) and import sanity (`test_imports`).
  Residual debt for this slice: `no residual debt (convert/normalize helpers)`.
- 2026-05-07: Wave E-tail decomposition slice (results_analyzer cmd_all - selection stage extraction) complete:
  - introduced `smartrain/workflows/analyze/analyze_all_selection_service.py`.
  - moved baseline/others/profile selection and interactive run picker from `cmd_all` into the service.
  - preserved test monkeypatch surface by injecting `_filtered_run_records`, `prompt_int`, `prompt_text`, `prompt_choice` callbacks from `results_analyzer`.
  - validated targeted analyze/CLI/import regression (`133 passed, 1 skipped`).
  Residual debt for this slice: `no residual debt (cmd_all selection stage)`.
- 2026-05-07: Wave E-tail decomposition slice (results_analyzer cmd_all - data.yaml context extraction) complete:
  - introduced `smartrain/workflows/analyze/analyze_all_data_yaml_service.py`.
  - moved `report_languages` parsing and `run_data_yaml_map` resolution/normalization branch from `cmd_all` into the service.
  - preserved prompt and auto-selection compatibility by injecting `_build_run_data_yaml_map`, `_auto_select_data_yaml`, `prompt_choice`, `prompt_text` callbacks.
  - validated targeted analyze/CLI/import regression (`133 passed, 1 skipped`).
  Residual debt for this slice: `no residual debt (cmd_all data.yaml stage)`.
- 2026-05-07: Wave E-tail decomposition slice (results_analyzer cmd_all - baseline artifacts stage extraction) complete:
  - introduced `smartrain/workflows/analyze/analyze_all_baseline_artifacts_service.py`.
  - moved cross-run baseline artifact generation (`compare`, `runs_summary`, `system_profile_compare`, `test_system_profile_compare`, `leaderboard`) from `cmd_all` into the service.
  - preserved behavior/compatibility by injecting `cmd_compare`, `cmd_export_table`, `_write_system_profile_compare_csv`, `_write_test_system_profile_compare_csv`, `cmd_leaderboard`.
  - validated targeted analyze/CLI/import regression (`133 passed, 1 skipped`).
  Residual debt for this slice: `no residual debt (cmd_all baseline artifacts stage)`.
- 2026-05-07: Wave E-tail decomposition slice (results_analyzer cmd_all - quality stage extraction) complete:
  - introduced `smartrain/workflows/analyze/analyze_all_quality_stage_service.py`.
  - moved missing-metrics recompute planning, `cmd_test_metrics_plot` invocation, and `metric_sources` load logic from `cmd_all` into the service.
  - preserved compatibility via callback injection for `_collect_missing_metrics_recompute_plan` and `cmd_test_metrics_plot`.
  - validated targeted analyze/CLI/import regression (`133 passed, 1 skipped`).
  Residual debt for this slice: `no residual debt (cmd_all quality stage)`.
- 2026-05-07: Wave E-tail decomposition slice (results_analyzer cmd_all - speed stage extraction) complete:
  - introduced `smartrain/workflows/analyze/analyze_all_speed_stage_service.py`.
  - moved speed-stage orchestration (`group_runs_by_data_yaml`, inference benchmark merge/fallback rows, leaderboard speed merge, inference plot, speed-quality artifacts, cache stats read) from `cmd_all` into the service.
  - preserved compatibility via callback injection for `_group_runs_by_data_yaml`, `cmd_inference_benchmark`, `cmd_inference_plot`, `_write_speed_quality_artifacts`, and `_record_failure`.
  - validated targeted analyze/CLI/import regression (`133 passed, 1 skipped`).
  Residual debt for this slice: `no residual debt (cmd_all speed stage)`.

- 2026-05-05: Plan-vs-code audit sync (post 7-E4 close). Historical entries below remain as change log, but continuation scope is now concentrated in: (a) final 5-E2 parity for provider-specific cls/seg outputs in real external forks, (b) Phase F / Wave 8 guardrails and anti-pattern hardening, (c) train-service decoupling from `model_training_module` (`mtm.*` coupling).
- 2026-05-05: Continuation preparation checkpoint: Wave 7 (`7-E4`) is closed in current scope; `format-compare` metrics path is canonical/unified (`canonical_gateway.load_metrics` with split support). Next execution priority should switch to Wave 8 while keeping 5-E2 residual parity items explicitly tracked.

- 2026-05-04: Extended **canonical_gateway** (PR 6.5 partial): `resolve_task_context`, `load_metrics` (test CSV → `CanonicalMetricsRef`), `load_predictions` stub. Remaining: prediction artifact contract, consumer callsites using these APIs instead of ad-hoc `metrics_reader` where policy mandates gateway-only reads.
- 2026-05-05: Migrated `results_analyzer._build_run_record_canonical` test-metrics loading to `canonical_gateway.load_metrics` (gateway API) instead of direct `read_test_metrics_row`, reducing consumer-level bypass of canonical gateway. Remaining: propagate this gateway-first metrics read pattern to other consumer flows covered by PR 6.5.
- 2026-05-05: Extended this migration within `results_analyzer` compare/recompute helper paths (`cmd_compare`, `_build_speed_quality_artifacts`, `_collect_missing_metrics_recompute_plan`) by routing test-metrics reads through shared `_read_test_metrics_for_run`, which now uses `canonical_gateway.load_metrics` in canonical mode.
- 2026-05-05: Completed PR 6.5 consumer/gateway slice in current codebase scope: `model_test_cli` task inference now uses `resolve_task_context`; `inference_cli` canonical resolution uses `resolve_task_context` + `load_target`; gateway `load_predictions` now supports split/format filtering with file discovery contract and integration tests. Remaining wave-level concern is rollout policy (flag defaults/cutover), handled by PR 6.6–6.7.
- 2026-05-05: Implemented PR 6.6 baseline migration layer: `adapters/canonical/legacy/reader.py` + `mapper.py`, CLI command `smartrain migrate canonical` (`dry-run`/`apply`/`report-only`) with JSON+Markdown reports, and migration tests for apply, dry-run safety, and idempotency.
- 2026-05-05: Expanded migration report payload with per-item `rollback_hint` and top-level `operator_guidance`; added regression test to keep guidance present for failed targets. Remaining 6.6 depth: broader historical fixture matrix.
- 2026-05-05: Applied PR 6.7 cutover policy for canonical consumers: canonical read is default, legacy read path is emergency-only via explicit policy flag (`SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK=1` + `SMARTTRAIN_CANONICAL_READ=0`); added dedicated regressions for cutover/no-legacy usage.
- 2026-05-05: Removed temporary test-level legacy bridge from `test_model_test_cli.py` and `test_results_analyzer_workflows.py`; fixtures were updated to canonical-first expectations (discoverable run models under `run/models`) and assertions aligned to post-cutover behavior.
- 2026-05-05: Started Phase D (Wave 4) implementation: added backend protocol contracts + normalized execution result envelope, and tightened capability registry requirement validation. Remaining Wave 4 debt is adapter-level rollout (`UltralyticsAdapter` + external provider normalization), tracked as 4-D3/4-D4.
- 2026-05-05: Expanded capability routing to inference (`resolve_infer_backend`) and connected `inference_service` to capability policy checks. Runtime backend mismatch is currently warning-only to preserve compatibility with existing inference backend factory behavior; revisit after dedicated `UltralyticsAdapter` extraction in PR 4.3.
- 2026-05-05: Added reference `UltralyticsAdapter` and routed local inference backend creation through it. Remaining Phase D debt concentrates in PR 4.4: normalize external providers under the same adapter contract (train/test/inference entrypoints still use provider-specific wiring).
- 2026-05-05: Completed PR 4.4 in current codebase scope: `ExternalProviderAdapter` now normalizes external inference and external train execution wiring; `train_service` uses adapter contract with injected legacy runners for compatibility.
- 2026-05-05: Upgraded `services/test_backend_dispatch.py` from strategy function-map to explicit strategy objects (`PtStrategy`, `PtUniStrategy`, `NonPtNativeStrategy`) while keeping callable monkeypatch compatibility in registry dispatcher.
- 2026-05-05: Phase E (5-E1) foundation completed: task execution context + task metrics adapter framework wired into canonical gateway metrics path. Remaining Phase E debt: full classification/segmentation consumer flows and analyze-layer decomposition (`3-E3`), then artifact schema v2 migration (`7-E4`).
- 2026-05-05: Closed Wave 7 (7-E4) in current scope: `services/analyze_format_compare_service.py` now reads metric values via `canonical_gateway.load_metrics` (canonical/unified read) with `split` support; removed legacy CSV parsing dependency from the analyze format-compare path. Added/validated regression coverage for `format_compare` output generation.
- 2026-05-05: Added task-aware guard for internal `pt_uni` compare path in model testing: detection-only behavior is now explicit; classification/segmentation routes skip this internal compare branch cleanly.
- 2026-05-05: Extended task-aware consumer wiring into inference runtime: capability resolution now uses normalized CLI task hint (`--task`) instead of hardcoded detection-only routing. Remaining 5-E2 debt: task-specific runtime adapters/outputs are still largely shared and detection-centric.
- 2026-05-05: Reduced detection-centric behavior in backend adapter result contracts: `UltralyticsAdapter.infer` and `ExternalProviderAdapter.infer` now propagate normalized task type from request context into `BackendExecutionResult`. Remaining 5-E2 debt: runtime prediction/report payloads and format-specific execution paths are still mostly shared and not yet task-specialized end-to-end.
- 2026-05-05: Reduced detection-centric behavior in inference artifact schema path: base report now carries normalized `task_type`, and `artifact_schema_v2` derives `task_type` and metrics namespace from payload task context. Also fixed normalization edge case where canonical values (`segmentation`/`detection`) were collapsing to detection-only default.
- 2026-05-05: Continued 5-E2 inference runtime wiring at backend-factory level: `InferenceBackendRegistry.create_local_backend(...)` now validates capabilities against the incoming normalized task type (instead of fixed detection), and `UltralyticsAdapter` forwards task context to this policy check. Remaining 5-E2 debt: inference output shape is still detection-centric (`BackendPrediction.detections`), so cls/seg need task-specific prediction/result adapters for full end-to-end readiness.
- 2026-05-05: Continued 5-E2 output contract migration: `BackendPrediction` now carries task-aware payload (`task_type`, `outputs`) and keeps compatibility bridge (`detections` property). `inference_service` passes normalized task type into runtime `predict(...)`. Remaining 5-E2 debt: add concrete cls/seg task adapters that populate non-detection payloads (classification labels/top-k, segmentation masks/polygons) and migrate report writers to consume generic task outputs directly.
- 2026-05-05: Continued 5-E2 output migration end-to-end for inference report writer: runtime now emits task-specific outputs (`classification.top1/top_k`, `segments`) and consumer writes unified `images[].task_outputs` with task-aware summary counter (`task_outputs_total`). Remaining 5-E2 debt: external provider inference path still returns minimal/no task-specific payload and needs the same unified output contract to fully close cls/seg consumer wiring.
- 2026-05-05: Reduced remaining external consumer gap in 5-E2: external provider inference report now follows unified summary shape (`images_input/images_processed/images_skipped/detections_total/task_outputs_total` with numeric zero defaults) and keeps `images=[]` explicit. Remaining 5-E2 debt: no per-image structured task outputs are available from external providers yet (provider API/runtime contract extension required).
- 2026-05-05: Added external provider structured inference contract on consumer side: external runner return can now include `images[].task_outputs` and `return_code`; `inference_service` normalizes this payload into the same task-aware report shape used by local backends. Remaining 5-E2 debt: provider-specific launchers still mostly return rc-only; production providers need gradual adoption of structured payload contract for full cls/seg parity.
- 2026-05-05: Reduced provider-side gap for 5-E2: external infer adapters now pass `--result-json`, runner loads structured payload from launcher output, and default infer launchers (`mp_infer_launcher`, `mfel_infer_launcher`) emit `images[].task_outputs` with detections. Remaining 5-E2 debt: richer non-detection payloads (classification top-k / segmentation polygons from external providers) still depend on provider-specific model/task support and launcher extensions.
- 2026-05-05: Extended provider-side task richness for external infer launchers: `--task` is propagated end-to-end and structured output builders now emit classification (`top1`/`top_k`) and segmentation (`polygon_roi_xy`) payloads when runtime predictions expose required fields. Remaining 5-E2 debt: not all provider forks/models expose `probs`/`masks` consistently, so some external runs still legitimately produce detection-only or empty task-specific payloads.
- 2026-05-05: Closed Phase D residual policy debt: inference capability mismatch is no longer warning-only; `inference_service` now aborts on incompatible runtime backend id, with explicit alias mapping for expected runtime naming variants (`ultralytics:engine/trt/onnx`).
- 2026-05-05: Continued train-coupling reduction with compatibility-safe helper extraction: neutral runtime helpers moved to `services/train_runtime_helpers.py` (`build_run_name`, `resolve_external_eval_source`, `json_safe_train_summary`, `load_batch_from_training_metadata`); `train_service` now imports these directly while keeping monkeypatch-compatible fallback to `mtm._*` symbols.
- 2026-05-05: Deepened train-coupling reduction: additional external/train runtime primitives moved to shared helper layer (`normalize_external_run_layout`, `ensure_external_best_checkpoint_layout`, `write_external_fallback_metrics`, `run_mfel_external_val_fallback`, `maybe_free_cuda_memory`), and `train_service` now uses direct module imports for provider/catalog/hash/recommendation/runtime helpers with compatibility fallbacks for patched `mtm` symbols.
- 2026-05-05: Closed current deep train-coupling slice: `train_service` business path no longer calls `mtm.*` directly; runtime operations are routed through `_MtmRuntimeOps` adapter (`train_yolo`, `test_yolo`, `save_training_metadata`, `collect_system_profile`, legacy helper fallbacks), preserving monkeypatch compatibility while isolating service logic from module namespace coupling.
- 2026-05-05: Finalized pre-Phase-8 train-coupling closure in current scope: `_MtmRuntimeOps` is now callable-composition based (explicit function injection at composition root) instead of namespace access; service layer depends on runtime operations contract, and `model_training_module` wiring is confined to a single assembly point.
- 2026-05-05: Phase 8 bootstrap (8-F2) started: added repository PR checklist template (`.github/pull_request_template.md`) aligned with clean-code policy and refactor debt hygiene, plus regression guard `tests/regression/test_train_service_guardrails.py` to prevent accidental reintroduction of direct `mtm.*` usage outside composition root.
- 2026-05-05: Phase 8 anti-pattern cleanup (8-F1, slice #1): removed runtime `hasattr(...)` checks from external provider launchers (`mp_infer`, `mfel_infer`, `mfel_val`, `mfel_train`) in favor of explicit `getattr`/`callable` guards; added regression test `tests/regression/test_phase8_no_runtime_hasattr.py` to block `hasattr` usage in runtime-critical modules (`train_service`, `inference_service`, target launchers).
- 2026-05-05: Phase 8 anti-pattern cleanup (8-F1, slice #2): reduced duplication in `train_service` by centralizing confidence recommendation argument parsing/config assembly into shared helpers (`_confidence_recommendation_params`, `_confidence_recommendation_config`), keeping behavior identical and reducing copy-paste across external/builtin/test-only flows.
- 2026-05-05: Phase 8 guardrails baseline (`8-F2`) is now active in CI: added workflow `.github/workflows/phase8-guardrails.yml` and policy checker `scripts/ci/check_phase8_guardrails.py` to enforce no-`hasattr` rule for runtime-critical modules and composition-root-only `mtm.*` wiring in `train_service`.
- 2026-05-05: Phase 8 anti-pattern cleanup (8-F1, slice #3): reduced duplicated report-building/write code in `inference_service` local path by introducing `_write_local_inference_report(...)` and reusing it for initial report seed + skip-update + per-image update; this keeps CLI/runtime behavior intact while tightening layered orchestration boundaries.
- 2026-05-05: Phase 8 anti-pattern cleanup (8-F1, slice #4): reduced `run_inference_job` branch complexity by extracting external-provider preflight/migration helpers (`_apply_external_provider_inference_from_refs`, `_validate_external_inference_model_or_fail`) so argument normalization and provider model validation are explicit service helpers instead of inline branch logic.
- 2026-05-05: Started Wave 3-E3 structural split for `results_analyzer`: extracted session/artifact path helpers, data-yaml discovery, and table workflows (`scan`/`export-table`) into dedicated service modules with compatibility wrappers in `results_analyzer.py`. Remaining 3-E3 debt: migrate compare/interactive/report-artifact orchestration into dedicated service modules to complete full args/interactive/service/backends/artifacts symmetry.
- 2026-05-05: Continued Wave 3-E3 split: compare execution workflow moved to `services/analyze_compare_service.py` and wired from `results_analyzer.cmd_compare`. Remaining 3-E3 debt now concentrates on interactive/session report orchestration and large artifact builders (`format-compare`, recommendation/speed-quality bundles).
- 2026-05-05: Continued Wave 3-E3 split: moved recommendation/speed-quality artifact builders to `services/analyze_artifact_builders.py` and kept compatibility wrappers in `results_analyzer`. Remaining 3-E3 debt: extract `format-compare` builder and interactive orchestration/session assembly into dedicated services to finish symmetry.
- 2026-05-05: Continued Wave 3-E3 split: moved `format-compare` artifact builder into `services/analyze_format_compare_service.py`; `results_analyzer` now delegates via wrapper. Remaining 3-E3 debt is mainly interactive/session orchestration decomposition.
- 2026-05-05: Continued Wave 3-E3 split: moved interactive orchestration flow from `cmd_interactive` into `services/analyze_interactive_service.py` with dependency injection of compare/quality/speed/full runners. Remaining 3-E3 debt is now mostly residual session/report assembly in `results_analyzer` and possible further split of leaderboard/benchmark helpers.
- 2026-05-05: Completed planned 3-E3 scope: leaderboard orchestration (including speed fallback resolution from performance artifacts) moved into `services/analyze_leaderboard_service.py`; `results_analyzer` now primarily coordinates dedicated service modules.
- 2026-05-04: Initial **canonical write** slice (PR 6.4 phase A): `smartrain/adapters/canonical/write/*`, `canonical_gateway.persist_canonical_snapshot`, optional dual-write helper `run_dual_write`. **Opt-in** snapshot after successful test artifact persist via `SMARTTRAIN_CANONICAL_WRITE=1` (best-effort warning on failure). Remaining PR 6.4 scope: richer manifest/provenance, real legacy writer hook in dual-write, hash coverage for individual artifact files per plan.

## Open Items

- [x] **P0 / Wave 8 (8-F1):** системный anti-pattern pass по правилам из `docs/refactor/07-clean-code-rules.md` (runtime-critical модули: `services/inference_service.py`, `services/train_service.py`, external launchers). Notes: закрыто коммитом `8ed547c`.
- [x] **P1 / Wave 8 (8-F3):** удалить legacy-ветки/мосты по deprecation-policy (`docs/refactor/06-deprecation-and-alias-policy.md`) только после явных gate-условий и регрессий. Notes: закрыто коммитами `2f6f19d` + `8e9455a` (и релиз-ноты/депрекейшн подготовлены коммитом `9d605dc`).

## Next Execution Backlog (ready-to-start)

1. Wave 8 bootstrap PR: добавить guardrails/checklists + минимальный CI quality gate.
2. Anti-pattern PR #1: runtime-critical cleanup (`inference_service` + external launchers), без функциональных изменений.
3. Legacy-removal PR (8-F3): удалить готовые мосты только после прохождения новых guardrails и explicit deprecation gates.

- 2026-05-05: Closed 5-E2 residual debt in current scope: external inference now enforces stable degraded task contract for provider capability gaps (`classification: {}`, `segments: []`, detection fallback list), with regression coverage for empty-but-valid cls/seg payloads in `tests/test_inference_cli.py`.

## Notes

- 2026-05-04: Added as persistent debt register per refactor process requirement. Update this file during each meaningful refactor step.
- 2026-05-04: Introduced `smartrain/backends/train_test_registry.py` and wired `model_test_orchestrator` to resolve backend IDs via capability registry for test persistence paths. This is still detection-only and does not yet cover full train flow routing.
- 2026-05-04: Extended capability routing into `train_service` metadata (`training_provider` for local/test-only flows now resolved via `resolve_train_backend`). Remaining gap: runtime execution path selection is still hardcoded by format branches, not backend strategy dispatch.
- 2026-05-04: `model_test_orchestrator` now uses centralized non-PT dispatch helper (`_run_non_pt_format`) + unified backend resolver (`_backend_for`). Runtime logic is cleaner, but still closure-based inside one function; next cleanup is extracting dispatch strategies into dedicated module-level objects.
- 2026-05-04: Completed extraction to module-level strategy helper `smartrain/services/test_backend_dispatch.py`; orchestrator now calls dispatch service directly. Also added task-aware routing input for `model test` via `--task` (with metadata inference fallback) and generalized train/test registry to `KNOWN_TASKS`.
- 2026-05-04: Extracted external-provider training scenario into `_run_external_provider_flow(...)` inside `train_service`; behavior preserved, and main runner now delegates instead of embedding the full branch inline.
- 2026-05-04: Completed split of remaining `train_service` logic into `_run_builtin_train_and_eval_flow(...)` and `_run_test_only_flow(...)`; `run_train_after_setup(...)` now acts as a thin scenario router.
- 2026-05-04: Plan-vs-codebase audit added explicit debt markers for partially completed waves (orchestrator-split, backend-abstraction, task-abstraction) to track remaining scope hidden behind completed todo statuses.
- 2026-05-04: Started Wave 6 implementation track: added canonical domain package (`domain/canonical/*`) with DTO + validation and initial read adapters (`adapters/canonical/read/*`) including factory and equivalence tests.
- 2026-05-04: Added `orchestrators/canonical_gateway.py` and first staged consumer wiring in `model_test_cli` for task inference via feature flag `SMARTTRAIN_CANONICAL_READ=1` (with safe fallback to legacy metadata read path).
- 2026-05-04: Added second staged consumer wiring in `inference_cli` model resolution (`--run`/`--model-name`) via `canonical_gateway` under `SMARTTRAIN_CANONICAL_READ=1`, preserving legacy fallback behavior.
- 2026-05-04: Updated canonical consumer migration policy to no-fallback mode for new canonical paths: when `SMARTTRAIN_CANONICAL_READ=1`, `model_test_cli` task inference and `inference_cli` model resolution rely on canonical gateway without legacy fallback.
- 2026-05-04: Added initial canonical consumer wiring in `results_analyzer.cmd_scan` under `SMARTTRAIN_CANONICAL_READ=1` (no-fallback path, explicit error reporting per run row).
- 2026-05-04: Extended canonical read path in `results_analyzer` for `_filtered_run_records` and `cmd_leaderboard` under `SMARTTRAIN_CANONICAL_READ=1` via shared `_build_run_record_canonical`. Remaining gap: report helper branches (for example `_build_abbreviations_for_report` and artifact collectors) still read legacy metadata directly.
- 2026-05-04: Extended canonical read path to additional report/helper branches in `results_analyzer` (`_build_abbreviations_for_report`, `_collect_ultralytics_test_artifacts`, `_collect_confidence_recommendation_tables`) under `SMARTTRAIN_CANONICAL_READ=1`. Remaining gap: several export/compare helpers still use direct `load_metadata/flatten_metadata` and are not yet switched to canonical source.
- 2026-05-04: Extended canonical read path to export/system-profile helpers in `results_analyzer` (`cmd_export_table`, `_write_system_profile_compare_csv`) under `SMARTTRAIN_CANONICAL_READ=1` via shared `_flat_row_canonical`. Remaining gap: ` _write_test_system_profile_compare_csv` and some compare helpers still read `load_metadata/flatten_metadata` directly.
- 2026-05-04: Extended canonical read path to `_write_test_system_profile_compare_csv` under `SMARTTRAIN_CANONICAL_READ=1`; model/dataset columns now come from canonical payload in test system-profile export too. Remaining gap: `_collect_data_yaml_candidates_for_run` and selected legacy-only helper branches still read `training_metadata.json` directly and need staged canonical replacement/adapter support.
- 2026-05-04: Switched `_collect_data_yaml_candidates_for_run` to canonical mode under `SMARTTRAIN_CANONICAL_READ=1` (dataset `name` source now from canonical payload, no direct `load_metadata` calls in this mode). Remaining gap: canonical adapter still derives `dataset_ref` from run metadata internally, so full metadata-decoupling requires expanding canonical run adapter fields/producers.
- 2026-05-04: Reduced canonical run adapter dependency on `training_metadata.json` for dataset identity: `RunAdapter` now infers `dataset_ref` from run path (`.../runs/<dataset>/<run>`) when metadata is missing. Remaining gap: task/backend values still default from metadata-or-defaults and need richer canonical producers for full decoupling.
- 2026-05-04: Reduced canonical run adapter dependency for task/backend metadata too: when metadata is missing, `RunAdapter` infers `task_type` from target model naming (`-cls`/`-seg`) and `backend_type` from model format (`onnx`/`engine`/`trt`). Remaining gap: inference quality still depends on naming conventions and should be replaced by explicit canonical producer fields.
- 2026-05-04: Reduced `results_analyzer` duplication by introducing unified run row access (`_flat_row_for_run`) with canonical/legacy switching in one place; scan/export/system-profile/recommendation consumers now reuse it instead of repeating branch-specific metadata logic.
- 2026-05-04: Moved PT and internal `pt_uni` test execution from `model_test_orchestrator` into `test_backend_dispatch` (`run_pt_test_backend`, `run_internal_pt_uni_backend`) for symmetrical backend dispatch next to non-PT paths.
- 2026-05-05: Upgraded `services/test_backend_dispatch.py` from plain helper calls to a central format registry dispatcher (`run_test_backend_via_registry` + `TestBackendDispatchContext`), keeping wrapper compatibility while reducing orchestration branching and easing future backend additions.
