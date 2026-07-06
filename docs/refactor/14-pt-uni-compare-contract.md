# Internal `pt_uni` compare (PT vs unified PT runner)

## Scope

- Triggered from [`model_test_runner`](../../smartrain/services/testing/model_test_runner.py) when format `pt` is in the test plan and `task_type` is one of **`detection`**, **`classification`**, **`segmentation`** (normalized via [`task_to_metadata_task_type`](../../smartrain/core/training/train_profile.py)).
- Produces the internal format **`pt_uni`** artifacts alongside ordinary `pt` test metrics, using [`run_internal_pt_uni_backend`](../../smartrain/services/test_backend_dispatch.py) → [`run_native_format_backend`](../../smartrain/workflows/testing/model_test_backends.py) → Ultralytics `model.val(...)`.

## Task-specific behavior

| Task | Ultralytics `val` task | Notes |
|------|------------------------|--------|
| detection | default / detect | Same as historical behavior. |
| classification | `classify` | Passed via `task_type` on the test dispatch context. |
| segmentation | `segment` | Passed via `task_type` on the test dispatch context. |

## Artifacts

- Metrics CSV and test state follow the same layout as other model-test formats (`persist_target_test_artifacts_state` for `pt_uni`).
- **Deep diagnostics** (optional) still assume box-style predictions where applicable; for cls/seg, run without `--deep-diagnostics` if the environment lacks memory for box-heavy paths.

## Non-goals

- `pt_uni` is not a user-facing inference artifact format (see CLI inference docs).
- This document does not define public prediction bundle paths; see [`05-artifact-schema-v2.md`](./05-artifact-schema-v2.md) and [`05b-run-model-canonical-schema.md`](./05b-run-model-canonical-schema.md).
