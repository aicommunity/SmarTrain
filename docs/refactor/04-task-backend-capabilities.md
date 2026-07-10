# Task/Backend Capability Matrix

См. также контракт внутреннего сравнения **`pt_uni`**: [`14-pt-uni-compare-contract.md`](./14-pt-uni-compare-contract.md).

Rollout instance segmentation: [`tech-debt-instance-segmentation.md`](./tech-debt-instance-segmentation.md).

## Task Types

- `detection` — production-ready (Ultralytics PT + native ONNX/TRT test)
- `classification` — adapter-ready (Ultralytics PT; native test limited)
- `segmentation` — adapter-ready (Ultralytics PT train/test/infer; native ONNX/TRT test **limited** — guard/skip)

## Backend Types

- `ultralytics` (current default)
- `external:<provider>` (train: detection-first; infer: task propagation for cls/seg)
- `onnxruntime` (native test: detection-shaped eval; segmentation skipped by guard)
- `tensorrt` (native test: detection-shaped eval; segmentation skipped by guard)

## Matrix (current)

| Task | Train | Test | Inference |
|---|---|---|---|
| detection | ultralytics, external | ultralytics PT, native ONNX/TRT | ultralytics, external |
| classification | ultralytics, external (limited) | ultralytics PT, pt_uni; native limited | ultralytics, external (degraded contract possible) |
| segmentation | ultralytics PT (**ready**), external (**limited**) | ultralytics PT + pt_uni (**ready**); native ONNX/TRT (**skipped**) | ultralytics PT (**ready**), external (polygon when provider supports masks) |

## Capability Contract

- `supports(task_type, model_format) -> bool`
- `train/evaluate/predict` interfaces return normalized result payloads.
- Task-aware guards: native non-PT test for `segmentation` is skipped with explicit `capability_gap` (see Operational Limits in [`09-tech-debt.md`](./09-tech-debt.md)).
