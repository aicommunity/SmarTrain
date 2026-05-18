# Task/Backend Capability Matrix

См. также контракт внутреннего сравнения **`pt_uni`**: [`14-pt-uni-compare-contract.md`](./14-pt-uni-compare-contract.md).

## Task Types

- `detection`
- `classification` (planned)
- `segmentation` (planned)

## Backend Types

- `ultralytics` (current default)
- `external:<provider>` (already present for some flows)
- `onnxruntime` (planned adapter)
- `tensorrt` (planned adapter)

## Matrix (Target)

| Task | Train | Test | Inference |
|---|---|---|---|
| detection | ultralytics, external | ultralytics, native backends | ultralytics, external |
| classification | adapter-ready | adapter-ready | adapter-ready |
| segmentation | adapter-ready | adapter-ready | adapter-ready |

## Capability Contract

- `supports(task_type, model_format) -> bool`
- `train/evaluate/predict` interfaces return normalized result payloads.
