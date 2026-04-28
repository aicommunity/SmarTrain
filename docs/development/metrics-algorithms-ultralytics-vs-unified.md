# Metrics Algorithms: Ultralytics vs Unified

## Scope
This document describes how detection metrics are computed in:
- Ultralytics native validation (`model.val()`).
- Smart-train unified evaluation pipeline (`pt_uni`, `onnx`, `engine`, `trt`).

The goal is methodological equivalence, so metric differences reflect backend/model behavior rather than evaluation math differences.

## Core Definitions
- **TP**: prediction matched to one GT of the same class with IoU >= threshold.
- **FP**: prediction not matched to any GT at threshold.
- **FN**: GT not matched by any prediction.
- **Precision**: `P = TP / (TP + FP)`.
- **Recall**: `R = TP / (TP + FN)`.
- **F1**: `F1 = 2PR / (P + R)`.

## Matching Algorithm
For each image:
1. Build IoU matrix `IoU[gt, pred]`.
2. Zero-out entries where classes differ.
3. For each IoU threshold in `0.50, 0.55, ..., 0.95`:
   - keep candidate pairs with `IoU >= threshold`;
   - sort candidates by descending IoU;
   - enforce one-to-one assignment (unique prediction, unique GT);
   - mark matched predictions as correct for this threshold.

The resulting `correct` tensor has shape `[N_pred, 10]`.

## AP and mAP
Per class and per IoU threshold:
1. Sort predictions by confidence descending.
2. Build cumulative TP/FP arrays.
3. Build precision-recall curve.
4. Compute AP using COCO 101-point interpolation:
   - precision envelope `mpre`;
   - interpolation grid `x = linspace(0, 1, 101)`;
   - `AP = trapz(interp(x, mrec, mpre), x)`.

Aggregates:
- `mAP50`: mean AP over classes at IoU=0.50.
- `mAP50-95`: mean AP over classes and IoU thresholds 0.50..0.95.

## Precision/Recall/F1 Operating Point
Ultralytics uses confidence grid `x = linspace(0, 1, 1000)`:
- class-wise `P(x)`, `R(x)`, `F1(x)` are built from sorted detections;
- global operating point is selected at max smoothed mean `F1`;
- reported box metrics are class-mean values at that point:
  - `Box-P` = mean class precision;
  - `Box-R` = mean class recall;
  - `Box-F1` = mean class F1.

## Curves and Artifacts
- `pr.csv`: mean precision over classes against recall grid (Ultralytics-style PR source).
- `BoxPR_curve.png`, `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`: produced from Ultralytics-compatible curve arrays.
- `pr_per_class.csv`: per-class PR samples and AP.

## Defaults and Evaluation Parameters
To stay aligned with Ultralytics validation defaults:
- `conf = 0.001`
- `iou = 0.7` (NMS threshold)
- AP IoU grid fixed to `0.50:0.05:0.95`.

`imgsz`, `conf`, `iou` are persisted and exposed in report comparison metadata.

## Implementation References
- Unified evaluator: `smartrain/model_test_backends.py`
  - `_build_ultralytics_style_stats()`
  - `_compute_ultralytics_style_payload()`
  - `_write_native_eval_artifacts()`
- Ultralytics reference:
  - `ultralytics/utils/metrics.py` (`ap_per_class`, `compute_ap`)
  - `ultralytics/engine/validator.py` (`match_predictions`)
  - `ultralytics/models/yolo/detect/val.py` (validation postprocess and NMS flow)
