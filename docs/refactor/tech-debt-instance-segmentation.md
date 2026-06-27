# Tech debt: instance segmentation rollout

Companion checklist: [`10-implementation-checklist.md`](./10-implementation-checklist.md) (section «Instance Segmentation Rollout»).

## Register

| id | status | context | deferred_decision | resolution |
|----|--------|---------|-------------------|------------|
| TD-SEG-001 | DONE | `dataset_augment.py` | Polygon-aware augment: Albumentations keypoints vs custom polygon warp | `yolo_augment_geom.py` + keypoint/bbox Compose; flip/rotate/photometric for polygons. |
| TD-SEG-002 | DONE | `native_eval.py` | Mask eval для ONNX/TRT vs permanent skip + guard | Interim: skip + `capability_gap` in manifest; `--force-native-seg-test` experimental. Native mask eval WONTFIX. |
| TD-SEG-003 | DONE | `ultralytics_test_contract.py` | Включать ли `Mask*` plots в completeness contract | Box* required; `ULTRALYTICS_PT_MASK_PLOTS_OPTIONAL` recommended-not-required. |
| TD-SEG-004 | DONE | `METRIC_AGG_COLUMNS` | Единый список vs task-aware discovery из CSV headers | `tasks/metric_columns.py` + fallback in format_compare. |
| TD-SEG-005 | DEFERRED | `dataset_former.py` dedup | Polygon IoU vs bbox-enclosing IoU | v1: exact duplicate strings only; polygon IoU deferred. |
| TD-SEG-006 | DONE | `weighted_yolo_dataset.py` | Count по polygon instances vs enclosing bbox area | Ultralytics `label["cls"]` counts instances; augment `count_label_instances` for polygons. |
| TD-SEG-007 | DONE | External providers train | Когда провайдеры получат `-seg` catalog | `--task` propagated via adapters/runner/launchers; MFEL custom configs may still lack seg heads. |
| TD-SEG-008 | DONE | E2E train segment | Real weights fixture в CI vs mock-only | `tests/integration/test_train_segment_smoke.py` (task alias/metadata); full weights `@pytest.mark.slow` WONTFIX for CI. |
| TD-SEG-009 | DONE | `bbox-copy` augment | Поддержка polygon copy-paste или WONTFIX | WONTFIX: explicit error when `--enable-bbox-copy` on polygon datasets. |
| TD-SEG-010 | WONTFIX | COCO import | Нужен ли конвертер COCO→YOLO polygon для Ultralytics | Out of Ultralytics-first scope; CVAT polygon import/export added instead. |

## Burn-down checklist

- [x] TD register initialized (Stage 0)
- [x] TD-SEG-001
- [x] TD-SEG-002
- [x] TD-SEG-003
- [x] TD-SEG-004
- [x] TD-SEG-005 (DEFERRED)
- [x] TD-SEG-006
- [x] TD-SEG-007
- [x] TD-SEG-008
- [x] TD-SEG-009
- [x] TD-SEG-010 (WONTFIX)
- [x] Rollout stages SEG-0…SEG-8

## Stage closure log

- 2026-06-27 Stage 0: initialized TD register (TD-SEG-001…010 OPEN); companion checklist added to `10-implementation-checklist.md`.
- 2026-06-27 Stage 1: polygon safety-net tests added (yolo_labels, rotate, orient, ROI, report); no hidden bugs in rotate/ROI.
- 2026-06-27 Stage 2: EN/RU docs + native ONNX/TRT skip guard for segmentation; TD-SEG-002 interim DONE.
- 2026-06-27 Stage 3: segmentation metrics adapter + optional Mask plots; TD-SEG-003 DONE; TD-SEG-004 deferred to Stage 6.
- 2026-06-27 Stage 4: polygon-aware augment; TD-SEG-001 DONE; TD-SEG-009 WONTFIX.
- 2026-06-27 Stage 5: dataset_stats bbox/polygon counts; CVAT polygon import/export; TD-SEG-005 DEFERRED; TD-SEG-006 DONE.
- 2026-06-27 Stage 6: task-aware metric columns; TD-SEG-004 DONE.
- 2026-06-27 Stage 7: inference `--save-overlay` + viz service; tests added.
- 2026-06-27 Stage 8: external `--task segment` propagation; task smoke tests; final burn-down (TD-SEG-007/008/010).
