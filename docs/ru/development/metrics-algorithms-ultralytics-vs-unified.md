# Алгоритмы метрик: Ultralytics vs Unified

## Область применения
Документ описывает расчет метрик детекции в двух вариантах:
- нативная валидация Ultralytics (`model.val()`),
- unified-оценка в smart-train (`pt_uni`, `onnx`, `engine`, `trt`).

Цель: методическая эквивалентность, чтобы расхождения метрик отражали backend/модель, а не разницу формул.

## Базовые определения
- **TP**: предсказание сопоставлено с GT того же класса при IoU >= порога.
- **FP**: предсказание не сопоставлено ни с одним GT.
- **FN**: GT остался без сопоставленного предсказания.
- **Precision**: `P = TP / (TP + FP)`.
- **Recall**: `R = TP / (TP + FN)`.
- **F1**: `F1 = 2PR / (P + R)`.

## Алгоритм сопоставления (matching)
Для каждого изображения:
1. Строится матрица IoU `IoU[gt, pred]`.
2. Обнуляются пары с разными классами.
3. Для каждого порога IoU в `0.50, 0.55, ..., 0.95`:
   - берутся пары `IoU >= threshold`;
   - пары сортируются по убыванию IoU;
   - применяется one-to-one дедупликация (уникальный pred и уникальный GT);
   - отмечаются корректные детекции для данного порога.

Итог: булев массив `correct` размера `[N_pred, 10]`.

## AP и mAP
Для каждого класса и каждого IoU-порога:
1. Предсказания сортируются по confidence (убывание).
2. Строятся кумулятивные TP/FP.
3. Формируется precision-recall кривая.
4. AP вычисляется как COCO 101-point interpolation:
   - precision envelope `mpre`;
   - сетка `x = linspace(0, 1, 101)`;
   - `AP = trapz(interp(x, mrec, mpre), x)`.

Агрегации:
- `mAP50`: средний AP по классам при IoU=0.50.
- `mAP50-95`: средний AP по классам и порогам IoU 0.50..0.95.

## Выбор рабочей точки для Precision/Recall/F1
Используется сетка confidence `x = linspace(0, 1, 1000)`:
- строятся class-wise кривые `P(x)`, `R(x)`, `F1(x)`;
- глобальная точка выбирается по максимуму сглаженного среднего `F1`;
- `Box-P`, `Box-R`, `Box-F1` считаются как средние по классам в этой точке.

## Кривые и артефакты
- `pr.csv`: средняя по классам precision от recall (источник PR-кривой в стиле Ultralytics).
- `BoxPR_curve.png`, `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`: строятся из Ultralytics-совместимых массивов кривых.
- `pr_per_class.csv`: семплы PR и AP по классам.

## Дефолты и параметры оценки
Для выравнивания с Ultralytics:
- `conf = 0.001`
- `iou = 0.7` (NMS)
- IoU-grid AP: `0.50:0.05:0.95`.

Параметры `imgsz`, `conf`, `iou` сохраняются в артефактах и выводятся в отчете.

## Provenance для сравнения форматов (раздел 4)
В сравнения добавлены поля происхождения расчета:
- `inference_source`
- `gt_source`
- `nms_profile`

Они пишутся в:
- `artifacts/format_compare/format_metrics_compare_*.csv`
- `artifacts/format_compare/format_eval_settings.csv`

Эти поля нужны, чтобы отделять backend-эффекты от методологических расхождений.

## Различия: было / стало / неизбежный остаток

### Было (legacy unified)
- `pt_uni` считался через `predict` по изображениям, а `pt` через `val`-validator.
- GT читался упрощенным парсером без полной валидации Ultralytics.
- NMS/matching/AP могли расходиться по деталям реализации.
- В результате различия `pt` vs `pt_uni` были частично методологическими.

### Стало (validator-style unified)
- `pt_uni` переведен на validator-style путь (`model.val`) с теми же `imgsz/conf/iou`.
- Для non-pt форматов сохранен единый metric core с Ultralytics-совместимыми `ap_per_class` и IoU-grid.
- GT собирается через Ultralytics-проверку `verify_image_label`.
- В отчетах раздела 4 фиксируется provenance (`inference_source`, `gt_source`, `nms_profile`).

### Неизбежный остаток (backend-эффекты)
- Различия численной точности/ядра инференса: PyTorch vs ONNX Runtime vs TensorRT.
- Различия экспортных графов и runtime-плагинов.
- Нестабильность отдельных backend (OOM/runtime/crash), отражаемая в `format_compare_issues.json`.

## Чеклист диагностики расхождений
Если `pt` и `pt_uni` расходятся, проверять по порядку:
1. `inference_source`: ожидается validator-style для `pt` и `pt_uni`; и backend-specific источник для `onnx/engine/trt`.
2. `nms_profile`: сравнивать профили validator/NMS между форматами.
3. `gt_source`: должен быть один и тот же валидированный источник GT.
4. Совпадение параметров `imgsz/conf/iou`.
5. `format_compare_issues.json` (OOM/runtime/missing artifacts).

## Ссылки на реализацию
- Unified в проекте:
  - `smartrain/model_test_backends.py`
  - `_build_ultralytics_style_stats()`
  - `_compute_ultralytics_style_payload()`
  - `_write_native_eval_artifacts()`
- Ultralytics:
  - `ultralytics/utils/metrics.py` (`ap_per_class`, `compute_ap`)
  - `ultralytics/engine/validator.py` (`match_predictions`)
  - `ultralytics/models/yolo/detect/val.py` (валидационный pipeline и NMS)
