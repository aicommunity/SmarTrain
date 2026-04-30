> English version: [../../cli/inference.md](../../cli/inference.md)

# CLI: inference

`smartrain inference` запускает инференс детекции объектов по папке или по split датасета.

Команда пишет:

- `inference_results.json` (основной отчет)
- `environment_profile.json` (профиль окружения машины и рантайма)

Оба файла сохраняются в:

- `workspace/inference/<model>/<timestamp-source>/`

## Поддерживаемые типы моделей

Локальные артефакты моделей:

- `pt`
- `onnx`
- `engine`
- `trt`

Внешние ссылки моделей:

- provider-scoped refs вида `provider:model` (`--weights` / `--model-name`) через flow с `--external-provider`.

Примечание: `pt_uni` — внутренний режим сравнения метрик (PT vs PT-uni, test/val), а не пользовательский тип модели для инференса.

## Быстрые примеры

```bash
smartrain inference --model-name my_model --data-mode folder --source-dir ./images --device cpu
smartrain inference --weights ./runs/ds/run_001/models/run_001.engine --data-mode folder --source-dir ./images
smartrain inference --weights dr-yolo:yolov8n --external-repo /opt/dr-yolo --data-mode folder --source-dir ./images
```

## Контракт метрик производительности

Отчет содержит dual-профиль в `performance`:

- `performance.end_to_end` - image I/O + preprocessing + вызов inference + postprocessing + обновление отчета
- `performance.infer_only` - тайминг только backend inference call
- `performance.stage_breakdown_ms` - стадийные тайминги, если backend их отдает
- `performance.methodology` - контекст замера и caveats

Формат статистик latency (для `end_to_end` и `infer_only`):

- `images_total`
- `warmup_images`
- `duration_s`
- `throughput_img_s`
- `latency_ms.all` с `count/mean/p50/p90/p95/p99/min/max/std`
- `latency_ms.steady` с теми же полями после исключения warmup

## Контракт артефактов

Секции верхнего уровня `inference_results.json`:

- `created_at`
- `workspace`
- `model`
- `parameters`
- `source`
- `output`
- `summary`
- `performance`
- `artifacts`
- `images`

`artifacts.environment_profile`:

- `path_absolute`
- `path_relative`

`environment_profile.json` включает:

- host/OS данные (platform, kernel, cpu count, machine)
- python данные (version, executable, implementation)
- версии фреймворков (`torch`, `ultralytics`, `onnxruntime`, `tensorrt`, `numpy`, `pillow`)
- best-effort данные по GPU (`nvidia-smi`, CUDA availability через torch)

## Caveats

- External providers могут не отдавать per-image telemetry; тогда детальный stage timing недоступен.
- Stage breakdown зависит от backend и может заполняться частично.
- Между backend сравнивать метрики нужно с учетом различий рантайма (provider selection, CUDA/TRT runtime, precision, особенности экспорта модели).
- `infer_only` лучше для сравнения backend-ов, а `end_to_end` - для оценки пользовательской pipeline latency.
