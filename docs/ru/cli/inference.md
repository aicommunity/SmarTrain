> English version: [../../cli/inference.md](../../cli/inference.md)

# CLI: inference

`smartrain inference` запускает инференс по папке, архиву или split датасета (детекция, instance segmentation, классификация — в зависимости от модели/task).

Команда пишет:

- `inference_results.json` (основной отчет)
- `environment_profile.json` (профиль окружения машины и рантайма)
- по умолчанию YOLO-датасет `<basename>_autolabeled/` (псевдо-разметка из предсказаний)
- опционально `pred_overlays/` с отрисованными предсказаниями (по умолчанию вкл. при экспорте датасета)
- опционально overlay-изображения с полигонами при `--save-overlay` (instance segmentation, legacy)

Основные JSON сохраняются в:

- `workspace/inference/<model>/<timestamp-source>/`

## Экспорт YOLO-датасета (autolabel)

По умолчанию включён (`--export-dataset`, отключение: `--no-export-dataset`).

По умолчанию экспорт делит результат на **независимые YOLO-поддатасеты** (`--export-split-dirs`, отключение: `--no-export-split-dirs`). В каждом `part_XXX/` — до `--export-files-per-dir` **фактически экспортированных** кадров (после фильтра confidence меток; по умолчанию `500`).

Разбиение (по умолчанию):

```
<basename>_autolabeled/
  autolabel_manifest.json          # корневой index (layout=independent_parts)
  part_000/
    images/
    labels/
    data.yaml
    autolabel_manifest.json
  part_001/
    ...
pred_overlays/
  part_000/
  part_001/
```

Плоский layout (`--no-export-split-dirs`):

```
<basename>_autolabeled/
  images/
  labels/
  data.yaml
  autolabel_manifest.json
```

- **`basename`** — имя исходной папки или архива (`--source` / `--source-dir`) или `{dataset}-{split}` для `dataset-split`.

## Источники данных

| Режим | Флаги | Поддержка архивов |
|-------|-------|-------------------|
| `folder` | `--source` или `--source-dir` | Да: `.zip`, `.tar`, `.tar.gz`, `.tgz` — распаковка в `tmp/extracted_datasets/` |
| `dataset-split` | `--dataset`, `--split` | Да, если `data_path` в `datasets_info.json` указывает на архив |

Архивы распаковываются в кэш workspace (`tmp/extracted_datasets/`) с инвалидацией по mtime/size. В отчёте `source.source_archive_*` сохраняет путь к исходному архиву, если он был использован.

- В датасет попадают **только кадры с ≥1 детекцией/сегментом** после фильтра confidence.
- **`autolabel_manifest.json`** — модель, параметры инференса/экспорта, статистика, `file_mapping` (в part при split; в корне — index по parts).

Флаги экспорта / инференса:

| Флаг | По умолчанию | Назначение |
|------|--------------|------------|
| `--export-dataset` / `--no-export-dataset` | вкл | Экспорт YOLO-датасета |
| `--export-label-conf-min` | `0.25` | Мин. confidence для записи метки |
| `--export-label-conf-max` | `1.0` | Макс. confidence для записи метки |
| `--export-visualize` / `--no-export-visualize` | вкл при export-dataset | Папка `pred_overlays/` с отрисовкой как в `vis` |
| `--export-split-dirs` / `--no-export-split-dirs` | вкл | Независимые поддатасеты `part_XXX/` (+ зеркало overlays) |
| `--export-files-per-dir` | `500` | Макс. экспортированных кадров на поддатасет |
| `--batch-size` | `8` | Размер батча локального Ultralytics-инференса (для external не применяется) |

`--conf` задаёт порог инференса; `--export-label-conf-*` дополнительно фильтрует метки при записи в датасет (из уже полученных предсказаний).

Classification в YOLO не экспортируется (предупреждение, каталог датасета не создаётся). Если ни один кадр не проходит фильтр confidence при экспорте, каталоги `<basename>_autolabeled/` и `pred_overlays/` не создаются. При OOM уменьшите `--batch-size`.

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
smartrain inference --model-name my_model --data-mode folder --source raw_data/images.zip --device cpu
smartrain inference --model-name my_model --data-mode folder --source-dir ./images --no-export-dataset
smartrain inference --model-name my_model --data-mode folder --source-dir ./images --export-label-conf-min 0.4 --export-label-conf-max 0.9
smartrain inference --weights ./runs/ds/run_001/models/run_001.engine --data-mode folder --source-dir ./images
smartrain inference --weights dr-yolo:yolov8n --external-repo /opt/dr-yolo --data-mode folder --source-dir ./images
smartrain inference --weights yolo11s-seg.pt --data-mode folder --source-dir ./images --save-overlay
```

### Instance segmentation overlay

Для моделей `*-seg.pt` в JSON есть `segments` (полигоны). Флаг `--save-overlay` сохраняет RGB-превью с контурами полигонов рядом с отчётом.

## Выбор устройства

- `--device` поддерживает `cpu`, индекс GPU (`0`) и `cuda:N`.
- В интерактивном режиме можно вводить номер, токен или имя GPU.
- Устройство по умолчанию: `GPU 0`, если CUDA доступна, иначе `cpu`.
- Те же правила выбора применяются в `train` и `test`.

## Разрешение входа (`--img-size`)

Если `--img-size` не задан, inference определяет размер входа из контекста модели (в порядке приоритета):

1. `training_metadata.json`, `args.yaml` и другие metadata-файлы рядом с моделью или в родительских каталогах до `models/`
2. sidecar `*.meta.json` рядом с весами
3. имя артефакта с токеном `_imgsz{N}x{N}_` (например ONNX после `model convert`)
4. статическая H/W входа ONNX-графа

Если ни один источник не найден, используется fallback **640** с предупреждением `[WARN]`. Явный `--img-size` всегда имеет приоритет (`img_size_source: cli`).

В `inference_results.json` → `parameters.img_size_source` сохраняется метка источника (например `training_metadata`, `artifact_filename`, `fallback_640`).

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

`artifacts.autolabel_dataset` (при `--export-dataset`):

- `path_absolute`, `path_relative`
- `manifest_absolute` → `autolabel_manifest.json`
- `images_exported`, `labels_total`

`artifacts.pred_overlays` (при `--export-visualize`):

- `path_absolute`, `path_relative`
- `images_rendered`

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
- Экспортные метки — псевдо-разметка из модели, не ground truth.
- При включённом export-dataset в `pred_overlays/` попадают только кадры, записанные в датасет.
