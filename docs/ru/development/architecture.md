> English version: [../../development/architecture.md](../../development/architecture.md)

# Архитектура и диаграммы

Документ фиксирует фактические потоки в коде и помогает быстро локализовать изменения.

Источники правды для этого раздела: `smartrain/cli.py`, `smartrain/model_training_module.py`, `smartrain/results_analyzer.py`, `smartrain/training_queue.py`, `smartrain/workspace_paths.py`.

## 1) Верхнеуровневая архитектура

```mermaid
flowchart TD
  cli[smartrain CLI] --> datasetPipeline[Пайплайн_датасетов]
  cli --> trainingFlow[Пайплайн_обучения]
  cli --> queueFlow[Пайплайн_очереди]
  cli --> analyticsFlow[Аналитика_и_реестр]
  datasetPipeline --> workspaceFs[Файловый_workspace]
  trainingFlow --> workspaceFs
  queueFlow --> workspaceFs
  analyticsFlow --> workspaceFs
```

Что показывает: четыре ключевых контура системы, связанные через файловый workspace.
Как читать: от `smartrain CLI` к подсистемам, затем к общему состоянию в FS.
Практический вывод: изменения контрактов файлов влияют сразу на несколько команд.

## 2) Последовательность `smartrain train`

```mermaid
sequenceDiagram
  participant User
  participant CLI as cli.py
  participant Train as model_training_module.py
  participant Profile as train_profile.py
  participant YOLO as ultralytics.YOLO
  User->>CLI: smartrain train ...
  CLI->>Train: main(argv)
  Train->>Profile: merge параметров
  Train->>Train: resolve dataset and runtime data.yaml
  Train->>YOLO: train()
  Train->>YOLO: val()
  Train->>Train: write metrics and training_metadata.json
```

Что показывает: полный путь от CLI-вызова до артефактов обучения.
Как читать: сверху вниз по временной оси, где каждый вызов уточняет контекст.
Практический вывод: при проблемах с параметрами нужно проверять этап merge-профиля до старта `YOLO.train()`.

## 3) Жизненный цикл данных в рабочем каталоге

```mermaid
flowchart TD
  rawData[raw_data and external sources] --> scan[scan]
  scan --> datasets[datasets and datasets_info.json]
  datasets --> fusion[fusion or augment or balance or roi]
  fusion --> train[train]
  train --> runs[runs]
  runs --> analyze[analyze]
  runs --> registry[registry models-add]
  analyze --> analytics[analytics]
  registry --> models[models]
```

Что показывает: как данные двигаются между основными каталогами.
Как читать: слева направо от источника к финальным артефактам.
Практический вывод: если ломается следующий этап конвейера, первым делом проверяется целостность `datasets_info.json` и выходы `fusion`.

## 4) Состояния задачи очереди

```mermaid
stateDiagram-v2
  [*] --> Waiting
  Waiting --> Running: executor picks task
  Running --> Done: return code 0
  Running --> Error: return code != 0
  Done --> [*]
  Error --> [*]
```

Что показывает: статусы строки из `queue.txt`.
Как читать: переходы определяются результатом запуска команды.
Практический вывод: обработка ретраев не автоматизирована, повторный запуск выполняется вручную.

Примечание по терминам: статусы на диаграмме (`Waiting`, `Running`, `Done`, `Error`) соответствуют фактическим строкам в `status.txt`.

## 5) Контракты артефактов

```mermaid
flowchart TD
  scanOut[datasets_info.json and class_names.json] --> fusionInput[fusion]
  fusionInput --> dataYaml[data.yaml]
  dataYaml --> trainRun[run directory]
  trainRun --> trainingMeta[training_metadata.json]
  trainRun --> metricsCsv[test_metrics.csv]
  trainRun --> registryManifest[model_manifest.json]
  trainRun --> analyzeTable[analyze export-table csv]
```

Что показывает: зависимости между файлами-контрактами.
Как читать: стрелка означает, что один артефакт является входом для следующего шага.
Практический вывод: любые изменения схемы `training_metadata.json` требуют проверки `analyze` и `registry`.

## 6) Путь `scan/fusion` для zip и CVAT 1.1

```mermaid
sequenceDiagram
  participant User
  participant Scan as datasets_json_former.py
  participant Access as dataset_access.py
  participant CVAT as cvat11_converter.py
  participant Fusion as dataset_former.py
  User->>Scan: smartrain scan
  Scan->>Access: resolve source root
  Access-->>Scan: structure cvat11 or zip
  User->>Fusion: smartrain fusion ...
  Fusion->>Access: iter image and label buckets
  Access->>CVAT: temporary labels from annotations.xml
  CVAT-->>Fusion: yolo-compatible label stream
  Fusion-->>User: merged dataset
```

Что показывает: как CVAT/zip-источники приводятся к единому потоку для merge.
Как читать: временные метки генерируются на этапе доступа к данным, а не как отдельный обязательный импорт.
Практический вывод: `cvat import` нужен не всегда, так как `fusion` умеет нативно работать с `cvat11`.
