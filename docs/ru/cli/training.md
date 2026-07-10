> English version: [../../cli/training.md](../../cli/training.md)

# CLI: обучение

## `train`

Основная команда обучения/валидации.

```bash
smartrain train --data my_dataset -y
smartrain train --data my_seg_dataset --task segment --model yolo11s-seg.pt -y
smartrain train --test-only --model-dir /path/to/run --data /path/to/dataset
```

### Instance segmentation

- `--task segment` и модель `*-seg.pt` (например `yolo11s-seg.pt`).
- Разметка — YOLO-полигоны. См. [форматы данных](../reference/data-formats.md).
- В интерактиве при task=segment предлагаются модели с `-seg`.
- Полный набор графиков после обучения: `smartrain test --formats pt --task segment`.
- Native ONNX/engine/TRT test для segmentation по умолчанию пропускается.
- Экспериментально: `smartrain test --formats onnx --task segment --force-native-seg-test` (bbox-native eval; mask-метрики ненадёжны).

Источники параметров и приоритет:

`CLI > --ultralytics_yaml > --config > defaults`

Поле `data` из `--ultralytics_yaml` игнорируется, используется выбранный `--data`.
Также из `--ultralytics_yaml` игнорируются: `project`, `name`, `exist_ok`, `cfg`, `device`, `model_dir`, `target_path`, `workspace`, `save_dir`, `runs_dir`, `output_dir`.

То есть параметры командной строки всегда имеют наивысший приоритет.

Двухэтапная защита для внешнего YAML:

1. На этапе merge из `--ultralytics_yaml` отбрасываются сервисные и path-like ключи.
2. На этапе finalize перед `YOLO.train(...)` принудительно задаются `data`, `project`, `name`, `exist_ok`.

Это не позволяет стороннему `args.yaml` перенаправить запуск в чужие директории.

Примечание по переносимости путей:

- Избегайте абсолютных машинно-зависимых путей (например `/mnt/*`) во внешнем `args.yaml`.
- Для выбора устройства используйте `--device` из CLI; `device` в `--ultralytics_yaml` игнорируется.

Выбор модели:

- Список поддерживаемых алиасов для обучения выводится командой `smartrain info`.
- В интерактивном режиме `train` предлагает выбор из этого списка и пункт `<manual>` для ручного ввода.
- `--model` принимает и YOLO-алиасы, и путь к весам; для обычных YOLO-алиасов автоматически добавляется `.pt`.
- Для внешних провайдеров поддерживается префиксный формат `provider:model` (например: `dr-yolo:yolov8n`).
- Для внешних провайдеров действует строгая валидация алиаса по каталогу провайдера; неподдерживаемый алиас отклоняется до запуска.
- Если указан `--external-provider`, но `--model` не указан, автоматически выбирается дефолтный алиас модели провайдера.

Выбор устройства:

- `--device` принимает: `cpu`, индекс GPU (`0`), `cuda:N`, а в интерактивном режиме также имя GPU.
- Поведение по умолчанию (для всех команд): `GPU 0`, если CUDA доступна, иначе `cpu`.
- В интерактивном режиме можно выбирать:
  - номером из списка,
  - явным токеном (`cpu`, `0`, `cuda:0`),
  - именем GPU (точное/нормализованное совпадение).
- Одинаковая модель выбора используется в `train`, `test` и `inference`.

## `clearml-upload`

Отдельная команда для загрузки артефактов запуска в ClearML:

```bash
smartrain clearml-upload /path/to/run_folder --project MyProject
```

Во время обучения можно включить интеграцию флагом `--clearml`.
