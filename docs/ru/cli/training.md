> English version: [../../cli/training.md](../../cli/training.md)

# CLI: обучение

## `train`

Основная команда обучения/валидации.

```bash
smartrain train --data my_dataset -y
smartrain train --test-only --model-dir /path/to/run --data /path/to/dataset
```

Источники параметров и приоритет:

`CLI > --ultralytics_yaml > --config > defaults`

Поле `data` из `--ultralytics_yaml` игнорируется, используется выбранный `--data`.
Также из `--ultralytics_yaml` игнорируются: `project`, `name`, `exist_ok`, `cfg`, `device`, `model_dir`, `target_path`, `workspace`.

То есть параметры командной строки всегда имеют наивысший приоритет.

Выбор модели:

- Список поддерживаемых алиасов для обучения выводится командой `smartrain info`.
- В интерактивном режиме `train` предлагает выбор из этого списка и пункт `<manual>` для ручного ввода.
- `--model` принимает и YOLO-алиасы, и путь к весам; для обычных YOLO-алиасов автоматически добавляется `.pt`.

## `clearml-upload`

Отдельная команда для загрузки артефактов запуска в ClearML:

```bash
smartrain clearml-upload /path/to/run_folder --project MyProject
```

Во время обучения можно включить интеграцию флагом `--clearml`.
