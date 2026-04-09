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

То есть параметры командной строки всегда имеют наивысший приоритет.

## `clearml-upload`

Отдельная команда для загрузки артефактов запуска в ClearML:

```bash
smartrain clearml-upload /path/to/run_folder --project MyProject
```

Во время обучения можно включить интеграцию флагом `--clearml`.
