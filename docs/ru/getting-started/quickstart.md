> English version: [../../getting-started/quickstart.md](../../getting-started/quickstart.md)

# Быстрый старт

Это руководство предполагает, что `smartrain` уже установлен, а команды запускаются из корня workspace.

## Основной путь: обучение на датасете и формирование отчетов

1. **Создайте рабочую папку (если её ещё нет)**
   `deploy` нужно запускать уже изнутри этой рабочей папки.

Режим запуска: только non-interactive.

```bash
mkdir -p /path/to/my_workspace
cd /path/to/my_workspace
```

2. **Один раз инициализируйте структуру workspace**
   Команда создаёт стандартные директории: `raw_data/`, `datasets/`, `runs/`, `analytics/` и др.

Режим запуска: только non-interactive.

```bash
smartrain deploy
```

3. **Добавьте источники датасета и выполните индексацию**
   `scan` распознаёт поддерживаемые layout-источники, нормализует метаданные и обновляет индекс.

Режим запуска: только non-interactive.

```bash
# Вариант A: источники уже лежат в workspace/raw_data/
smartrain scan

# Вариант B: передайте явный путь к источнику
smartrain scan --dataset /data/datasets/my_dataset
```

Поддерживаемые layout-форматы датасетов:

- YOLO split directories layout
- YOLO flat paired directories layout
- YOLO flat with subset subfolders
- YOLO nested split under images/labels
- Darknet YOLO dataset layout
- CVAT for images 1.1 layout

Поддерживаемые аннотации: YOLO bbox (`class_id cx cy w h`) и полигоны сегментации.
Точное соответствие этим названиям и internal IDs SmarTrain, а также различия layout, см. в `docs/ru/reference/data-formats.md`.

4. **Запустите обучение модели на выбранном датасете**
   В `runs/...` появится run с весами, метриками и `training_metadata.json`.

Режимы запуска:

- Interactive (без аргументов): `smartrain train`
- Non-interactive (минимальные аргументы):

```bash
smartrain train --data my_dataset --model yolo11n.pt -y
```

5. **Сформируйте отчет по датасету**
   Команда создаёт визуальный/текстовый отчёт по качеству и покрытию классов.

Режимы запуска:

- Interactive (без аргументов): `smartrain report dataset`
- Non-interactive (минимальные аргументы):

```bash
smartrain report dataset --dataset my_dataset -n 6 --languages en,ru
```

6. **Сформируйте аналитический отчет по запускам**
   Команда формирует итоговые аналитические артефакты по запускам.

Режимы запуска:

- Interactive (без аргументов): `smartrain analyze`
- Non-interactive (минимальные аргументы):

```bash
smartrain analyze all --report-languages en,ru
```

`smartrain analyze scan` опционален как отдельная предварительная проверка, но не обязателен перед `smartrain analyze`/`smartrain analyze all`.

## Опциональные шаги

- **Сравнение разных запусков**
  Режим запуска: non-interactive.
  ```bash
  smartrain analyze compare --baseline /path/to/run_a --others /path/to/run_b /path/to/run_c
  ```
- **Работа с датасетами перед обучением** (при необходимости): `fusion`, `augment`, `balance`, `prune`, `orient`, `roi`.

## Куда сохраняются результаты

- Отчеты по датасету: `analytics/datasets-reports/<dataset>_<timestamp>/`
- Сессии анализа: `analytics/analyze-reports/<session>/`
- Обучающие запуски: `runs/<dataset>/<run>/`
