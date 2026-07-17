> English version: [../../README.md](../../README.md)

# Smart Train (`smartrain`)

CLI-набор для подготовки YOLO-датасетов, обучения моделей, исполнения очередей и анализа прогонов.

## Быстрый старт

Требования: Python `3.10+`.

```bash
git clone <repo-url>
cd smart-train
pip install -e .
```

Основной сценарий работы — из текущего каталога проекта:

```bash
smartrain deploy
smartrain scan
smartrain merge --dataset ds_a --dataset ds_b --classes "class_a,class_b"
smartrain train --data 2026-01-01_12-00-00-merged -y
```

## Что внутри

- Единая точка входа: `smartrain` (модуль `smartrain.cli`).
- Модель единого рабочего каталога: `raw_data/`, `datasets/`, `runs/`, `analytics/`, `models/`, `inference/`, `tmp/`.
- Поддержка конвейера: `scan -> merge -> train -> analyze`.
- Отдельные инструменты: `queue`, `registry`, `dataset` (`report`, `rename`, `convert`), `model`, `normalize-data-yaml`, `migrate`, `migrate-models`, `clearml-upload`, `plot`, `sahi`, `heatmap`, `orient`, `rotate`, `vis`, `deps`.

## Принцип работы

`smartrain` использует единый корень workspace и строит процесс вокруг файловых контрактов:

- `scan` синхронизирует источники и обновляет каталог датасетов;
- `fusion` формирует итоговый датасет под обучение;
- `train` создаёт run-каталог с метриками и метаданными;
- `analyze` и `registry` работают по артефактам в `runs/`.

## Ключевые команды

| Команда | Назначение |
|---|---|
| `smartrain deploy` | Инициализация структуры workspace |
| `smartrain scan` | Синхронизация источников и обновление каталога датасетов |
| `smartrain merge` | Сборка итогового датасета для обучения |
| `smartrain train` | Обучение/валидация модели YOLO |
| `smartrain inference` | Инференс по папке или сплиту датасета с сохранением JSON-отчёта |
| `smartrain queue` / `smartrain queue-run` | Управление и запуск очереди команд |
| `smartrain analyze` | Сводки, сравнение запусков, PR-кривые, бенчмарк инференса |
| `smartrain registry` | Каталогизация артефактов запусков и промо моделей |

## Документация

Актуальная документация организована по разделам в `docs/`:

- [Навигация по документации](index.md)
- [Старт и базовые сценарии](getting-started/quickstart.md)
- [CLI-руководство](cli/overview.md)
- [Справочник форматов и API](reference/api.md)
- [Архитектура и диаграммы](development/architecture.md)

## Тестирование

```bash
pip install -e ".[dev]"
pytest
```

## Важные детали

- Интерактивный режим включается только если команда запущена вообще без аргументов (нужен TTY).
- Интерактивные команды по датасетам: `fusion`, `augment`, `balance`, `stats`, `roi`, `orient`, `inference`, `prune`, а также `train`.
- Очистка датасетов: `prune` (`prune empty` для пустых пар, `prune dedup` для дублей, `prune classes` для неиспользуемых классов, `prune size` для меток меньше `NxM` с дефолтом `20x20`; по умолчанию этот режим удаляет кадры без оставшейся разметки, для отключения используйте `--no-drop-empty-images`).
- Если аргументы переданы частично и обязательных не хватает, команда выводит понятную ошибку о неполных аргументах и не уходит в prompt-режим.
- В справке команд есть практические блоки `Examples` / `Quick examples` для типовых сценариев.
- Пресеты `smartrain balance`:
  - `--preset weights-safe` для консервативной балансировки
  - `--preset rfs-aggressive` для более агрессивного усиления tail-классов
  - `--preset hybrid-default` как универсальный дефолт
  - `--preset hybrid-aug-tail-budget` для hybrid-aug с ограничением роста, tail-first бюджетом и прореживанием head
- Подвыборки после `balance`: по умолчанию включено `--eval-coverage` (по возможности не оставлять пустыми `val`/`test` и улучшать покрытие классов в eval); отключить — `--no-eval-coverage`. В интерактивном режиме этот пункт задаётся отдельным вопросом.
- Для `hash --validate`: `0` при совпадении, `1` при несовпадении, `2` при ошибке.
- По умолчанию очередь workspace использует `queue.txt` и `tmp/status.txt`.
- Расширения зависимостей:
  - `pip install -e ".[dev]"` для разработки и тестов
  - `pip install -e ".[clearml]"` для ClearML
  - `pip install -e ".[sahi]"` для SAHI
  - `smartrain deps install` или `pip install -e ".[export]"` для PDF/ODT отчётов (`pypandoc-binary`, `weasyprint`)
  - проверка: `smartrain deps doctor`

Автодополнение:
  - best-effort автонастройка выполняется при первом запуске `smartrain` после установки;
  - ручной fallback:
    - `smartrain --install-completion`
    - `smartrain --show-completion`

## Частые сценарии

Сканирование с явным списком источников:

```bash
smartrain scan --datasets-list /path/to/workspace/raw_data/datasets_list.txt
```

Проверка хеша датасета:

```bash
smartrain hash --dataset my_dataset
smartrain hash /path/to/dataset --validate a1b2c3d4
```

Запуск очереди без открытия GUI-терминала:

```bash
smartrain queue run --no-gui
```

Быстрый просмотр запусков:

```bash
smartrain analyze scan
smartrain analyze export-table -o runs_summary.csv
```

## Длительные запуски по SSH (tmux)

Для долгого обучения на удаленном сервере используйте `tmux`, чтобы процесс не прерывался при обрыве SSH-соединения.

Установите `tmux` один раз (пример для Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y tmux
```

Минимальный сценарий:

```bash
tmux new -s smartrain-train
smartrain train --data my_dataset --model yolo11n.pt --device 0
```

- Отсоединиться от сессии без остановки обучения: `Ctrl+B`, затем `D`
- Подключиться снова после переподключения по SSH: `tmux attach -t smartrain-train`
- Остановить обучение в активной сессии: `Ctrl+C`
- Удалить ненужную сессию: `tmux kill-session -t smartrain-train`

Также можно использовать вспомогательные скрипты из `scripts/`:

```bash
./scripts/tmux_train_start.sh --session smartrain-train -- smartrain train --data my_dataset --model yolo11n.pt --device 0
./scripts/tmux_train_attach.sh --session smartrain-train
./scripts/tmux_train_stop.sh --session smartrain-train
```

Опционально: лог в файл с сохранением live-вывода в консоли:

```bash
./scripts/tmux_train_start.sh --session smartrain-train -- bash -lc 'smartrain train --data my_dataset --model yolo11n.pt --device 0 2>&1 | tee -a runs/train.log'
```

### Операционные рецепты

Проверить активные tmux-сессии:

```bash
tmux ls
```

Проверить, что процесс обучения в сессии еще жив:

```bash
tmux list-panes -t smartrain-train -F '#{pane_current_command} #{pane_pid}'
```

Вернуть live-вывод после переподключения:

```bash
tmux attach -t smartrain-train
```

Если сессия уже attach в другом месте, принудительно переподключиться:

```bash
tmux attach -d -t smartrain-train
```

Корректная остановка и очистка:

```bash
./scripts/tmux_train_stop.sh --session smartrain-train
tmux kill-session -t smartrain-train
```

### FAQ (tmux по SSH)

**Сессия есть, но нового вывода не видно. Что проверить в первую очередь?**
- Переподключиться с принудительным detach: `tmux attach -d -t smartrain-train`
- Проверить текущую команду в pane: `tmux list-panes -t smartrain-train -F '#{pane_current_command} #{pane_pid}'`
- Если запускали обучение с `tee`, посмотреть лог (например, `runs/train.log`).

**SSH оборвался. Обучение остановилось?**
- Обычно нет, если запуск был внутри `tmux`.
- Подключитесь снова и выполните: `tmux ls`, затем `tmux attach -t smartrain-train`.

**Ctrl+C не останавливает запуск из текущей оболочки.**
- Сначала убедитесь, что вы attach к нужной `tmux`-сессии/окну.
- Либо отправьте прерывание явно: `./scripts/tmux_train_stop.sh --session smartrain-train`.

**Как быстро найти логи последнего запуска?**
- Пример:
  - `ls -lt runs | head`
  - `tail -n 200 runs/train.log` (если использовали `tee -a runs/train.log`)

**Как почистить старые tmux-сессии?**
- Список сессий: `tmux ls`
- Удалить одну: `tmux kill-session -t <session>`
- Удалить все сессии tmux-сервера (осторожно): `tmux kill-server`
