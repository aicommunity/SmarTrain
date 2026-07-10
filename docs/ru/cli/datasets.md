> English version: [../../cli/datasets.md](../../cli/datasets.md)

# CLI: датасеты

## `scan`

Обновляет индекс датасетов и синхронизирует источники в рабочем каталоге.

- Выходные файлы: `datasets_info.json`, `class_names.json`, `datasets_scan_summary.json`.
- Поддерживает источники из `raw_data/`, `--dataset`, `--datasets-list`.
- Полезные режимы: `--mode refresh`, `--purge-processed-raw`.
- **`--strip-unused-classes`** (по умолчанию **вкл.**): для **новых** датасетов после копирования в `datasets/` и конвертации (CVAT 1.1 → YOLO и т.п.) удаляет из `data.yaml` / `obj.names` классы без инстансов в разметке; `class_id` в аннотациях перенумеровывается. Отключить: `--no-strip-unused-classes`. Поддерживаются все structure ID scan (`split`, `flat`, `darknet`, `cvat11`, `cvsdcldet`, …).
- После успешного скана можно переписать абсолютные пути внутри workspace на переносимые относительные: `--repair-relative-paths` или только показать план — `--repair-relative-paths-dry-run`; при необходимости добавьте `--repair-relative-paths-include-datasets-list` для строк в `raw_data/datasets_list.txt`. Только `data.yaml` по-прежнему правит отдельная команда `normalize-data-yaml`.

## `normalize-data-yaml`

Перезаписывает `data.yaml` во всех вложенных каталогах `datasets/**/data.yaml`: убирает `path`, делает `train`/`val`/`test` относительными. Чужие абсолютные пути (другая машина) заменяются на `train/images`, `val/images` и т.п., если такие папки реально есть в этом датасете.

Пример: `smartrain normalize-data-yaml --workspace /path/to/workspace` или `--datasets-dir ... --dry-run`.

## `merge` (алиас: `fusion`)

Собирает новый датасет из нескольких источников:

- выбор входов: `--dataset` (повторяемый) или `--datasets` (CSV);
- управление классами: `--classes`, `--exclude-classes`, `--merge-classes`, `--common-classes-only`;
- разбиение: `--fusion-split train,val,test`;
- **`--strip-unused-classes`**: после merge удалить из output классы без инстансов (remap `class_id` в `.txt`).
- Интерактивный режим (`smartrain merge` из TTY) поддерживает настройку merge: после выбора классов можно добавлять правила `sources -> target`, которые в replay сериализуются в повторяемые `--merge-classes`.

Эквивалент в неинтерактивном режиме:

```bash
smartrain merge --dataset ds_a --dataset ds_b --classes "class_ab,other" --merge-classes "class_a,class_b" class_ab
```

## `split`

Переразбивает один существующий датасет на `train`/`valid`/`test` без слияния источников:

```bash
smartrain split --dataset my_dataset --split-ratio 0.7,0.2,0.1
smartrain split --dataset my_dataset --exclude-test --output-name my_dataset_resplit
```

- `--split-ratio train,val,test` — случайное перераспределение внутри каждого входного bucket (по умолчанию `0.8,0.1,0.1`; тот же алгоритм, что у `merge --fusion-split`);
- `--exclude-test` — не брать test-buckets из исходника;
- создаёт новый датасет в `datasets/` с `data.yaml` и обновляет `datasets_info.json`.

## `prune`

```bash
smartrain prune empty --dataset my_dataset
smartrain prune dedup --dataset my_dataset
smartrain prune classes --dataset my_dataset
smartrain prune size --dataset my_dataset
smartrain prune size --dataset my_dataset --min-size 12x18
smartrain prune size --dataset my_dataset --size-mode and
smartrain prune size --dataset my_dataset --no-drop-empty-images
```

- **`prune empty`** — удаляет пустые пары image/label в `<dataset>_pruned`.
- **`prune dedup`** — удаляет дубли изображений по содержимому в `<dataset>_deduped`.
- **`prune classes`** — копирует датасет в `<dataset>_classes_pruned`, удаляет неиспользуемые классы из метаданных (`data.yaml`, `obj.names`), перенумеровывает `class_id` в аннотациях; файлы изображений и label-файлов не удаляются.
- **`prune size`** — копирует датасет в `<dataset>_size_pruned`, удаляет строки разметки с bbox меньше порога `NxM` в пикселях (`--min-size`, по умолчанию `20x20`). Правило отбраковки: `--size-mode or` (по умолчанию) — если **любая** сторона меньше порога; `--size-mode and` — только если **обе** стороны меньше порога. Затем по умолчанию удаляет кадры, где не осталось разметки. Чтобы сохранить такие кадры с пустым label-файлом, используйте `--no-drop-empty-images`.

## `filter`

Фильтрация YOLO bbox в новый датасет (по умолчанию `<dataset>_fltd`). Два независимых режима (можно комбинировать; отключение: `--no-edge-filter` / `--no-size-filter`):

```bash
smartrain filter --dataset my_dataset
smartrain filter --dataset my_dataset --stats-only
smartrain filter --dataset my_dataset --dry-run
smartrain filter --dataset my_dataset --drop-images
smartrain filter --dataset my_dataset --no-edge-filter --size-filter --classes startup_marker
smartrain filter --dataset my_dataset --size-filter --size-dims width --drop-images
smartrain filter --dataset my_dataset --no-edge-filter --size-filter --size-dims width --size-baseline-mode stable --drop-images
```

- **Pass 1** — baseline ширины/высоты bbox по классам только из inset-зоны (`--baseline-inset-margin`, по умолчанию `0.01`; опционально `--baseline-inset-margin-px`). Используется обоими режимами.
- **Edge filter** (`--edge-filter`, по умолчанию вкл.) — удаление near-edge bbox, слишком малых по абсолютным (`--abs-min-width-px`, `--abs-min-height-px`) и относительным порогам (`--rel-quantile`, `--rel-width-factor`, `--rel-height-factor`). Зона близости: `--filter-proximity-margin` (по умолчанию = baseline inset); строгое касание/OOB: `--edge-eps`. Ограничение сторон: `--edge-sides` (`any`, `horizontal`, `vertical`, `up`, `down`, `left`, `right`; по умолчанию `any`). С **`--empirical-bounds`** (по умолчанию выкл.): bbox у физического края кадра — по краям изображения; inset bbox — по перцентильному hull класса (`--empirical-percentile`, по умолчанию `0.10` → p10–p90) из inset-образцов (`--empirical-inset-only`, по умолчанию вкл.). Опционально отдельный hull на разрешение: `--empirical-by-format` (по умолчанию вкл.).
- **Size filter** (`--size-filter`, по умолчанию выкл.) — удаление bbox **меньше типичного размера класса** в любом месте кадра. Контроль сторон: `--size-dims` (`any`, `width`, `height`; по умолчанию `any`). `--classes` ограничивает классы для фильтрации; остальные проходят без изменений.
  - **`--size-baseline-mode inset`** (по умолчанию) — типичный размер из inset-образцов; порог = `--rel-width-factor` × квантиль(`--rel-quantile`) inset-ширин/высот.
  - **`--size-baseline-mode stable`** — для классов со стабильным размером: bulk-trim по **всем** образцам (`--size-bulk-split-ratio` × медиана, по умолчанию 0.5), typical = квантиль внутри bulk (`--size-typical-quantile`, по умолчанию 0.25), порог = `--rel-width-factor` × typical. Опционально per-resolution: `--size-by-format`.
- Опционально: `--min-visibility`, `--min-area-px`, `--max-aspect-ratio` (только edge filter).
- **`--drop-images`** — кадр и исходная разметка убираются из train/val/test и архивируются в `_filter_audit/dropped_images/<split>/images|labels` (не участвуют в `data.yaml`, обучении и stats).
- **`--prune-empty`** (по умолчанию вкл.) — удаляет пары, где **была** разметка, но после фильтрации строк не осталось; архив в `_filter_audit/dropped_images/`.
- **`--drop-background`** (по умолчанию выкл.) — удаляет исходные кадры без аннотаций (нет label-файла или он пустой); при включении уходит в `_filter_audit/dropped_images/`.
- Частичное удаление меток — снятые строки пишутся в `_filter_audit/removed_labels/<split>/labels/` с теми же относительными путями, что и в основном датасете.
- Пути и счётчики аудита — в `filter_manifest.json` → `stats_after.audit`.
- **`--stats-only`** / **`--dry-run`** — прогноз без записи; интерактив (`smartrain filter` из TTY) показывает таблицу preview и replay-команду.
- В output: `filter_manifest.json` и `dataset_passport.json`.

## `augment`, `balance`, `orient`, `rotate`, `roi`

### Instance segmentation (полигоны YOLO)

Разметка: `class_id x1 y1 x2 y2 ...` (нормализованные вершины). См. [форматы данных](../reference/data-formats.md).

**Поддерживается:** `rotate`, `orient`, `roi --mode yolo_segment`, `report`, `merge` (фильтрация классов; `fusion` сохранён как алиас).

**Ограничения:**

- `augment` — для полигонов используйте `--label-type segment`; copy-paste bbox (`--enable-bbox-copy`) для polygon-датасетов не поддерживается.
- `balance` — head-undersampling по **описывающему bbox** полигона (приближение).
- Native ONNX/engine/TRT **test** для segmentation по умолчанию пропускается; используйте PT test.

```bash
smartrain roi --dataset my_seg --mode yolo_segment --weights yolo11s-seg.pt
```

- `augment` — автономные аугментации; выборка flip **`--flip-sampling`**; опциональный orthogonal ±90° **`--enable-orthogonal-rotate`**; conveyor-эффекты по отдельности, шум с **`--conveyor-noise-types`** / **`--conveyor-noise-intensity`** / **`--conveyor-noise-selection`** (зонтик **`--enable-conveyor`** включает все пять); в интерактиве каждый эффект спрашивается отдельно (noise по умолчанию выкл.); флаги **`--aug-class-aware-geo`** и **`--aug-total-bbox-cap-mult`** те же, что для `balance` hybrid-aug (ссылки на DODA/CUDA — см. выше; в одиночном `augment` class-aware по умолчанию **выкл.** для совместимости). При включённом капе по bbox **`--aug-budget-tail-first`** (по умолчанию **вкл.**) задаёт порядок обхода train: сначала кадры с большим хвостовым приоритетом `max_c (n_max/n_c)^γ`, затем head; **`--aug-budget-tail-gamma`** — показатель γ (по умолчанию `1.0`); отключить упорядочивание — **`--no-aug-budget-tail-first`**;
- `balance` — балансировка классов; после балансировки по умолчанию действует `--eval-coverage` — при необходимости перераспределяет элементы между `train`/`val`/`test`, чтобы поддержать eval и покрытие классов, но не допускает попадания одного и того же source-изображения в разные сплиты; если уникальных кадров недостаточно, `val/test` могут остаться частично недозаполненными; отключить — `--no-eval-coverage`;
  - ручная настройка приоритетов классов: `--class-weight-multiplier "other:0.6,tear_up:1.1"` применяет множители после базового вычисления class weights;
  - авто-ограничение head-классов включено по умолчанию (`--auto-head-cap`): инструмент автоматически рассчитывает рекомендованные множители ослабления для слишком крупных классов по train-статистике; отключить — `--no-auto-head-cap`;
  - стратегия **`hybrid-aug`**: тот же hybrid-сэмплинг, что у `hybrid`, затем офлайн-запуск **`augment`** только для сплита **train**. Пресеты аугментации: `--aug-preset geo-photo` (по умолчанию: flip + фотометрия + поворот относительно центра, якорь center) или `conveyor-lite` (добавляет все conveyor-эффекты через `--enable-conveyor`). **`--aug-class-aware-geo`** (по умолчанию **вкл.**) снижает вероятность flip / фотометрии / conveyor на кадрах с доминирующими классами, чтобы офлайн geo-photo не раздувала вклад majority по числу bbox (мотивация: class-independent DA может усиливать перекос — **DODA**, [ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/54d2d38a56a74387d5916ee40e462295-Paper-Conference.pdf); разная сила аугментации по классам — **CUDA**, [arXiv:2302.05499](https://arxiv.org/abs/2302.05499)). **По умолчанию** для hybrid-aug включён режим контролируемого роста и приоритета хвоста: `--aug-total-bbox-cap-mult 1.10`, `--aug-budget-tail-first`, `--aug-budget-tail-gamma 1.0`, а также `--train-head-bbox-undersample median-factor --train-head-bbox-cap-mult 5.0`; для eval-сплитов по умолчанию также включено консервативное прореживание head `--eval-head-bbox-undersample median-factor --eval-head-bbox-cap-mult 8.0 --eval-head-bbox-min-count 30 --eval-head-bbox-max-remove-frac 0.35` (все параметры можно переопределить явными флагами). **`--aug-total-bbox-cap-mult`** пробрасывается в `augment`: опционально ограничить суммарное число bbox на train после аугментации значением `ceil(mult × базовая сумма до доп. кадров)` при сохранении всех базовых hybrid train-кадров (слив только под **дополнительные** аугментированные файлы). При cap **> 0** в augment также передаются **`--aug-budget-tail-first`** / **`--aug-budget-tail-gamma`**, чтобы сначала тратить слоты бюджета на хвост (MVP: только порядок train-кадров). **`--aug-enable-bbox-copy`** включает copy-paste по bbox (по умолчанию выкл.). Промежуточный каталог вида `[output-name]_balanced_aug__hybrid` удаляется с диска и из `datasets_info.json` после успешного augment, если не задано `--keep-hybrid-intermediate`; при записи манифеста в `balance_manifest.json` фиксируются настройки head-прореживания train/eval и блок **`post_augment`** (`class_aware_geo`, `total_bbox_cap_mult`, `budget_tail_first`, `budget_tail_gamma`, суммы bbox до/после augment);
  - опциональное **прореживание head по bbox**: `--train-head-bbox-undersample median-factor` и `--train-head-bbox-cap-mult` (по умолчанию `5.0`) убирают лишние строки разметки YOLO для классов выше `floor(mult * медиана числа bbox на класс)` со стратифицированным round-robin; при использовании смотрите в `balance_manifest.json` ключ `head_bbox_undersample`;
  - контекст: таксономия long-tailed learning [arXiv:2110.04596](https://arxiv.org/abs/2110.04596), обзоры по detection/long-tail [arXiv:2408.00483](https://arxiv.org/abs/2408.00483); сочетание rebalance и офлайн-аугментации согласуется с практикой на перекошенных бенчмарках (в т.ч. линия COCO-ZIPF: [arXiv:2403.07113](https://arxiv.org/abs/2403.07113)).
- `orient` — коррекция поворота кадров;
- `rotate` — фиксированный поворот всего датасета на `90`, `180` или `270`° по часовой стрелке в `datasets/<name>_rot<angle>` (интерактивный режим по умолчанию);
- `roi` — кроп по ROI-модели.

#### `augment`: сколько вариантов на кадр

Для каждого исходного кадра в `--splits` (по умолчанию `train`) всегда создаётся **копия оригинала**. Дополнительные файлы по типам аугментации независимы (не все комбинации):

| Тип | По умолчанию | Вариантов на кадр | Случайность |
|-----|--------------|-------------------|-------------|
| Flip (`--enable-flip`) | выкл. | 0–1 (`probabilistic`) или все для `--flip` (`exhaustive`) | `--flip-prob`; `--flip-sampling` |
| Orthogonal ±90° (`--enable-orthogonal-rotate`) | выкл. | 0–1 или оба направления (`exhaustive`) | `--orthogonal-rotate-prob`, `--orthogonal-rotate-direction` |
| Photometric / conveyor | выкл. | 0–1 общий файл | опционально class-aware geo |
| Center-rotate | **вкл.** | до `--rotate-copies` | случайный угол ±`--center-rotate-deg` |
| Bbox copy | выкл. | до `--bbox-copy-copies` | детерминированно по seed |

Режимы flip: `horizontal`, `vertical`, `both` (один проход H+V), `h-and-v` (два отдельных файла). В отличие от `smartrain rotate`, orthogonal augment добавляет опциональные ±90° варианты только для выбранных сплитов.

Шум conveyor (`--enable-conveyor-noise`): типы **`--conveyor-noise-types`** (по умолчанию `iso,shot,gaussian`), сила **`--conveyor-noise-intensity`**, выбор **`--conveyor-noise-selection`** `random` или `stack`.

Все перечисленные команды формируют `dataset_passport.json` в новом каталоге датасета.

В `data.yaml` для переносимости корень датасета задаётся **каталогом, в котором лежит сам файл** (ключ `path` не обязателен); пути `train`/`val`/`test` — относительные к этому каталогу, без ведущего `./`, в духе Ultralytics.

## `dataset convert`

Конвертация датасетов между поддерживаемыми форматами (CVAT for images 1.1, YOLO, CvsDclDet). Источники: каталог workspace (`datasets/`), `raw_data/` или явные внешние пути.

```bash
smartrain dataset convert
smartrain dataset convert --source-zip /path/to/export.zip --to yolo --output-dir datasets/task_yolo
smartrain dataset convert --source-dir datasets/task_yolo --to cvat11_zip --output-dir /path/to/out.cvat11.zip
smartrain dataset convert --source-dir raw_data/my_det --to cvat11 --output-dir converted_raw_data/my_det
smartrain dataset convert --source-dir raw_data/my_det --to cvat11 --rename-classes white_line line --zip
smartrain dataset convert --dataset my_dataset --to cvat11_zip --output-dir /tmp/my_dataset.cvat11.zip
```

- **Интерактивный режим** (`smartrain dataset convert` из TTY): выбор источника (каталог / `raw_data` / ручной путь или zip), показ формата, выбор цели (`yolo`, `cvat11`, `cvat11_zip`), путь вывода, опциональное переименование классов для CvsDclDet, затем zip-архив (по умолчанию **нет**) и удаление папки после zip (по умолчанию **да**, если zip включён).
- **`--to`**: `yolo`, `cvat11` (папка), `cvat11_zip` (только zip).
- **`--zip` / `--no-zip`**: упаковать папку в zip (`.cvat11.zip` для CVAT, `.zip` для YOLO).
- **`--delete-after-zip` / `--no-delete-after-zip`**: удалить папку после zip (по умолчанию при `--zip`).
- Пишет `dataset_passport.json` для папочного вывода. `datasets_info.json` обновляется отдельно через `smartrain scan`.

Миграция с удалённой команды `smartrain cvat`:

| Было | Стало |
|------|-------|
| `cvat import --cvat-zip X --output-dir Y` | `dataset convert --source-zip X --to yolo --output-dir Y` |
| `cvat export --dataset-dir D --zip-path Z` | `dataset convert --source-dir D --to cvat11_zip --output-dir Z` |
| `cvat from-cvsdcldet --source-dir S --output-dir O --zip` | `dataset convert --source-dir S --to cvat11 --output-dir O --zip` |

## `dataset report`

Многоязычный отчёт с примерами по классам (Markdown + PNG; опционально PDF/ODT):

```bash
smartrain dataset report --dataset my_dataset
smartrain dataset report --dataset my_dataset -n 6 --languages en,ru
smartrain dataset report
```

- Папка по умолчанию: `analytics/datasets-reports/<dataset>_<timestamp>/`.
- Интерактивный режим (`smartrain dataset report` из TTY): выбор датасета и параметров, replay-команда.
- PDF/ODT через pandoc, WeasyPrint, `fpdf2` или `odfpy` (см. overview для зависимостей).

## `dataset rename`

Переименование датасета в каталоге workspace и обновление связанных ссылок:

```bash
smartrain dataset rename --dataset old_name --new-name new_name
smartrain dataset rename --dataset old_name --new-name new_name --dry-run
smartrain dataset rename
```

- Переименовывает `datasets/<old>/` в `datasets/<new>/` и переносит ключ в `datasets_info.json`.
- Также переименовывает `runs/<old>/` и `models/<old>/`, если они существуют.
- Обновляет ссылки в `dataset_passport.json`, `training_metadata.json`, `args.yaml`, `queue.txt` и артефактах в `analytics/`.
- **`--dry-run`**: показать план без изменений.
- **`--move-data-path`**: обязателен, если у датасета кастомный `data_path` вне `datasets/<name>/`.
- Интерактивный режим (`smartrain dataset rename` из TTY): выбор датасета, ввод нового имени, предпросмотр плана и replay-команда.
- Не изменяет источники в `raw_data/` и `class_names.json`.

## `hash`

Проверка и вычисление хеша структуры датасета:

```bash
smartrain hash --dataset my_dataset
smartrain hash /abs/path/to/dataset --validate a1b2c3d4
```

Коды выхода `--validate`: `0` совпадение, `1` несовпадение, `2` ошибка.

## `stats`

Текущее поведение:

- `smartrain stats` запускает единый режим статистики (по датасетам и классам за один запуск).
- `smartrain stats compare` запускает отдельный режим сравнения двух датасетов.
- Legacy-формы `smartrain stats classes` и `smartrain stats datasets` приняты для совместимости и маршрутизируются в тот же единый режим stats.
