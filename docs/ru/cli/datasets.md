> English version: [../../cli/datasets.md](../../cli/datasets.md)

# CLI: датасеты

## `scan`

Обновляет индекс датасетов и синхронизирует источники в рабочем каталоге.

- Выходные файлы: `datasets_info.json`, `class_names.json`, `datasets_scan_summary.json`.
- Поддерживает источники из `raw_data/`, `--dataset`, `--datasets-list`.
- Полезные режимы: `--mode refresh`, `--purge-processed-raw`.
- **`--strip-unused-classes`** (по умолчанию выкл.): для **новых** датасетов после копирования в `datasets/` и конвертации (CVAT 1.1 → YOLO и т.п.) удаляет из `data.yaml` / `obj.names` классы без инстансов в разметке; `class_id` в аннотациях перенумеровывается. Поддерживаются все structure ID scan (`split`, `flat`, `darknet`, `cvat11`, `cvsdcldet`, …).
- После успешного скана можно переписать абсолютные пути внутри workspace на переносимые относительные: `--repair-relative-paths` или только показать план — `--repair-relative-paths-dry-run`; при необходимости добавьте `--repair-relative-paths-include-datasets-list` для строк в `raw_data/datasets_list.txt`. Только `data.yaml` по-прежнему правит отдельная команда `normalize-data-yaml`.

## `normalize-data-yaml`

Перезаписывает `data.yaml` во всех вложенных каталогах `datasets/**/data.yaml`: убирает `path`, делает `train`/`val`/`test` относительными. Чужие абсолютные пути (другая машина) заменяются на `train/images`, `val/images` и т.п., если такие папки реально есть в этом датасете.

Пример: `smartrain normalize-data-yaml --workspace /path/to/workspace` или `--datasets-dir ... --dry-run`.

## `fusion`

Собирает новый датасет из нескольких источников:

- выбор входов: `--dataset` (повторяемый) или `--datasets` (CSV);
- управление классами: `--classes`, `--exclude-classes`, `--merge-classes`, `--common-classes-only`;
- разбиение: `--fusion-split train,val,test`;
- **`--strip-unused-classes`**: после merge удалить из output классы без инстансов (remap `class_id` в `.txt`).

## `prune`

```bash
smartrain prune empty --dataset my_dataset
smartrain prune dedup --dataset my_dataset
smartrain prune classes --dataset my_dataset
```

- **`prune empty`** — удаляет пустые пары image/label в `<dataset>_pruned`.
- **`prune dedup`** — удаляет дубли изображений по содержимому в `<dataset>_deduped`.
- **`prune classes`** — копирует датасет в `<dataset>_classes_pruned`, удаляет неиспользуемые классы из метаданных (`data.yaml`, `obj.names`), перенумеровывает `class_id` в аннотациях; файлы изображений и label-файлов не удаляются.

## `augment`, `balance`, `orient`, `rotate`, `roi`

### Instance segmentation (полигоны YOLO)

Разметка: `class_id x1 y1 x2 y2 ...` (нормализованные вершины). См. [форматы данных](../reference/data-formats.md).

**Поддерживается:** `rotate`, `orient`, `roi --mode yolo_segment`, `report`, `fusion` (фильтрация классов).

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
