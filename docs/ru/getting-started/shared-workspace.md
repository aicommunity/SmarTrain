> English version: [../getting-started/shared-workspace.md](../getting-started/shared-workspace.md)

# Общий workspace (SMB / несколько машин)

`smartrain` использует файловый workspace. Несколько машин могут указывать `--workspace` (или `SMART_TRAIN_WORKSPACE`) на одну SMB/CIFS-папку при соблюдении правил координации ниже.

## Ограничения SMB

- Используйте **lock-файлы O_EXCL** в `tmp/locks/` (на Windows по умолчанию; на Linux-клиентах SMB задайте `SMART_TRAIN_SMB_LOCKS=1`).
- Advisory `flock` на соседних `*.lock` **ненадёжен** между клиентами SMB.
- После **SIGKILL** или обрыва питания устаревшие peer/lock удаляются при **следующем запуске на том же хосте** (`reconcile_local_holders`).

## Структура координации

```
workspace/tmp/peers/     # heartbeat JSON на каждую CLI-сессию
workspace/tmp/locks/     # scan.lock, catalog.lock, queue.lock, run-*.lock
```

## Уровни операций

| Уровень | Команды | Поведение |
|---------|---------|-----------|
| Auto-scan | Команды с `ensure_scan=True` (`train`, `merge`, …) | `scan.lock` try-acquire (0–2 с); при занятости — **пропуск auto-scan**, команда продолжается |
| 0 | `inference`, read-only `analyze`, `hash`, `stats` | Предупреждение о других peers; без блокировки |
| 1 | `scan`, `merge`, `augment`, `split`, … | `catalog.lock` на время команды (ожидание по умолчанию 30 с); явный `scan` — `scan.lock` (по умолчанию 300 с) |
| 2 | мутации `queue` | `queue.lock` |
| 3 | `train`, `test`, тяжёлый `analyze` на один run | Advisory `run-<dataset>-<run>.lock` (предупреждение, если нет `--force-resource-lock`) |

## Полезные флаги

```bash
smartrain --no-auto-scan train --data ds_a -y
smartrain --no-peer-warn inference --model my_run --source /path/images
smartrain scan --wait-for-scan=120
smartrain merge --catalog-lock-timeout=60 ...
smartrain workspace peers
smartrain workspace peers --json
```

## Auto-scan

Перед многими командами выполняется тихий `scan`. Если другая машина держит `scan.lock`, auto-scan **пропускается** (без бесконечного ожидания). Для полного обновления каталога запускайте `smartrain scan` явно.

## Ctrl+C и восстановление

- **Ctrl+C / SIGTERM:** активная сессия удаляет peer-файл и удерживаемые locks.
- **Сбой / kill -9:** при следующем запуске `smartrain` на **том же хосте** удаляются устаревшие peer/lock, если PID больше не соответствует работающему `smartrain`.

## Безопасные сценарии

| Сценарий | OK? |
|----------|-----|
| Два `train` на разных датасетах | Да |
| `train`, пока на другой машине `scan` | Да (auto-scan на train пропускается) |
| Два явных `scan` одновременно | Нет (сериализация через `scan.lock`) |
| Два `test` одного run | Нет (без `--force-resource-lock`) |

## См. также

- [Структура workspace](workspace.md)
- `smartrain sync` — подтягивание недостающих артефактов между копиями workspace
