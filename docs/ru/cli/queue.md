> English version: [../../cli/queue.md](../../cli/queue.md)

# CLI: очередь

## Файлы очереди

- Основной файл в рабочем каталоге: `queue.txt`
- Файл статусов: `tmp/status.txt`

Резервный режим без рабочего каталога использует устаревшие пути рядом с модулем (`training_queue.txt` и `tmp/status.txt`).

## Команды

```bash
smartrain queue list
smartrain queue add -- smartrain train --data my_dataset -y
smartrain queue run --no-gui
smartrain queue-run --no-gui
```

## Статусы задач

- `Waiting to be completed`
- `Running`
- `Done`
- `Error`
- `Waiting retry N/M` — после retryable-сбоя (попытка N из max M)

## Повторы

```bash
smartrain queue run --no-gui --max-retries 2 --retry-backoff-sec 30 --retry-exit-codes 1
smartrain queue-run --no-gui --max-retries 2
```

- `--max-retries` по умолчанию `0` (как раньше: сразу `Error`)
- Backoff: `min(600, backoff * 2^(attempt-1))`
- Ошибки разбора / отсутствующей команды — non-retryable → сразу `Error`
