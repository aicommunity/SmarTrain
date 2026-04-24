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
