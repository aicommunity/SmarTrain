> English version: [../../migration/legacy-commands.md](../../migration/legacy-commands.md)

# Устаревшие режимы и совместимость

## Устаревшие или переходные сценарии

- `smartrain plot` — устаревшая обёртка над `smartrain analyze`.
- `smartrain cvat` удалён; используйте `smartrain dataset convert`.
- Устаревшие режимы `fusion`/`roi` без рабочего каталога по-прежнему поддерживаются.
- Внутри исполнителя очереди есть резервный путь `training_queue.txt`, но в режиме рабочего каталога основной файл — `queue.txt`.

## Рекомендации

- Для новых сценариев всегда использовать подход с рабочим каталогом.
- Для аналитики использовать `smartrain analyze ...`, а не `plot`.
