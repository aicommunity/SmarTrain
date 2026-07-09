> Russian version: [../ru/migration/legacy-commands.md](../ru/migration/legacy-commands.md)

# Legacy modes and compatibility

## Legacy and transitional commands

- `smartrain plot` is an outdated wrapper for `smartrain analyze`.
- `smartrain cvat` was removed; use `smartrain dataset convert`.
- `smartrain fusion` is now a legacy alias for `smartrain merge`; `roi` legacy mode without workspace is still supported.
- Inside the queue executor there is a backup path `training_queue.txt`, but in workspace mode the main file is `queue.txt`.

## Recommendations

- For new automation and operational usage, always use the workspace-based approach.
- For analytics, use `smartrain analyze ...`, not `plot`.
