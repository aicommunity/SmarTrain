> Russian version: [../ru/cli/queue.md](../ru/cli/queue.md)

# CLI: queue

## Queue files

- Main file in the workspace: `queue.txt`
- Status file: `tmp/status.txt`

Fallback mode without a workspace uses legacy paths near the module (`training_queue.txt` and `tmp/status.txt`).

## Commands

```bash
smartrain queue list
smartrain queue add -- smartrain train --data my_dataset -y
smartrain queue remove --id 1
smartrain queue clear -y
smartrain queue run --no-gui
smartrain queue-run --no-gui
```

## Task statuses

- `Waiting to be completed`
- `Running`
- `Done`
- `Error`
- `Waiting retry N/M` — after a retryable failure (attempt N of max M)

## Retries

```bash
smartrain queue run --no-gui --max-retries 2 --retry-backoff-sec 30 --retry-exit-codes 1
smartrain queue-run --no-gui --max-retries 2
```

- `--max-retries` default `0` (legacy: fail → `Error` immediately)
- Backoff: `min(600, backoff * 2^(attempt-1))`
- Parse / missing command failures are non-retryable → immediate `Error`
