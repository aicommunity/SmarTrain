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
smartrain queue run --no-gui
smartrain queue-run --no-gui
```

## Task statuses

- `Waiting to be completed`
- `Running`
- `Done`
- `Error`
