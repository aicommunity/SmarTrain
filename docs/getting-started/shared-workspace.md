> Russian version: [../ru/getting-started/shared-workspace.md](../ru/getting-started/shared-workspace.md)

# Shared workspace (SMB / multi-machine)

`smartrain` uses a file-based workspace. Several machines can point `--workspace` (or `SMART_TRAIN_WORKSPACE`) at the same SMB/CIFS share if you follow the coordination rules below.

## SMB limitations

- Prefer **O_EXCL lock files** under `tmp/locks/` (enabled by default on Windows; set `SMART_TRAIN_SMB_LOCKS=1` on Linux clients using SMB).
- Advisory `flock` on sibling `*.lock` files is **not reliable** across SMB clients.
- After **SIGKILL** or power loss, stale lock/peer files are removed on the **next launch on the same host** (`reconcile_local_holders`).

## Coordination layout

```
workspace/tmp/peers/     # heartbeat JSON per CLI session
workspace/tmp/locks/     # scan.lock, catalog.lock, queue.lock, run-*.lock
```

## Operation tiers

| Tier | Commands | Behavior |
|------|----------|----------|
| Auto-scan | Commands with `ensure_scan=True` (`train`, `merge`, …) | `scan.lock` try-acquire (0–2 s); if busy → **skip auto-scan**, command continues |
| 0 | `inference`, read-only `analyze`, `hash`, `stats` | Warn about other peers; no command lock |
| 1 | `scan`, `merge`, `augment`, `split`, … | `catalog.lock` for the main command (bounded wait, default 30 s); explicit `scan` uses `scan.lock` (default wait 300 s) |
| 2 | `queue` mutations | `queue.lock` |
| 3 | `train`, `test`, heavy `analyze` on one run | Advisory `run-<dataset>-<run>.lock` (warn unless `--force-resource-lock`) |

## Useful flags

```bash
smartrain --no-auto-scan train --data ds_a -y
smartrain --no-peer-warn inference --model my_run --source /path/images
smartrain scan --wait-for-scan=120
smartrain merge --catalog-lock-timeout=60 ...
smartrain workspace peers
smartrain workspace peers --json
```

## Auto-scan

Most dataset/training commands run a quiet `scan` first. If another machine holds `scan.lock`, auto-scan is **skipped** (no infinite wait). Run `smartrain scan` explicitly when you need a full catalog refresh.

## Ctrl+C and recovery

- **Ctrl+C / SIGTERM:** the active session removes its peer file and held locks.
- **Crash / kill -9:** the next `smartrain` start on the **same host** cleans stale peer/lock files whose PID is no longer a running `smartrain` process.

## Safe scenarios

| Scenario | OK? |
|----------|-----|
| Two `train` on different datasets | Yes |
| `train` while another host runs `scan` | Yes (auto-scan skipped on train) |
| Two explicit `scan` at once | No (serialized by `scan.lock`) |
| Two `test` on the same run | No (unless `--force-resource-lock`) |

## Related

- [Workspace layout](workspace.md)
- `smartrain sync` — merge missing artifacts between workspace copies
