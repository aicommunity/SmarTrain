> Russian version: [../ru/cli/overview.md](../ru/cli/overview.md)

# CLI: overview

Entry point: `smartrain` (Typer router with unified command behavior).

## Command groups

- Datasets: `scan`, `fusion`, `augment`, `balance`, `orient`, `roi`, `hash`, `stats`
- Training: `train`, `clearml-upload`
- Queue: `queue`, `queue-run`
- Analytics: `analyze`, `plot` (outdated wrapper)
- Register: `registry`
- Format tools: `cvat`, `sahi`, `heatmap`

## Reference

```bash
smartrain --help
smartrain <command> --help
```

For nested commands:

```bash
smartrain queue list --help
smartrain analyze inference-benchmark --help
```

Calling `smartrain <command>` without required arguments usually prints that command's help.
`smartrain train` is special: without arguments it enters interactive setup (TTY required).
Most important commands and groups also include `Examples` / `Quick examples` directly in help output.

Balance and stats additions:

- `smartrain balance` supports `weights`, `rfs`, and `hybrid` strategies, plus weight/rfs tuning flags.
- `smartrain balance --preset {weights-safe,rfs-aggressive,hybrid-default}` applies tuned defaults for common scenarios.
- `smartrain stats --balance-ready` prints imbalance metrics and balancing recommendations.
