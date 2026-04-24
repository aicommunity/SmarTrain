> Russian version: [../ru/cli/providers.md](../ru/cli/providers.md)

# CLI: external providers

## `smartrain providers`

Manage external training backends installed into provider-specific virtual environments.

```bash
smartrain providers status
smartrain providers install --all -y
smartrain providers doctor --verbose
smartrain providers uninstall --provider dr-yolo -y
```

Subcommands:

- `install`: clone/install selected providers and write global provider index.
- `uninstall`: remove selected provider deployments and index records.
- `status`: show index state (`installed`/`not_installed`) and repo paths.
- `doctor`: run readiness checks (repo, entrypoints, venv, runtime imports).

## Provider model aliases in `train`/`inference`

Use provider-prefixed model references:

```bash
smartrain train --external-provider dr-yolo --model yolov8n
smartrain train --model dr-yolo:yolov8n
smartrain inference --weights dr-yolo:yolov8n --data-mode folder --source-dir images/
```

Rules:

- `provider:model` auto-fills `--external-provider`.
- Validation is strict for external aliases: unsupported aliases are rejected with a clear error.
- In interactive `train`, model list includes installed provider aliases and supports `<manual>`.

## Default behavior for external providers

When `--external-provider` is set and explicit values are not passed:

- default model comes from provider catalog;
- default runtime launcher values are used (`epochs=70`, `batch=8`, `img_size=640`);
- run folder naming is normalized and sanitized:
  - `YYYY-MM-DD_HH-MM_<provider>_<model>_<epochs>epochs_b<batch>-<dataset_hash>`

## Artifact contract for external runs

External runs are normalized to the same contract as built-in training:

- `train/weights/best.pt`
- `test/` outputs
- `test_metrics.csv`
- `training_metadata.json`

This contract is required for downstream commands (`analyze`, `registry`, `inference` tooling).
