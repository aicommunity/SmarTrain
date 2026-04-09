> Russian version: [../ru/cli/training.md](../ru/cli/training.md)

# CLI: training

## `train`

Core training/validation team.

```bash
smartrain train --data my_dataset -y
smartrain train --test-only --model-dir /path/to/run --data /path/to/dataset
```

Parameter sources and priority:

`CLI > --ultralytics_yaml > --config > defaults`

The `data` field from `--ultralytics_yaml` is ignored, the selected `--data` is used.

That is, command line options always have the highest priority.

## `clearml-upload`

A separate command to load run artifacts into ClearML:

```bash
smartrain clearml-upload /path/to/run_folder --project MyProject
```

During training, you can enable integration with the `--clearml` flag.
