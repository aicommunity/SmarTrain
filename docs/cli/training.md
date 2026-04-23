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
Also ignored from `--ultralytics_yaml`: `project`, `name`, `exist_ok`, `cfg`, `device`, `model_dir`, `target_path`, `workspace`.

That is, command line options always have the highest priority.

Model selection:

- To list aliases supported by the default training backend, run `smartrain info`.
- In interactive mode, `train` offers model selection from this list and provides `<manual>` for custom values.
- `--model` accepts both YOLO aliases and explicit weights path; for plain YOLO aliases `.pt` is added automatically.

## `clearml-upload`

A separate command to load run artifacts into ClearML:

```bash
smartrain clearml-upload /path/to/run_folder --project MyProject
```

During training, you can enable integration with the `--clearml` flag.
