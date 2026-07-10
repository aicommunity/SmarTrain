from __future__ import annotations

import os
from typing import Any


def get_interactive_default(args, attr: str, fallback, baseline_cfg: dict[str, Any], baseline_key: str):
    val = getattr(args, attr, None)
    if val is not None and (fallback is None or val != fallback):
        return val
    if baseline_key in baseline_cfg:
        return baseline_cfg[baseline_key]
    return fallback


def run_interactive_train_setup(
    args,
    *,
    model_version: str,
    manual_model_entry: str,
    epochs_default: int,
    batch_default: int,
    img_size_default: int,
    ultralytics_yaml_ignored_keys: set[str],
    resolve_workspace_root_cb,
    workspace_layout_cb,
    prompt_input_cb,
    load_available_datasets_cb,
    prompt_dataset_name_cb,
    collect_available_base_runs_cb,
    print_available_base_runs_cb,
    prompt_base_run_args_yaml_cb,
    load_ultralytics_yaml_cb,
    extract_smartrain_options_cb,
    normalize_model_spec_cb,
    train_model_picker_options_cb,
    model_matches_task_cb,
    pick_model_interactive_cb,
    installed_external_provider_ids_cb,
    prompt_int_cb,
    prompt_train_device_cb,
    prompt_yes_no_cb,
    prompt_optional_int_cb,
    prompt_optional_float_cb,
    default_device_value_cb,
) -> bool:
    from prompt_toolkit.completion import WordCompleter

    print("[INFO] Interactive train mode (Enter = default).")
    installed_external = installed_external_provider_ids_cb()
    if installed_external:
        print("[INFO] Installed external providers:")
        for pid in installed_external:
            print(f"  - {pid}")

    try:
        ws = resolve_workspace_root_cb(getattr(args, "workspace", None))
    except ValueError:
        ws_raw = prompt_input_cb("Workspace path: ", default=os.getcwd()).strip()
        if not ws_raw:
            print("[ERROR] Workspace not set.")
            return False
        ws = os.path.abspath(os.path.expanduser(ws_raw))
        args.workspace = ws

    layout = workspace_layout_cb(ws)
    dataset_names = load_available_datasets_cb(layout)
    if not dataset_names:
        print(
            "[ERROR] There are no available datasets in datasets/datasets_info.json."
            "Please scan first."
        )
        return False
    args.data = prompt_dataset_name_cb(dataset_names)
    baseline_u_cfg: dict[str, Any] = {}
    baseline_sm_opts: dict[str, Any] = {}
    available_runs = collect_available_base_runs_cb(layout, args.data)
    print_available_base_runs_cb(args.data, available_runs)
    args.pretrained_run = getattr(args, "pretrained_run", None)
    args.pretrained_model = getattr(args, "pretrained_model", None)
    args.pretrained_weights = getattr(args, "pretrained_weights", None)

    baseline_args_yaml = prompt_base_run_args_yaml_cb(
        available_runs,
        default_path=str(getattr(args, "base_run_args_yaml", "") or "") or None,
    )
    args.base_run_args_yaml = baseline_args_yaml
    if baseline_args_yaml:
        try:
            baseline_profile = load_ultralytics_yaml_cb(baseline_args_yaml)
            baseline_filtered = {
                k: v for k, v in baseline_profile.items() if k not in ultralytics_yaml_ignored_keys
            }
            baseline_u_cfg, baseline_sm_opts = extract_smartrain_options_cb(baseline_filtered)
            print(f"[INFO] Baseline run used: {baseline_args_yaml}")
        except Exception as e:
            print(f"[WARNING] Failed to read args.yaml of base run: {e}")
            baseline_u_cfg, baseline_sm_opts = {}, {}

    args.ultralytics_yaml = (
        prompt_input_cb(
            "Path to external Ultralytics args.yaml (--ultralytics_yaml, empty=do not use): ",
            default=str(getattr(args, "ultralytics_yaml", "") or ""),
        ).strip()
        or None
    )
    if args.ultralytics_yaml:
        print(
            "[INFO] For --ultralytics_yaml: data/project/name/exist_ok and service path keys "
            "ignored; data is always taken from the selected dataset."
        )
    ultra_u_cfg: dict[str, Any] = {}
    ultra_sm_opts: dict[str, Any] = {}
    if args.ultralytics_yaml:
        try:
            ultra_profile = load_ultralytics_yaml_cb(args.ultralytics_yaml)
        except Exception as e:
            print(f"[ERROR] Failed to read --ultralytics_yaml: {e}")
            return False
        filtered = {k: v for k, v in ultra_profile.items() if k not in ultralytics_yaml_ignored_keys}
        ultra_u_cfg, ultra_sm_opts = extract_smartrain_options_cb(filtered)

    task_choices = ["detect", "segment", "classify", "pose", "obb"]
    if "task" in ultra_u_cfg:
        args.task = str(ultra_u_cfg["task"])
        print(f"[INFO] Task taken from --ultralytics_yaml: {args.task}")
    else:
        task_default = str(get_interactive_default(args, "task", "detect", baseline_u_cfg, "task"))
        task_completer = WordCompleter(task_choices, ignore_case=True)
        args.task = (
            prompt_input_cb(
                "Task (detect/segment/classify/pose/obb): ",
                default=task_default,
                completer=task_completer,
            ).strip()
            or task_default
        )

    if "model" in ultra_u_cfg:
        args.model = normalize_model_spec_cb(str(ultra_u_cfg["model"]), add_pt_when_missing=True)
        print(f"[INFO] Model taken from --ultralytics_yaml: {args.model}")
    else:
        model_default = normalize_model_spec_cb(
            str(get_interactive_default(args, "model", model_version, baseline_u_cfg, "model")),
            add_pt_when_missing=True,
        )
        all_options = train_model_picker_options_cb(model_default.replace(".pt", ""))
        task_options = [opt for opt in all_options if (opt == manual_model_entry or model_matches_task_cb(opt, args.task))]
        options = task_options or all_options
        default_alias = model_default.replace(".pt", "")
        if default_alias not in options:
            default_alias = options[0]
        model_choice = pick_model_interactive_cb(options, default_alias)
        if model_choice == manual_model_entry:
            model_choice = (
                prompt_input_cb(
                    "Manual model alias/path (--model): ",
                    default=model_default,
                ).strip()
                or model_default
            )
        selected_external_provider = None
        if ":" in model_choice:
            provider_part, model_part = model_choice.split(":", 1)
            if provider_part and model_part:
                selected_external_provider = provider_part
                model_choice = model_part
        if selected_external_provider:
            args.external_provider = selected_external_provider
            print(f"[INFO] External provider selected from model alias: {selected_external_provider}")
        else:
            args.external_provider = None
        args.model = normalize_model_spec_cb(model_choice, add_pt_when_missing=True)
    print(f"[INFO] Final model for launch: {args.model}")

    if "epochs" in ultra_u_cfg:
        args.epochs = int(ultra_u_cfg["epochs"])
        print(f"[INFO] Epochs taken from --ultralytics_yaml: {args.epochs}")
    else:
        args.epochs = prompt_int_cb(
            "Epoches (--epochs)",
            int(get_interactive_default(args, "epochs", epochs_default, baseline_u_cfg, "epochs")),
        )
    if "batch" in ultra_u_cfg:
        args.batch = int(ultra_u_cfg["batch"])
        print(f"[INFO] Batch taken from --ultralytics_yaml: {args.batch}")
    else:
        args.batch = prompt_int_cb(
            "Batch (--batch)",
            int(get_interactive_default(args, "batch", batch_default, baseline_u_cfg, "batch")),
        )
    if "imgsz" in ultra_u_cfg:
        args.img_size = int(ultra_u_cfg["imgsz"])
        print(f"[INFO] Image size taken from --ultralytics_yaml: {args.img_size}")
    else:
        args.img_size = prompt_int_cb(
            "Images Size (--img-size)",
            int(get_interactive_default(args, "img_size", img_size_default, baseline_u_cfg, "imgsz")),
        )
    args.device = prompt_train_device_cb(
        str(get_interactive_default(args, "device", default_device_value_cb(), baseline_u_cfg, "device"))
    )

    default_target = str(getattr(args, "target_path", None) or layout.runs)
    args.target_path = (prompt_input_cb("Run directory (--target-path): ", default=default_target).strip() or default_target)

    args.test_only = prompt_yes_no_cb("Test only without training (--test-only)?", default=bool(getattr(args, "test_only", False)))
    if args.test_only:
        model_dir_default = str(getattr(args, "model_dir", "") or "")
        while True:
            model_dir = prompt_input_cb("Path to model (--model-dir): ", default=model_dir_default).strip()
            if model_dir:
                args.model_dir = model_dir
                break
            print("[ERROR] --test-only requires --model-dir.")
    else:
        args.model_dir = getattr(args, "model_dir", None)

    args.val_imgsz = prompt_optional_int_cb(
        "Size val/test (--val-imgsz, empty=how train)",
        get_interactive_default(args, "val_imgsz", None, baseline_u_cfg, "imgsz"),
    )
    args.val_conf = prompt_optional_float_cb(
        "conf threshold (--val-conf, empty=default Ultralytics)",
        get_interactive_default(args, "val_conf", None, baseline_u_cfg, "conf"),
    )
    args.val_iou = prompt_optional_float_cb(
        "IoU threshold (--val-iou, empty=default Ultralytics)",
        get_interactive_default(args, "val_iou", None, baseline_u_cfg, "iou"),
    )

    if "weighted_sampling" in ultra_sm_opts:
        args.weighted_sampling = bool(ultra_sm_opts["weighted_sampling"])
    else:
        args.weighted_sampling = prompt_yes_no_cb(
            "Enable weighted sampling (--weighted-sampling)?",
            default=bool(get_interactive_default(args, "weighted_sampling", False, baseline_sm_opts, "weighted_sampling")),
        )
    if "clearml" in ultra_sm_opts:
        args.clearml = bool(ultra_sm_opts["clearml"])
    else:
        args.clearml = prompt_yes_no_cb(
            "Log to ClearML (--clearml)?",
            default=bool(get_interactive_default(args, "clearml", False, baseline_sm_opts, "clearml")),
        )
    if args.clearml:
        if "clearml_project" in ultra_sm_opts:
            args.clearml_project = str(ultra_sm_opts["clearml_project"]).strip() or None
        else:
            default_cm_project = str(getattr(args, "clearml_project", "") or "")
            if "clearml_project" in baseline_sm_opts:
                default_cm_project = str(baseline_sm_opts["clearml_project"] or "")
            args.clearml_project = (
                prompt_input_cb(
                    "ClearML Project (--clearml-project): ",
                    default=default_cm_project,
                ).strip()
                or None
            )
    args.non_interactive = prompt_yes_no_cb(
        "Do not ask for confirmation if the folder exists (--yes)?",
        default=bool(getattr(args, "non_interactive", False)),
    )

    if not any((args.pretrained_run, args.pretrained_model, args.pretrained_weights)):
        use_pretrained = prompt_yes_no_cb(
            "Use initialization from existing trained weights (--pretrained-*)?",
            default=False,
        )
        if use_pretrained:
            source_kind = prompt_input_cb(
                "Pretrained source type (run/model/path): ",
                default="run",
            ).strip().lower() or "run"
            if source_kind == "run":
                if available_runs:
                    for idx, row in enumerate(available_runs, start=1):
                        print(f"  {idx:>3}. {row.get('run_rel', row.get('run_dir', '-'))}")
                    selected_run = prompt_input_cb(
                        "Pretrained run (number or path): ",
                        default=str(available_runs[0].get("run_dir", "")),
                    ).strip()
                    if selected_run.isdigit():
                        n = int(selected_run)
                        if 1 <= n <= len(available_runs):
                            selected_run = str(available_runs[n - 1].get("run_dir", "")).strip()
                    args.pretrained_run = selected_run or str(available_runs[0].get("run_dir", "")).strip()
                else:
                    args.pretrained_run = prompt_input_cb(
                        "Pretrained run path (--pretrained-run): ",
                        default="",
                    ).strip() or None
            elif source_kind == "model":
                models_dir = os.path.abspath(str(getattr(layout, "models", "")))
                model_names = []
                if os.path.isdir(models_dir):
                    model_names = sorted([d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))])
                if model_names:
                    for idx, name in enumerate(model_names, start=1):
                        print(f"  {idx:>3}. {name}")
                    selected_model = prompt_input_cb(
                        "Pretrained model (number or name): ",
                        default=model_names[0],
                    ).strip()
                    if selected_model.isdigit():
                        n = int(selected_model)
                        if 1 <= n <= len(model_names):
                            selected_model = model_names[n - 1]
                    args.pretrained_model = selected_model or model_names[0]
                else:
                    args.pretrained_model = prompt_input_cb(
                        "Pretrained model name (--pretrained-model): ",
                        default="",
                    ).strip() or None
            else:
                args.pretrained_weights = (
                    prompt_input_cb("Pretrained weights path (--pretrained-weights): ", default="").strip() or None
                )
            if args.pretrained_run:
                print(f"[INFO] Pretrained run selected: {args.pretrained_run}")
            if args.pretrained_model:
                print(f"[INFO] Pretrained model selected: {args.pretrained_model}")
            if args.pretrained_weights:
                print(f"[INFO] Pretrained weights selected: {args.pretrained_weights}")
    return True

