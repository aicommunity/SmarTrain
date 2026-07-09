from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from smartrain.core.runtime.run_artifacts import materialize_preferred_run_model, preferred_run_model_path
from smartrain.core.runtime.workspace_paths import WorkspaceLayout
from smartrain.services.inference_runtime_helpers import resolve_model_from_name


def handle_aux_train_commands(
    argv: list[str],
    *,
    run_resume_command_cb: Callable[[list[str]], int],
    run_calc_confidence_command_cb: Callable[[list[str]], int],
) -> int | None:
    if argv and argv[0] == "resume":
        return run_resume_command_cb(argv[1:])
    if argv and argv[0] == "calc-confidence":
        return run_calc_confidence_command_cb(argv[1:])
    return None


def run_train_cli_pipeline(
    argv: list[str],
    *,
    request: Any,
    parse_args_cb: Callable[[list[str]], Any],
    apply_external_provider_defaults_cb: Callable[[Any], None],
    list_provider_specs_cb: Callable[[], list[Any]],
    parse_external_model_ref_cb: Callable[[Any], Any],
    validate_external_model_ref_cb: Callable[..., Any],
    build_train_arg_parser_cb: Callable[[], Any],
    run_interactive_train_setup_cb: Callable[[Any], bool],
    emit_replay_cb: Callable[..., str | None],
    load_train_profile_cb: Callable[[str], dict[str, Any]],
    load_ultralytics_yaml_cb: Callable[[str | None], dict[str, Any]],
    merge_sources_with_priority_cb: Callable[..., tuple[dict[str, Any], dict[str, Any]]],
    merge_cli_into_ultralytics_cfg_cb: Callable[..., None],
    apply_cli_smartrain_overrides_cb: Callable[[dict[str, Any], Any], None],
    resolve_device_request_cb: Callable[[Any], str | None],
    resolve_cli_paths_with_profile_cb: Callable[[Any, dict[str, Any]], tuple[str, str, str]],
    normalize_model_spec_cb: Callable[..., str],
    ensure_device_available_or_raise_cb: Callable[[str | None], None],
    device_display_name_cb: Callable[[str | None], str],
    run_train_after_setup_cb: Callable[..., int | None],
    default_device_value_cb: Callable[[], str],
    model_version_default: str,
    epochs_default: int,
    batch_default: int,
    img_size_default: int,
) -> int | None:
    args = parse_args_cb(argv)
    apply_external_provider_defaults_cb(args)
    known_provider_ids = {spec.id for spec in list_provider_specs_cb()}
    try:
        parsed_ref = validate_external_model_ref_cb(
            parse_external_model_ref_cb(getattr(args, "model", None)),
            known_provider_ids=known_provider_ids,
        )
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 2
    if parsed_ref.is_external and parsed_ref.provider_id and not getattr(args, "external_provider", None):
        args.external_provider = parsed_ref.provider_id
        args.model = parsed_ref.model_ref
        print(f"[INFO] External provider inferred from --model: {parsed_ref.provider_id}")
    parser = build_train_arg_parser_cb()
    interactive_mode = request.interactive_allowed
    replay_cmd = None
    if interactive_mode:
        import sys

        if not sys.stdin.isatty():
            print(
                "[ERROR] Interactive train mode requires a terminal (TTY)."
                "Either run in terminal or pass arguments."
            )
            return None
        try:
            ok = run_interactive_train_setup_cb(args)
        except Exception as e:
            print(f"[ERROR] Train interactive mode error: {e}")
            return None
        if not ok:
            return None
        request.interactive_used = True
        replay_cmd = emit_replay_cb(command_name="train", parser=parser, args=args, stage="before launch")

    profile = load_train_profile_cb(args.config) if args.config else {}
    ultra_profile = load_ultralytics_yaml_cb(getattr(args, "ultralytics_yaml", None))
    u_cfg, sm_opts = merge_sources_with_priority_cb(
        config_profile=profile,
        ultralytics_profile=ultra_profile,
        args=args,
    )
    merge_cli_into_ultralytics_cfg_cb(
        u_cfg,
        model=getattr(args, "model", None),
        epochs=getattr(args, "epochs", None),
        batch=getattr(args, "batch", None),
        imgsz=getattr(args, "img_size", None),
        task=getattr(args, "task", None),
        device=getattr(args, "device", None),
        defaults={
            "model": model_version_default,
            "epochs": epochs_default,
            "batch": batch_default,
            "imgsz": img_size_default,
            "task": "detect",
            "device": default_device_value_cb(),
        },
    )
    apply_cli_smartrain_overrides_cb(sm_opts, args)
    u_cfg["device"] = resolve_device_request_cb(u_cfg.get("device"))
    try:
        workspace_root, data, target_dir = resolve_cli_paths_with_profile_cb(args, u_cfg)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return None

    u_cfg.pop("data", None)

    pretrained_flags = [
        bool(getattr(args, "pretrained_run", None)),
        bool(getattr(args, "pretrained_model", None)),
        bool(getattr(args, "pretrained_weights", None)),
    ]
    if sum(1 for x in pretrained_flags if x) > 1:
        print("[ERROR] Use only one of --pretrained-run / --pretrained-model / --pretrained-weights.")
        return 2
    pretrained_path = None
    if getattr(args, "pretrained_weights", None):
        raw = str(args.pretrained_weights).strip()
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (Path(workspace_root) / p).resolve()
        pretrained_path = str(p)
    elif getattr(args, "pretrained_run", None):
        layout = WorkspaceLayout(workspace_root)
        raw = str(args.pretrained_run).strip()
        run_dir = Path(raw).expanduser()
        if not run_dir.is_absolute():
            run_dir = (Path(layout.runs) / raw).resolve()
        best_pt = Path(preferred_run_model_path(str(run_dir), ".pt"))
        if not best_pt.is_file():
            materialized = materialize_preferred_run_model(str(run_dir), ext=".pt", move=True, normalize_metadata=True)
            if materialized is not None:
                best_pt = Path(materialized)
        pretrained_path = str(best_pt)
    elif getattr(args, "pretrained_model", None):
        layout = WorkspaceLayout(workspace_root)
        model_path, _ = resolve_model_from_name(layout, str(args.pretrained_model).strip())
        pretrained_path = str(model_path)
    if pretrained_path:
        if not os.path.isfile(pretrained_path):
            print(f"[ERROR] Pretrained weights not found: {pretrained_path}")
            return 2
        args.model = pretrained_path
        u_cfg["model"] = pretrained_path
        if getattr(args, "external_provider", None):
            print("[WARNING] --external-provider ignored because pretrained .pt init is selected.")
            args.external_provider = None
        print(f"[INFO] Using pretrained initialization: {pretrained_path}")

    model_version = normalize_model_spec_cb(u_cfg.get("model", model_version_default), add_pt_when_missing=True)
    u_cfg["model"] = model_version
    ensure_device_available_or_raise_cb(str(u_cfg.get("device")) if u_cfg.get("device") is not None else None)
    print(
        f"[INFO] Train device: "
        + device_display_name_cb(str(u_cfg.get("device")) if u_cfg.get("device") is not None else None)
    )
    epochs = int(u_cfg.get("epochs", epochs_default))
    batch = int(u_cfg.get("batch", batch_default))
    img_size = u_cfg.get("imgsz", img_size_default)
    try:
        img_size = int(img_size) if img_size is not None else img_size_default
    except (TypeError, ValueError):
        img_size = img_size_default

    return run_train_after_setup_cb(
        args=args,
        request=request,
        parser=parser,
        u_cfg=u_cfg,
        sm_opts=sm_opts,
        workspace_root=workspace_root,
        data=data,
        target_dir=target_dir,
        model_version=model_version,
        epochs=epochs,
        batch=batch,
        img_size=img_size,
        replay_cmd=replay_cmd,
    )
