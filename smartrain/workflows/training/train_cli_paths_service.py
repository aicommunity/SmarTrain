from __future__ import annotations

import os


def resolve_cli_paths_with_profile(
    args,
    u_cfg: dict,
    *,
    workspace_env_var: str,
    resolve_workspace_root_cb,
    workspace_layout_cb,
    resolve_training_data_path_cb,
    resolve_profile_data_path_cb,
    dataset_root_from_data_yaml_cb,
) -> tuple[str | None, str, str]:
    """
    workspace, dataset_root (directory with data.yaml), target_base.
    """
    try:
        ws = resolve_workspace_root_cb(args.workspace)
    except ValueError:
        ws = None

    if ws is not None:
        layout = workspace_layout_cb(ws)
        os.makedirs(layout.runs, exist_ok=True)
        if args.data is not None:
            dataset_path = resolve_training_data_path_cb(layout, args.data)
        elif u_cfg.get("data"):
            yp = resolve_profile_data_path_cb(str(u_cfg["data"]))
            dataset_path = dataset_root_from_data_yaml_cb(yp)
        else:
            raise ValueError(
                "When using workspace, specify --data or the data: field in the --config profile."
            )
        if args.target_path is not None:
            target_base = os.path.abspath(os.path.expanduser(args.target_path))
        else:
            target_base = layout.runs
        return ws, dataset_path, target_base

    if args.data is not None:
        dataset_path = os.path.abspath(os.path.expanduser(args.data))
    elif u_cfg.get("data"):
        yp = resolve_profile_data_path_cb(str(u_cfg["data"]))
        dataset_path = dataset_root_from_data_yaml_cb(yp)
    else:
        raise ValueError(
            f"Specify --workspace (or {workspace_env_var}) and --data (or data in YAML), "
            "or without workspace - --data and --target-path."
        )

    if args.target_path is None:
        raise ValueError(
            f"Without workspace, specify --target-path (base run directory) or specify {workspace_env_var}."
        )
    target_base = os.path.abspath(os.path.expanduser(args.target_path))
    return None, dataset_path, target_base

