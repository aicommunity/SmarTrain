from __future__ import annotations

import os
from typing import Any, Callable

import yaml

from smartrain.core.runtime.run_artifacts import run_tmp_dir
from smartrain.core.runtime.workspace_paths import resolve_workspace_root


def collect_data_yaml_candidates_for_run(
    run_dir: str,
    workspace_cli: str | None,
    *,
    unified_read_enabled: bool,
    dataset_name_resolver: Callable[[str], str | None],
    metadata_loader: Callable[[str], dict[str, Any]] | None = None,
) -> list[tuple[str, str]]:
    rd = os.path.abspath(run_dir)
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(path: str | None, source: str) -> None:
        if not path:
            return
        abs_path = os.path.abspath(path)
        if abs_path in seen or not os.path.isfile(abs_path):
            return
        seen.add(abs_path)
        out.append((abs_path, source))

    args_yaml = os.path.join(rd, "train-ultralytics", "args.yaml")
    if not os.path.isfile(args_yaml):
        args_yaml = os.path.join(rd, "train", "args.yaml")
    if os.path.isfile(args_yaml):
        try:
            with open(args_yaml, "r", encoding="utf-8") as f:
                payload = yaml.safe_load(f) or {}
            data_val = payload.get("data")
            if isinstance(data_val, str) and data_val.strip():
                path_value = os.path.expanduser(data_val.strip())
                if os.path.isabs(path_value) and os.path.isfile(path_value):
                    _add(path_value, "train/args.yaml:data")
                run_relative_path = os.path.abspath(os.path.join(rd, path_value))
                if os.path.isfile(run_relative_path):
                    _add(run_relative_path, "train/args.yaml:data(run-relative)")
                try:
                    workspace = resolve_workspace_root(workspace_cli)
                    workspace_relative_path = os.path.abspath(os.path.join(workspace, path_value))
                    if os.path.isfile(workspace_relative_path):
                        _add(workspace_relative_path, "train/args.yaml:data(workspace-relative)")
                except ValueError:
                    pass
        except Exception:
            pass

    runtime_yaml = os.path.join(str(run_tmp_dir(rd)), "_runtime_data_train.yaml")
    if os.path.isfile(runtime_yaml):
        _add(runtime_yaml, "_runtime_data_train.yaml")
    else:
        legacy_runtime_yaml = os.path.join(rd, "_runtime_data_train.yaml")
        if os.path.isfile(legacy_runtime_yaml):
            _add(legacy_runtime_yaml, "_runtime_data_train.yaml(legacy)")

    dataset_payload: dict[str, Any] = {}
    if unified_read_enabled:
        resolved_name = dataset_name_resolver(rd)
        if resolved_name:
            dataset_payload["name"] = resolved_name
    else:
        if metadata_loader is None:
            return out
        try:
            metadata = metadata_loader(rd)
        except Exception:
            return out
        dataset_payload = (metadata.get("training_info") or {}).get("dataset") or {}
    try:
        workspace = resolve_workspace_root(workspace_cli)
    except ValueError:
        workspace = os.getcwd()
    for key in ("path_under_workspace", "path_absolute", "path_relative"):
        value = dataset_payload.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        if key == "path_under_workspace":
            candidate = os.path.join(workspace, value, "data.yaml")
        elif key == "path_absolute":
            candidate = os.path.join(os.path.expanduser(value), "data.yaml")
        else:
            candidate = os.path.join(rd, value, "data.yaml")
        if os.path.isfile(candidate):
            _add(candidate, f"training_metadata.dataset.{key}")
    dataset_name = dataset_payload.get("name")
    if isinstance(dataset_name, str) and dataset_name.strip():
        candidate = os.path.join(workspace, "datasets", dataset_name.strip(), "data.yaml")
        if os.path.isfile(candidate):
            _add(candidate, "training_metadata.dataset.name -> workspace/datasets")
    return out
