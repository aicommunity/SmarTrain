from __future__ import annotations

from typing import Any

from smartrain.core.training.train_profile import extract_smartrain_options

_ULTRALYTICS_YAML_IGNORED_KEYS = frozenset(
    {
        "data",
        "project",
        "name",
        "exist_ok",
        "cfg",
        "device",
        "model_dir",
        "target_path",
        "workspace",
    }
)


def merge_sources_with_priority(
    *,
    config_profile: dict[str, Any],
    ultralytics_profile: dict[str, Any],
    args: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    u_cfg, sm_opts = extract_smartrain_options(config_profile)

    if ultralytics_profile:
        ignored = sorted(k for k in ultralytics_profile.keys() if k in _ULTRALYTICS_YAML_IGNORED_KEYS)
        if ignored:
            print("[WARNING] --ultralytics_yaml: keys ignored: " + ", ".join(ignored))
        filtered = {k: v for k, v in ultralytics_profile.items() if k not in _ULTRALYTICS_YAML_IGNORED_KEYS}
        u_from_ultra, sm_from_ultra = extract_smartrain_options(filtered)
        cli_key_map = {
            "model": "model",
            "epochs": "epochs",
            "batch": "batch",
            "imgsz": "img_size",
            "task": "task",
        }
        overridden_by_cli: list[str] = []
        for yaml_key, cli_attr in cli_key_map.items():
            if yaml_key in u_from_ultra and getattr(args, cli_attr, None) is not None:
                overridden_by_cli.append(yaml_key)
        if overridden_by_cli:
            print(
                "[WARNING] --ultralytics_yaml: The following keys will be overridden by the CLI: "
                + ", ".join(sorted(overridden_by_cli))
            )
        u_cfg.update(u_from_ultra)
        sm_opts.update(sm_from_ultra)
    return u_cfg, sm_opts
