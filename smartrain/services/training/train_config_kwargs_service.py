from __future__ import annotations

import copy

_PATH_LIKE_FORCED_KEYS = frozenset({"save_dir", "runs_dir", "output_dir"})


def finalize_train_kwargs(ultralytics_cfg: dict, data_yaml: str, model_dir: str) -> dict:
    """Force Ultralytics train directory under ``model_dir``."""
    from smartrain.core.runtime.run_artifacts import remove_empty_train_ultralytics_dir

    remove_empty_train_ultralytics_dir(model_dir)
    k = copy.deepcopy(ultralytics_cfg)
    overwritten: list[str] = []
    if "data" in k:
        overwritten.append("data")
    if "project" in k:
        overwritten.append("project")
    if "name" in k:
        overwritten.append("name")
    if "exist_ok" in k:
        overwritten.append("exist_ok")
    sanitized_path_like: list[str] = []
    for key in sorted(_PATH_LIKE_FORCED_KEYS):
        if key in k:
            sanitized_path_like.append(key)
            k.pop(key, None)
    k.pop("data", None)
    k["data"] = data_yaml
    k["project"] = model_dir
    k["name"] = "train-ultralytics"
    k["exist_ok"] = True
    k.setdefault("mode", "train")
    if overwritten:
        print(
            "[WARNING] Train service keys have been forced to be overridden: "
            + ", ".join(sorted(set(overwritten)))
        )
    if sanitized_path_like:
        print(
            "[WARNING] Train service path-like keys ignored for safety: "
            + ", ".join(sanitized_path_like)
        )
    return k


def load_ultralytics_yaml(path: str | None, *, load_train_profile_cb) -> dict:
    if not path:
        return {}
    raw = load_train_profile_cb(path)
    if not isinstance(raw, dict):
        return {}
    return raw

