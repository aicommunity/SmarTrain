from __future__ import annotations

import copy


def finalize_train_kwargs(ultralytics_cfg: dict, data_yaml: str, model_dir: str) -> dict:
    """Force Ultralytics train directory under ``model_dir``."""
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
    k.pop("data", None)
    k["data"] = data_yaml
    k["project"] = model_dir
    k["name"] = "train-ultralytics"
    k["exist_ok"] = False
    k.setdefault("mode", "train")
    if overwritten:
        print(
            "[WARNING] Train service keys have been forced to be overridden: "
            + ", ".join(sorted(set(overwritten)))
        )
    return k


def load_ultralytics_yaml(path: str | None, *, load_train_profile_cb) -> dict:
    if not path:
        return {}
    raw = load_train_profile_cb(path)
    if not isinstance(raw, dict):
        return {}
    return raw

