from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from smartrain.core.training.train_model_resolver import TrainModelResolver


def normalize_model_spec(
    spec: Any,
    *,
    default_model: str,
    add_pt_when_missing: bool = False,
) -> str:
    resolver = TrainModelResolver()
    return resolver.resolve(
        None if spec is None else str(spec),
        default_model=default_model,
        add_pt_when_missing=add_pt_when_missing,
    ).normalized


def extract_effective_loaded_model(model: Any, fallback: str) -> str:
    for candidate in (
        getattr(model, "ckpt_path", None),
        getattr(model, "model_name", None),
        (getattr(model, "overrides", {}) or {}).get("model"),
    ):
        if candidate:
            return str(candidate)
    return str(fallback)


def extract_model_family_scale(spec: str) -> tuple[str, str] | None:
    token = Path(str(spec)).name.lower()
    if token.endswith(".pt"):
        token = token[:-3]
    for suffix in ("-seg", "-cls", "-pose", "-obb"):
        if token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    match = re.match(r"^(yolo(?:v)?\d+)([nslmx])$", token)
    if not match:
        return None
    return match.group(1), match.group(2)

