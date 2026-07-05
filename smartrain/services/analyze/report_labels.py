from __future__ import annotations

import os
import re
from typing import Any, Callable

from smartrain.services.analyze.metrics_reader import (
    _infer_model_from_args_yaml,
    _infer_model_from_run_dir_name,
)


def infer_short_model_name(run_dir: str, *, model_hint: str | None = None) -> str:
    """Return a compact YOLO model token (e.g. yolo11n) for display labels."""
    from_args = _infer_model_from_args_yaml(run_dir)
    if from_args:
        return from_args
    from_name = _infer_model_from_run_dir_name(run_dir)
    if from_name:
        return from_name
    hint = str(model_hint or "").strip()
    if hint:
        m = re.search(r"(yolo[a-z0-9]*[nslmx](?:-(?:seg|cls|pose|obb))?)", hint, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower()
        if len(hint) <= 16:
            return hint
    run_name = os.path.basename(os.path.abspath(run_dir.rstrip(os.sep)))
    if run_name.startswith("detect_"):
        token = run_name[len("detect_") :].split("_", 1)[0]
        if token:
            return token.lower()
    return run_name[:16] if run_name else "model"


def format_run_display_label(index: int, short_model: str) -> str:
    model = str(short_model or "model").strip() or "model"
    return f"M{int(index)} {model}"


def build_run_display_labels(
    run_dirs: list[str],
    *,
    build_run_record_cb: Callable[[str], Any] | None = None,
) -> dict[str, str]:
    """Map run_dir, basename, and long model names to ``M{n} {short_model}`` labels."""
    out: dict[str, str] = {}
    for idx, run_dir in enumerate(run_dirs, start=1):
        run_dir_abs = os.path.abspath(run_dir.rstrip(os.sep))
        run_name = os.path.basename(run_dir_abs)
        model_hint = ""
        if build_run_record_cb is not None:
            try:
                rec = build_run_record_cb(run_dir_abs)
                model_hint = str(getattr(rec, "model", None) or "").strip()
            except Exception:
                model_hint = ""
        short_model = infer_short_model_name(run_dir_abs, model_hint=model_hint or None)
        label = format_run_display_label(idx, short_model)
        out[run_dir_abs] = label
        out[run_name] = label
        if model_hint and model_hint not in out:
            out[model_hint] = label
    return out
