from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from smartrain.services.analyze.metrics_reader import (
    _infer_model_from_args_yaml,
    _infer_model_from_run_dir_name,
)


@dataclass(frozen=True, slots=True)
class RunLegendRow:
    index: int
    short_label: str
    enriched_label: str
    architecture: str
    dataset_label: str
    dataset_name: str
    epochs: str
    batch: str
    run_name: str
    run_dir: str
    role: str


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


def run_dir_display_suffix(run_name: str, *, max_tail: int = 8) -> str:
    name = str(run_name or "").strip()
    if not name:
        return ""
    m = re.search(r"-([a-f0-9]{6,})$", name, flags=re.IGNORECASE)
    if m:
        tail = m.group(1)
        return tail[-max_tail:] if len(tail) > max_tail else tail
    return name[-max_tail:] if len(name) > max_tail else name


def format_run_display_label(index: int, short_model: str) -> str:
    model = str(short_model or "model").strip() or "model"
    return f"M{int(index)} {model}"


def format_enriched_display_label(
    index: int,
    short_model: str,
    *,
    dataset_label: str = "",
    epochs: Any = None,
    batch: Any = None,
    collision: bool = False,
) -> str:
    short = format_run_display_label(index, short_model)
    if not collision:
        return short
    parts = [f"M{int(index)}", str(short_model or "model").strip() or "model"]
    ds = str(dataset_label or "").strip()
    if ds:
        parts.append(ds)
    if epochs is not None and str(epochs).strip() not in {"", "nan", "None"}:
        try:
            parts.append(f"{int(float(epochs))}ep")
        except (TypeError, ValueError):
            parts.append(f"{epochs}ep")
    if batch is not None and str(batch).strip() not in {"", "nan", "None"}:
        try:
            parts.append(f"b{int(float(batch))}")
        except (TypeError, ValueError):
            parts.append(f"b{batch}")
    return " · ".join(parts)


def _read_run_training_fields(
    run_dir: str,
    *,
    build_run_record_cb: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    from smartrain.services.analyze.ultralytics_test_artifacts import build_ultralytics_run_info

    model_hint = ""
    dataset_name = ""
    if build_run_record_cb is not None:
        try:
            rec = build_run_record_cb(run_dir)
            model_hint = str(getattr(rec, "model", None) or "").strip()
            dataset_name = str(getattr(rec, "dataset_name", None) or "").strip()
        except Exception:
            pass
    info = build_ultralytics_run_info(run_dir, model_fallback=model_hint or None)
    if not dataset_name:
        dataset_name = str(info.get("dataset_name") or "").strip()
    return {
        "model_hint": model_hint,
        "dataset_name": dataset_name,
        "epochs": info.get("epochs"),
        "batch_size": info.get("batch_size"),
        "train_image_size": info.get("train_image_size"),
        "val_imgsz": info.get("val_imgsz"),
    }


def build_run_legend_rows(
    run_dirs: list[str],
    *,
    baseline: str = "",
    dataset_labels: dict[str, str] | None = None,
    build_run_record_cb: Callable[[str], Any] | None = None,
) -> list[RunLegendRow]:
    dataset_labels = dataset_labels or {}
    baseline_abs = os.path.abspath(baseline.rstrip(os.sep)) if baseline else ""
    short_models: list[str] = []
    rows_meta: list[dict[str, Any]] = []
    for idx, run_dir in enumerate(run_dirs, start=1):
        run_dir_abs = os.path.abspath(run_dir.rstrip(os.sep))
        run_name = os.path.basename(run_dir_abs)
        fields = _read_run_training_fields(run_dir_abs, build_run_record_cb=build_run_record_cb)
        short_model = infer_short_model_name(run_dir_abs, model_hint=fields.get("model_hint") or None)
        short_models.append(short_model)
        dataset_name = str(fields.get("dataset_name") or "").strip()
        dataset_label = dataset_labels.get(dataset_name, dataset_name)
        role = "baseline" if baseline_abs and run_dir_abs == baseline_abs else "candidate"
        rows_meta.append(
            {
                "index": idx,
                "run_dir_abs": run_dir_abs,
                "run_name": run_name,
                "short_model": short_model,
                "dataset_name": dataset_name,
                "dataset_label": dataset_label,
                "epochs": fields.get("epochs"),
                "batch_size": fields.get("batch_size"),
                "role": role,
            }
        )
    collision_models = {m for m in short_models if short_models.count(m) > 1}
    out: list[RunLegendRow] = []
    for meta in rows_meta:
        collision = meta["short_model"] in collision_models
        enriched = format_enriched_display_label(
            meta["index"],
            meta["short_model"],
            dataset_label=str(meta["dataset_label"] or ""),
            epochs=meta.get("epochs"),
            batch=meta.get("batch_size"),
            collision=collision,
        )
        short_label = enriched if collision else format_run_display_label(meta["index"], meta["short_model"])
        epochs_txt = ""
        if meta.get("epochs") is not None and str(meta.get("epochs")).strip() not in {"", "nan", "None"}:
            try:
                epochs_txt = str(int(float(meta["epochs"])))
            except (TypeError, ValueError):
                epochs_txt = str(meta["epochs"])
        batch_txt = ""
        if meta.get("batch_size") is not None and str(meta.get("batch_size")).strip() not in {"", "nan", "None"}:
            try:
                batch_txt = str(int(float(meta["batch_size"])))
            except (TypeError, ValueError):
                batch_txt = str(meta["batch_size"])
        out.append(
            RunLegendRow(
                index=int(meta["index"]),
                short_label=short_label,
                enriched_label=enriched,
                architecture=str(meta["short_model"]),
                dataset_label=str(meta["dataset_label"] or ""),
                dataset_name=str(meta["dataset_name"] or ""),
                epochs=epochs_txt,
                batch=batch_txt,
                run_name=str(meta["run_name"]),
                run_dir=str(meta["run_dir_abs"]),
                role=str(meta["role"]),
            )
        )
    return out


def build_run_display_labels(
    run_dirs: list[str],
    *,
    build_run_record_cb: Callable[[str], Any] | None = None,
    dataset_labels: dict[str, str] | None = None,
    baseline: str = "",
) -> dict[str, str]:
    """Map run_dir, basename, and long model names to display labels."""
    legend = build_run_legend_rows(
        run_dirs,
        baseline=baseline,
        dataset_labels=dataset_labels,
        build_run_record_cb=build_run_record_cb,
    )
    out: dict[str, str] = {}
    for row in legend:
        out[row.run_dir] = row.short_label
        out[row.run_name] = row.short_label
        out[row.enriched_label] = row.short_label
        out[f"M{row.index}"] = row.short_label
    if build_run_record_cb is not None:
        for row in legend:
            try:
                rec = build_run_record_cb(row.run_dir)
                model_hint = str(getattr(rec, "model", None) or "").strip()
                if model_hint and model_hint not in out:
                    out[model_hint] = row.short_label
            except Exception:
                continue
    return out


def build_report_labels_context(
    run_dirs: list[str],
    *,
    baseline: str = "",
    build_run_record_cb: Callable[[str], Any] | None = None,
) -> tuple[dict[str, str], list[RunLegendRow], dict[str, str]]:
    dataset_to_idx: dict[str, int] = {}
    dataset_counter = 1
    dataset_labels: dict[str, str] = {}
    for rd in run_dirs:
        fields = _read_run_training_fields(rd, build_run_record_cb=build_run_record_cb)
        dataset_name = str(fields.get("dataset_name") or "").strip()
        if not dataset_name:
            continue
        if dataset_name not in dataset_to_idx:
            dataset_to_idx[dataset_name] = dataset_counter
            dataset_counter += 1
        dataset_labels[dataset_name] = f"D{dataset_to_idx[dataset_name]}"
    legend = build_run_legend_rows(
        run_dirs,
        baseline=baseline,
        dataset_labels=dataset_labels,
        build_run_record_cb=build_run_record_cb,
    )
    abbreviations = build_run_display_labels(
        run_dirs,
        build_run_record_cb=build_run_record_cb,
        dataset_labels=dataset_labels,
        baseline=baseline,
    )
    for name, label in dataset_labels.items():
        abbreviations.setdefault(name, label)
    return abbreviations, legend, dataset_labels
