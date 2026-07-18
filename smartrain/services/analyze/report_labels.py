from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from smartrain.services.analyze.metrics_reader import (
    _infer_model_from_args_yaml,
    _infer_model_from_run_dir_name,
)

_MODEL_FILE_EXT_PRIORITY = (".pt", ".onnx", ".engine", ".torchscript", ".openvino")


@dataclass(frozen=True, slots=True)
class RunModelIdentity:
    weight_stem: str
    model_files: tuple[str, ...]


def resolve_run_model_identity(run_dir: str) -> RunModelIdentity:
    """Resolve release weight stem and sibling export filenames under ``models/``."""
    from smartrain.core.runtime.run_artifacts import preferred_run_model_path, resolve_run_weights_stem

    root = str(run_dir or "").strip()
    if not root:
        return RunModelIdentity(weight_stem="", model_files=())
    try:
        stem = str(resolve_run_weights_stem(root) or "").strip()
    except Exception:
        stem = ""
    if not stem:
        try:
            preferred = Path(preferred_run_model_path(root, ".pt"))
            stem = preferred.stem
        except Exception:
            stem = ""
    if not stem:
        return RunModelIdentity(weight_stem="", model_files=())

    models_dir = Path(root).expanduser() / "models"
    found: dict[str, str] = {}
    if models_dir.is_dir():
        for path in sorted(models_dir.iterdir()):
            if not path.is_file():
                continue
            if path.stem != stem:
                continue
            found[path.suffix.lower()] = path.name
    if ".pt" not in found:
        found[".pt"] = f"{stem}.pt"

    ordered: list[str] = []
    for ext in _MODEL_FILE_EXT_PRIORITY:
        name = found.pop(ext, None)
        if name:
            ordered.append(name)
    for ext in sorted(found):
        ordered.append(found[ext])
    return RunModelIdentity(weight_stem=stem, model_files=tuple(ordered))


def load_dataset_class_names(data_yaml: str) -> dict[int, str]:
    """Load class id → name mapping from a YOLO ``data.yaml``."""
    path = str(data_yaml or "").strip()
    if not path or not os.path.isfile(path):
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    names = payload.get("names")
    if isinstance(names, dict):
        out: dict[int, str] = {}
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def resolve_run_class_names(
    run_dir: str,
    data_yaml: str | None = None,
    *,
    run_data_yaml_map: dict[str, str] | None = None,
    workspace_root: str | None = None,
) -> list[tuple[int, str]]:
    """Return class names ordered by output index for a run."""
    yaml_path = str(data_yaml or "").strip()
    if not yaml_path and run_data_yaml_map:
        yaml_path = _lookup_run_data_yaml(run_dir, run_data_yaml_map)
    if yaml_path and workspace_root and not os.path.isabs(yaml_path) and not os.path.isfile(yaml_path):
        candidate = os.path.join(workspace_root, yaml_path)
        if os.path.isfile(candidate):
            yaml_path = candidate
    names = load_dataset_class_names(yaml_path)
    return [(cls_id, names[cls_id]) for cls_id in sorted(names)]


def _lookup_run_data_yaml(run_dir: str, run_data_yaml_map: dict[str, str]) -> str:
    if not isinstance(run_data_yaml_map, dict) or not run_data_yaml_map:
        return ""
    raw = str(run_dir or "").strip()
    if not raw:
        return ""
    candidates = [raw, os.path.normpath(raw), os.path.abspath(raw) if os.path.exists(raw) else ""]
    basename = os.path.basename(raw.rstrip("/\\"))
    if basename:
        candidates.append(basename)
    for key, value in run_data_yaml_map.items():
        key_s = str(key or "").strip()
        if not key_s:
            continue
        if key_s in candidates or os.path.normpath(key_s) in candidates:
            return str(value or "").strip()
        if basename and os.path.basename(key_s.rstrip("/\\")) == basename:
            return str(value or "").strip()
    return str(run_data_yaml_map.get(raw) or "").strip()


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
    image_size: str
    run_name: str
    run_dir: str
    role: str
    comment: str = ""


def _format_imgsz_token(train_image_size: Any, val_imgsz: Any) -> str:
    train_txt = ""
    val_txt = ""
    for raw, slot in ((train_image_size, "train"), (val_imgsz, "val")):
        if raw is None or str(raw).strip() in {"", "nan", "None"}:
            continue
        try:
            val = int(float(raw))
        except (TypeError, ValueError):
            continue
        if slot == "train":
            train_txt = str(val)
        else:
            val_txt = str(val)
    if train_txt and val_txt and train_txt != val_txt:
        return f"i{train_txt}/{val_txt}"
    token = train_txt or val_txt
    return f"i{token}" if token else ""


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
    train_image_size: Any = None,
    val_imgsz: Any = None,
    collision: bool = False,
    run_name: str = "",
) -> str:
    short = format_run_display_label(index, short_model)
    if not collision:
        return short
    model = str(short_model or "model").strip() or "model"
    parts = [f"M{int(index)}", model]
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
    imgsz_token = _format_imgsz_token(train_image_size, val_imgsz)
    if imgsz_token:
        parts.append(imgsz_token)
    if len(parts) <= 2 and run_name:
        tail = run_dir_display_suffix(run_name)
        if tail:
            parts.append(tail)
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
    workspace_root: str | None = None,
) -> list[RunLegendRow]:
    from smartrain.core.runtime.path_portable import resolve_workspace_or_abs_path

    dataset_labels = dataset_labels or {}
    baseline_abs = (
        resolve_workspace_or_abs_path(workspace_root, baseline.rstrip("/\\")) if baseline else ""
    )
    short_models: list[str] = []
    rows_meta: list[dict[str, Any]] = []
    for idx, run_dir in enumerate(run_dirs, start=1):
        run_dir_abs = resolve_workspace_or_abs_path(workspace_root, run_dir.rstrip("/\\"))
        run_name = os.path.basename(run_dir_abs.rstrip("/\\"))
        fields = _read_run_training_fields(run_dir_abs, build_run_record_cb=build_run_record_cb)
        from smartrain.services.models.release_models_manifest import release_comment_for_run_dir

        release_comment = release_comment_for_run_dir(run_dir_abs)
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
                "train_image_size": fields.get("train_image_size"),
                "val_imgsz": fields.get("val_imgsz"),
                "role": role,
                "comment": release_comment,
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
            train_image_size=meta.get("train_image_size"),
            val_imgsz=meta.get("val_imgsz"),
            collision=collision,
            run_name=str(meta["run_name"]),
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
        imgsz_txt = _format_imgsz_token(meta.get("train_image_size"), meta.get("val_imgsz")).lstrip("i")
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
                image_size=imgsz_txt,
                run_name=str(meta["run_name"]),
                run_dir=str(meta["run_dir_abs"]),
                role=str(meta["role"]),
                comment=str(meta.get("comment") or ""),
            )
        )
    return out


def build_run_display_labels(
    run_dirs: list[str],
    *,
    build_run_record_cb: Callable[[str], Any] | None = None,
    dataset_labels: dict[str, str] | None = None,
    baseline: str = "",
    workspace_root: str | None = None,
) -> dict[str, str]:
    """Map run_dir, basename, and long model names to display labels."""
    legend = build_run_legend_rows(
        run_dirs,
        baseline=baseline,
        dataset_labels=dataset_labels,
        build_run_record_cb=build_run_record_cb,
        workspace_root=workspace_root,
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
    workspace_root: str | None = None,
) -> tuple[dict[str, str], list[RunLegendRow], dict[str, str]]:
    from smartrain.core.runtime.path_portable import resolve_workspace_or_abs_path

    dataset_to_idx: dict[str, int] = {}
    dataset_counter = 1
    dataset_labels: dict[str, str] = {}
    for rd in run_dirs:
        resolved = resolve_workspace_or_abs_path(workspace_root, rd)
        fields = _read_run_training_fields(resolved, build_run_record_cb=build_run_record_cb)
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
        workspace_root=workspace_root,
    )
    abbreviations = build_run_display_labels(
        run_dirs,
        build_run_record_cb=build_run_record_cb,
        dataset_labels=dataset_labels,
        baseline=baseline,
        workspace_root=workspace_root,
    )
    for name, label in dataset_labels.items():
        abbreviations.setdefault(name, label)
    return abbreviations, legend, dataset_labels
