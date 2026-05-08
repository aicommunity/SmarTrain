from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from smartrain.canonical.policy import emit_legacy_read_deprecation_warnings
from smartrain.canonical.refs import canonical_target_from_model_dir
from smartrain.canonical.schema import wrap_inference_report_v2
from smartrain.core.runtime.path_portable import relativize_if_under
from smartrain.core.runtime.run_artifacts import is_internal_conversion_artifact
from smartrain.core.runtime.run_discovery import find_run_directories
from smartrain.core.runtime.ultralytics_ephemeral import ultralytics_sidecar_dir
from smartrain.core.runtime.workspace_paths import WorkspaceLayout
from smartrain.core.training.train_profile import task_to_metadata_task_type
from smartrain.workflows.datasets.dataset_access import resolve_dataset_root_for_entry
from smartrain.workflows.datasets.dataset_roi_yolo import (
    _clamp_crop,
    _full_image_crop,
    _select_roi_boxes,
)
from smartrain.workflows.datasets.datasets_json_former import find_yaml_file
from smartrain.workflows.models.model_context import infer_img_size_from_model_context

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MANIFEST_NAME = "model_manifest.json"
SUPPORTED_INFERENCE_EXTS = {".pt", ".onnx", ".engine", ".trt"}
DATA_MODES = ("folder", "dataset-split")
ROI_POLICIES = ("largest", "highest_conf")
ON_EMPTY_MODES = ("skip", "full", "fail")


def sanitize_segment(value: str) -> str:
    out = re.sub(r"[^\w.\-+]+", "_", str(value), flags=re.UNICODE).strip("._")
    return out[:120] if out else "source"


def _parse_roi_class_ids(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    out: list[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out or None


def _resolve_run_ref(layout: WorkspaceLayout, ref: str) -> Path:
    s = str(ref).strip()
    if not s:
        raise ValueError("empty run reference")
    if s.isdigit():
        runs = find_run_directories(layout.runs)
        idx = int(s)
        if idx < 1 or idx > len(runs):
            raise ValueError(f"run index {idx} is out of range 1..{len(runs)}")
        return Path(runs[idx - 1]).resolve()
    return Path(s).expanduser().resolve()


def resolve_model(args: argparse.Namespace, layout: WorkspaceLayout) -> tuple[Path, str, str]:
    def _resolve_and_validate_canonical_weights(model: Any) -> Path:
        p = Path(str(model.weights_path)).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Canonical weights not found: {p}")
        if p.suffix.lower() not in SUPPORTED_INFERENCE_EXTS:
            raise FileNotFoundError(f"Unsupported canonical weights format: {p.suffix}")
        return p

    emit_legacy_read_deprecation_warnings()
    if args.model_name:
        from smartrain.orchestrators.canonical_gateway import load_target, resolve_task_context

        mdir = (Path(layout.models) / str(args.model_name).strip()).resolve()
        _ = resolve_task_context(str(mdir), source_kind="model")
        payload = load_target(str(mdir), source_kind="model")
        if not payload.models:
            raise FileNotFoundError(f"Canonical model payload has no models for: {mdir}")
        model = payload.models[0]
        p = _resolve_and_validate_canonical_weights(model)
        return p, str(model.model_id or mdir.name), "models"
    if args.run:
        run_dir = _resolve_run_ref(layout, str(args.run))
        from smartrain.orchestrators.canonical_gateway import load_target, resolve_task_context

        ctx = resolve_task_context(str(run_dir), source_kind="run")
        payload = load_target(str(run_dir), source_kind="run")
        if not payload.models:
            raise FileNotFoundError(f"Canonical run payload has no models for: {run_dir}")
        model = payload.models[0]
        p = _resolve_and_validate_canonical_weights(model)
        source_id = str(ctx.run_id or (payload.runs[0].run_id if payload.runs else run_dir.name))
        return p, source_id, "runs"
    if args.weights:
        w = Path(str(args.weights)).expanduser()
        if not w.is_absolute():
            w = (Path(layout.root) / w).resolve()
        if not w.is_file():
            raise FileNotFoundError(f"Weights not found: {w}")
        if w.suffix.lower() not in SUPPORTED_INFERENCE_EXTS:
            raise FileNotFoundError(f"Unsupported weights format: {w.suffix}")
        return w.resolve(), w.stem, "weights"
    raise ValueError("Specify one of --model-name, --run or --weights.")


def infer_img_size_from_model_context_safe(model_path: Path) -> int | None:
    return infer_img_size_from_model_context(model_path)


def _load_catalog(layout: WorkspaceLayout) -> dict[str, Any]:
    path = Path(layout.work_datasets_info_path())
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def collect_folder_images(source_dir: str, limit: int) -> list[str]:
    d = os.path.abspath(os.path.expanduser(source_dir))
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Source directory not found: {d}")
    images = sorted(
        p
        for p in glob(os.path.join(d, "**", "*"), recursive=True)
        if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)
    )
    if limit and limit > 0:
        return images[:limit]
    return images


def collect_split_images_for_dataset(
    layout: WorkspaceLayout, dataset: str, split: str, limit: int
) -> tuple[list[str], str]:
    catalog = _load_catalog(layout)
    if dataset not in catalog:
        raise KeyError(f"Dataset {dataset!r} not found in {layout.work_datasets_info_path()}")
    entry = catalog[dataset]
    dataset_root = resolve_dataset_root_for_entry(
        dataset_name=dataset,
        info=entry,
        workspace_root=layout.root,
        source_catalog_dir=layout.datasets,
        legacy_source_parent=layout.datasets,
    )
    ypath = find_yaml_file(dataset_root)
    if not ypath:
        raise FileNotFoundError(f"data.yaml not found for dataset: {dataset}")
    with open(ypath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML for dataset: {dataset}")
    split_rel = data.get(split)
    if not isinstance(split_rel, str) or not split_rel.strip():
        raise ValueError(f"data.yaml has no path for split={split!r}")
    split_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(ypath)), split_rel))
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"Split directory not found: {split_path}")
    images = sorted(
        p
        for p in glob(os.path.join(split_path, "**", "*"), recursive=True)
        if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS)
    )
    if limit and limit > 0:
        images = images[:limit]
    return images, split_path


def predict_roi_crop(roi_model: Any, image_path: str, args: argparse.Namespace) -> tuple[int, int, int, int]:
    with Image.open(image_path) as im:
        iw, ih = im.size
    proj = getattr(args, "_ultralytics_roi_project", None) or ultralytics_sidecar_dir(
        tempfile.gettempdir(), "smartrain_roi_infer"
    )
    roi_pred = roi_model.predict(
        source=image_path,
        conf=float(args.roi_conf),
        verbose=False,
        save=False,
        project=proj,
        name="roi-crop",
        exist_ok=True,
    )
    if not roi_pred:
        if args.roi_on_empty == "fail":
            raise RuntimeError(f"No ROI detections for: {image_path}")
        if args.roi_on_empty == "skip":
            return -1, -1, -1, -1
        return tuple(int(v) for v in _full_image_crop(iw, ih))
    r = roi_pred[0]
    if r.boxes is None or len(r.boxes) == 0:
        if args.roi_on_empty == "fail":
            raise RuntimeError(f"No ROI detections for: {image_path}")
        if args.roi_on_empty == "skip":
            return -1, -1, -1, -1
        return tuple(int(v) for v in _full_image_crop(iw, ih))
    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    class_ids = _parse_roi_class_ids(args.roi_class_ids)
    roi_list = _select_roi_boxes(xyxy, cls, confs, class_ids, args.roi_policy, iw, ih)
    if not roi_list:
        if args.roi_on_empty == "fail":
            raise RuntimeError(f"No ROI detections for: {image_path}")
        if args.roi_on_empty == "skip":
            return -1, -1, -1, -1
        return tuple(int(v) for v in _full_image_crop(iw, ih))
    x1, y1, x2, y2 = roi_list[0]
    return _clamp_crop(x1, y1, x2, y2, int(args.roi_pad_px), iw, ih)


def resolve_output_root(layout: WorkspaceLayout, model_name: str, source_short: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{ts}-{sanitize_segment(source_short)}"
    out = os.path.join(layout.root, "inference", sanitize_segment(model_name), run_name)
    os.makedirs(out, exist_ok=True)
    return out


def source_descriptor(
    args: argparse.Namespace, source_abs: str, source_short: str, layout: WorkspaceLayout
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "mode": args.data_mode,
        "name": source_short,
        "path_absolute": source_abs,
        "path_relative": relativize_if_under(layout.root, source_abs) or source_abs,
    }
    if args.data_mode == "dataset-split":
        source["dataset"] = args.dataset
        source["split"] = args.split
    return source


def build_report(
    *,
    args: argparse.Namespace,
    layout: WorkspaceLayout,
    model_source: str,
    model_name: str,
    model_path: Path,
    source_abs: str,
    source_short: str,
    out_root: str,
    report_path: str,
    images_input_count: int,
    image_rows: list[dict[str, Any]],
    skipped: int,
    performance: dict[str, Any] | None = None,
    environment_artifact_path: str | None = None,
) -> dict[str, Any]:
    task_type = task_to_metadata_task_type(getattr(args, "task", None))
    detections_total = sum(len(x.get("detections", [])) for x in image_rows)
    task_outputs_total = 0
    for row in image_rows:
        task_outputs = row.get("task_outputs")
        if not isinstance(task_outputs, dict):
            continue
        if task_type == "classification":
            cls = task_outputs.get("classification")
            if isinstance(cls, dict) and cls:
                task_outputs_total += 1
            continue
        if task_type == "segmentation":
            segs = task_outputs.get("segments")
            if isinstance(segs, list):
                task_outputs_total += len(segs)
            continue
        dets = task_outputs.get("detections")
        if isinstance(dets, list):
            task_outputs_total += len(dets)
    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "task_type": task_type,
        "workspace": {
            "root_absolute": layout.root,
            "root_relative": relativize_if_under(layout.root, layout.root) or ".",
        },
        "model": {
            "source": model_source,
            "name": model_name,
            "provider": {
                "type": "external" if str(getattr(args, "external_provider", "") or "").strip() else "builtin",
                "id": str(getattr(args, "external_provider", "") or "").strip() or "ultralytics",
            },
            "weights_absolute": str(model_path),
            "weights_relative": relativize_if_under(layout.root, str(model_path)) or str(model_path),
        },
        "parameters": {
            "conf": args.conf,
            "img_size": int(args.img_size),
            "device": args.device,
            "half": bool(args.half),
            "limit": int(args.limit),
            "data_mode": args.data_mode,
            "roi_pre_detect": bool(args.roi_pre_detect),
            "roi_weights": args.roi_weights,
            "roi_conf": args.roi_conf,
            "roi_policy": args.roi_policy,
            "roi_pad_px": args.roi_pad_px,
            "roi_on_empty": args.roi_on_empty,
            "roi_class_ids": _parse_roi_class_ids(args.roi_class_ids),
        },
        "source": source_descriptor(args, source_abs, source_short, layout),
        "output": {
            "dir_absolute": out_root,
            "dir_relative": relativize_if_under(layout.root, out_root) or out_root,
            "json_absolute": report_path,
            "json_relative": relativize_if_under(layout.root, report_path) or report_path,
        },
        "summary": {
            "images_input": images_input_count,
            "images_processed": len(image_rows),
            "images_skipped": skipped,
            "detections_total": detections_total,
            "task_outputs_total": task_outputs_total,
        },
        "performance": performance if isinstance(performance, dict) else None,
        "artifacts": {
            "environment_profile": {
                "path_absolute": environment_artifact_path,
                "path_relative": (
                    relativize_if_under(layout.root, environment_artifact_path) if environment_artifact_path else None
                ),
            }
        },
        "images": image_rows,
    }


def write_report(path: str, report: dict[str, Any]) -> None:
    report = wrap_inference_report_v2(report)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def resolve_external_source(args: argparse.Namespace, layout: WorkspaceLayout) -> str:
    if args.data_mode == "folder":
        return os.path.abspath(os.path.expanduser(str(args.source_dir)))
    _, split_dir = collect_split_images_for_dataset(
        layout,
        str(args.dataset),
        str(args.split),
        int(args.limit),
    )
    return split_dir

