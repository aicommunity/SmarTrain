from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from smartrain.run_model_contract.refs import unified_target_from_model_dir
from smartrain.run_model_contract.schema import wrap_inference_report_v2
from smartrain.core.runtime.path_portable import relativize_if_under
from smartrain.core.runtime.run_artifacts import is_internal_conversion_artifact
from smartrain.core.runtime.run_discovery import find_run_directories
from smartrain.core.runtime.ultralytics_ephemeral import ultralytics_sidecar_dir
from smartrain.core.runtime.workspace_paths import (
    WorkspaceLayout,
    extract_dataset_archive_to_cache,
    is_dataset_archive_path,
    resolve_dataset_root,
)
from smartrain.core.training.train_profile import task_to_metadata_task_type
from smartrain.core.workflow_adapters.inference_runtime_api import (
    find_yaml_file,
    infer_img_size_from_model_context,
    infer_img_size_with_source,
    resolve_inference_imgsz,
    resolve_dataset_root_for_entry,
    _clamp_crop,
    _full_image_crop,
    _select_roi_boxes,
)
from smartrain.services.datasets.dataset_roi_yolo import ON_EMPTY_MODES, ROI_POLICIES

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MANIFEST_NAME = "model_manifest.json"
SUPPORTED_INFERENCE_EXTS = {".pt", ".onnx", ".engine", ".trt"}
DATA_MODES = ("folder", "dataset-split")


@dataclass(frozen=True)
class ResolvedInferenceSource:
    working_path: str
    display_name: str
    source_archive: str | None = None


def _archive_display_name(path: str | os.PathLike[str]) -> str:
    name = os.path.basename(str(path))
    for suffix in (".tar.gz", ".tgz", ".tar", ".zip"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def resolve_inference_folder_source(source_path: str, workspace_root: str) -> ResolvedInferenceSource:
    abs_path = os.path.abspath(os.path.expanduser(source_path))
    if os.path.isdir(abs_path):
        display = os.path.basename(abs_path.rstrip(os.sep)) or "folder"
        return ResolvedInferenceSource(working_path=abs_path, display_name=display)
    if os.path.isfile(abs_path) and is_dataset_archive_path(abs_path):
        print(f"[INFO] Extracting archive to cache: {abs_path}")
        extracted = extract_dataset_archive_to_cache(workspace_root, abs_path)
        return ResolvedInferenceSource(
            working_path=extracted,
            display_name=_archive_display_name(abs_path),
            source_archive=abs_path,
        )
    raise FileNotFoundError(f"Source directory or archive not found: {abs_path}")


def _dataset_catalog_archive_path(layout: WorkspaceLayout, dataset: str, entry: dict[str, Any]) -> str | None:
    try:
        raw_root = resolve_dataset_root(layout.root, dataset, entry, layout.datasets)
    except Exception:
        return None
    abs_root = os.path.abspath(os.path.expanduser(raw_root))
    if os.path.isfile(abs_root) and is_dataset_archive_path(abs_root):
        return abs_root
    return None


def resolve_inference_source(
    args: argparse.Namespace, layout: WorkspaceLayout
) -> tuple[list[str], ResolvedInferenceSource]:
    limit = int(args.limit)
    if args.data_mode == "folder":
        source_path = str(getattr(args, "source", None) or args.source_dir or "")
        resolved = resolve_inference_folder_source(source_path, layout.root)
        images = collect_folder_images(resolved.working_path, limit)
        return images, resolved

    images, split_dir = collect_split_images_for_dataset(
        layout,
        str(args.dataset),
        str(args.split),
        limit,
    )
    catalog = load_catalog(layout)
    entry = catalog.get(str(args.dataset), {})
    source_archive = _dataset_catalog_archive_path(layout, str(args.dataset), entry) if isinstance(entry, dict) else None
    return images, ResolvedInferenceSource(
        working_path=split_dir,
        display_name=f"{args.dataset}-{args.split}",
        source_archive=source_archive,
    )


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


def resolve_model_from_name(layout: WorkspaceLayout, name: str) -> tuple[Path, str]:
    """Resolve promoted model directory name into a resolved weights path."""
    models_root = Path(layout.models).resolve()
    candidate_rel = Path(name)
    if candidate_rel.suffix.lower() in SUPPORTED_INFERENCE_EXTS and not candidate_rel.is_absolute():
        file_path = (models_root / candidate_rel).resolve()
        if file_path.is_file():
            parts = candidate_rel.as_posix().split("/")
            model_dir_name = parts[0] if parts else file_path.stem
            return file_path, model_dir_name

    mdir = (Path(layout.models) / name).resolve()
    if not mdir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {mdir}")

    manifest = mdir / MANIFEST_NAME
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        wf = payload.get("weights_file")
        if isinstance(wf, str) and wf.strip():
            p = (mdir / wf).resolve()
            if p.is_file():
                return p, name

    canonical = unified_target_from_model_dir(mdir)
    return canonical.model_path.resolve(), name


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

    if args.model_name:
        name = str(args.model_name).strip()
        candidate = Path(name)
        models_root = Path(layout.models).resolve()
        if candidate.suffix.lower() in SUPPORTED_INFERENCE_EXTS:
            file_path = candidate.resolve() if candidate.is_absolute() else (models_root / candidate).resolve()
            if file_path.is_file():
                if candidate.is_absolute():
                    display_name = file_path.parent.name
                else:
                    parts = candidate.as_posix().split("/")
                    display_name = parts[0] if parts else file_path.stem
                return file_path, display_name, "models"

        from smartrain.run_model_contract.gateway import load_target, resolve_task_context

        mdir = (models_root / name).resolve()
        _ = resolve_task_context(str(mdir), source_kind="model")
        payload = load_target(str(mdir), source_kind="model")
        if not payload.models:
            raise FileNotFoundError(f"Canonical model payload has no models for: {mdir}")
        model = payload.models[0]
        p = _resolve_and_validate_canonical_weights(model)
        return p, str(model.model_id or mdir.name), "models"
    if args.run:
        run_dir = _resolve_run_ref(layout, str(args.run))
        from smartrain.run_model_contract.gateway import load_target, resolve_task_context

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


def infer_img_size_with_source_safe(model_path: Path) -> tuple[int | None, str]:
    return infer_img_size_with_source(model_path)


def apply_inference_imgsz_from_model(model_path: Path, args: argparse.Namespace) -> tuple[int, str]:
    """Resolve args.img_size from model context unless explicitly set on CLI."""
    from smartrain.workflows.models.model_context import FALLBACK_IMGSZ_SOURCE

    explicit = getattr(args, "img_size", None)
    imgsz, source = resolve_inference_imgsz(model_path, explicit=explicit)
    args.img_size = int(imgsz)
    args.img_size_source = source
    if explicit is None:
        if source == FALLBACK_IMGSZ_SOURCE:
            print(
                f"[WARN] Model input size not found. Using fallback {imgsz}. "
                "Set --img-size to override."
            )
        else:
            print(f"[INFO] Resolved input size: {imgsz} (source: {source})")
    return int(imgsz), source


def discover_model_entries(layout: WorkspaceLayout) -> list[tuple[str, str, str]]:
    """Return tuples: (display_label, model_name_arg_value, model_dir_name)."""
    root = Path(layout.models)
    if not root.is_dir():
        return []
    out: list[tuple[str, str, str]] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(
            p
            for p in d.rglob("*")
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_INFERENCE_EXTS
            and not is_internal_conversion_artifact(p)
        )
        if not files:
            out.append((f"{d.name}/(no model files)", d.name, d.name))
            continue
        for fp in files:
            rel = fp.relative_to(root).as_posix()
            out.append((rel, rel, d.name))
    return out


def load_catalog(layout: WorkspaceLayout) -> dict[str, Any]:
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
    catalog = load_catalog(layout)
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
    args: argparse.Namespace,
    source_abs: str,
    source_short: str,
    layout: WorkspaceLayout,
    *,
    source_archive: str | None = None,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "mode": args.data_mode,
        "name": source_short,
        "path_absolute": source_abs,
        "path_relative": relativize_if_under(layout.root, source_abs) or source_abs,
    }
    if source_archive:
        source["source_archive_absolute"] = source_archive
        source["source_archive_relative"] = relativize_if_under(layout.root, source_archive) or source_archive
    if args.data_mode == "dataset-split":
        source["dataset"] = args.dataset
        source["split"] = args.split
    return source


def source_descriptor_from_resolved(
    args: argparse.Namespace, resolved: ResolvedInferenceSource, layout: WorkspaceLayout
) -> dict[str, Any]:
    return source_descriptor(
        args,
        resolved.working_path,
        resolved.display_name,
        layout,
        source_archive=resolved.source_archive,
    )


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
    source_archive: str | None = None,
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
            "img_size_source": str(getattr(args, "img_size_source", "") or ""),
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
            "export_dataset": bool(getattr(args, "export_dataset", True)),
            "export_label_conf_min": float(getattr(args, "export_label_conf_min", 0.25)),
            "export_label_conf_max": float(getattr(args, "export_label_conf_max", 1.0)),
            "export_visualize": (
                bool(getattr(args, "export_visualize"))
                if getattr(args, "export_visualize", None) is not None
                else bool(getattr(args, "export_dataset", True))
            ),
        },
        "source": source_descriptor(
            args,
            source_abs,
            source_short,
            layout,
            source_archive=source_archive,
        ),
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
    _, resolved = resolve_inference_source(args, layout)
    return resolved.working_path

