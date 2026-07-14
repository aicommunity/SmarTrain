"""Post-inference export: YOLO autolabel dataset and prediction overlays."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from smartrain.core.runtime.path_portable import relativize_if_under
from smartrain.core.runtime.workspace_paths import WorkspaceLayout
from smartrain.core.training.train_profile import task_to_metadata_task_type
from smartrain.services.datasets.yolo_labels import task_outputs_to_yolo_labels, write_yolo_labels
from smartrain.services.inference_runtime_helpers import sanitize_segment, write_report
from smartrain.services.visualization.color_registry import LabelColorRegistry
from smartrain.services.visualization.rendering import render_pred_overlay, save_rendered_image


MANIFEST_FILENAME = "autolabel_manifest.json"
SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class ExportOptions:
    export_dataset: bool
    export_visualize: bool
    label_conf_min: float
    label_conf_max: float
    export_split_dirs: bool = True
    export_files_per_dir: int = 500
    export_class_ids: frozenset[int] | None = None


@dataclass(frozen=True)
class ExportSummary:
    dataset_dir: str | None
    manifest_path: str | None
    images_exported: int
    labels_total: int
    images_skipped_empty: int
    overlay_paths: tuple[str, ...]
    overlay_dir: str | None
    parts_count: int = 0
    files_per_dir: int | None = None
    layout: str = "flat"
    images_skipped_no_class: int = 0


def resolve_export_options(args: argparse.Namespace) -> ExportOptions:
    export_dataset = bool(getattr(args, "export_dataset", True))
    raw_visualize = getattr(args, "export_visualize", None)
    if raw_visualize is None:
        export_visualize = export_dataset
    else:
        export_visualize = bool(raw_visualize)
    raw_class_ids = getattr(args, "export_class_ids", None)
    export_class_ids: frozenset[int] | None = None
    if raw_class_ids:
        export_class_ids = frozenset(int(x) for x in raw_class_ids)
    return ExportOptions(
        export_dataset=export_dataset,
        export_visualize=export_visualize,
        label_conf_min=float(getattr(args, "export_label_conf_min", 0.25)),
        label_conf_max=float(getattr(args, "export_label_conf_max", 1.0)),
        export_split_dirs=bool(getattr(args, "export_split_dirs", True)),
        export_files_per_dir=int(getattr(args, "export_files_per_dir", 500)),
        export_class_ids=export_class_ids,
    )


def validate_export_options(options: ExportOptions, *, parser: argparse.ArgumentParser | None = None) -> None:
    lo = float(options.label_conf_min)
    hi = float(options.label_conf_max)
    if lo < 0.0 or hi > 1.0 or lo > hi:
        msg = (
            "Invalid export label confidence range: "
            f"--export-label-conf-min={lo} and --export-label-conf-max={hi} "
            "(expected 0 <= min <= max <= 1)."
        )
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)
    if options.export_split_dirs and int(options.export_files_per_dir) < 1:
        msg = (
            f"Invalid --export-files-per-dir={options.export_files_per_dir} "
            "(expected >= 1 when --export-split-dirs is on)."
        )
        if parser is not None:
            parser.error(msg)
        raise ValueError(msg)


def resolve_autolabel_dataset_dir(out_root: str | Path, source_short: str) -> Path:
    base = sanitize_segment(str(source_short))
    return Path(out_root) / f"{base}_autolabeled"


def part_dirname(part_index: int) -> str:
    return f"part_{int(part_index):03d}"


def _image_size(row: dict[str, Any]) -> tuple[int, int]:
    size = row.get("image_size") if isinstance(row.get("image_size"), dict) else {}
    try:
        w = int(size.get("width", 0))
        h = int(size.get("height", 0))
    except Exception:
        w, h = 0, 0
    if w > 0 and h > 0:
        return w, h
    src = str(row.get("image_path_absolute") or "")
    if src and os.path.isfile(src):
        with Image.open(src) as im:
            return im.size
    return 0, 0


def _confidence_value(item: dict[str, Any]) -> float | None:
    if "confidence" not in item:
        return None
    try:
        return float(item["confidence"])
    except Exception:
        return None


def _passes_conf_filter(item: dict[str, Any], conf_min: float, conf_max: float) -> bool:
    conf = _confidence_value(item)
    if conf is None:
        return True
    return float(conf_min) <= float(conf) <= float(conf_max)


def filter_task_outputs(row: dict[str, Any], task_type: str, conf_min: float, conf_max: float) -> list[dict[str, Any]]:
    resolved = task_to_metadata_task_type(row.get("task_type", task_type))
    task_outputs = row.get("task_outputs") if isinstance(row.get("task_outputs"), dict) else {}
    if resolved == "segmentation":
        raw = task_outputs.get("segments")
        items = raw if isinstance(raw, list) else []
    elif resolved == "classification":
        return []
    else:
        raw = task_outputs.get("detections")
        if isinstance(raw, list) and raw:
            items = raw
        else:
            legacy = row.get("detections")
            items = legacy if isinstance(legacy, list) else []
    return [x for x in items if isinstance(x, dict) and _passes_conf_filter(x, conf_min, conf_max)]


def _item_class_index(item: dict[str, Any]) -> int | None:
    for key in ("class_index", "class_id"):
        if key not in item:
            continue
        try:
            return int(item[key])
        except Exception:
            continue
    return None


def filter_task_outputs_by_class(items: list[dict[str, Any]], class_ids: set[int] | frozenset[int] | None) -> list[dict[str, Any]]:
    if not class_ids:
        return items
    allowed = set(class_ids)
    out: list[dict[str, Any]] = []
    for item in items:
        cls_id = _item_class_index(item)
        if cls_id is not None and cls_id in allowed:
            out.append(item)
    return out


def _classification_top1_index(row: dict[str, Any]) -> int | None:
    task_outputs = row.get("task_outputs") if isinstance(row.get("task_outputs"), dict) else {}
    classification = task_outputs.get("classification")
    if not isinstance(classification, dict):
        return None
    top1 = classification.get("top1")
    if not isinstance(top1, dict):
        return None
    try:
        return int(top1.get("class_index"))
    except Exception:
        return None


def filter_export_task_outputs(
    row: dict[str, Any],
    task_type: str,
    *,
    conf_min: float,
    conf_max: float,
    class_ids: set[int] | frozenset[int] | None,
) -> list[dict[str, Any]]:
    resolved = task_to_metadata_task_type(row.get("task_type", task_type))
    if resolved == "classification":
        if not class_ids:
            return []
        top1_idx = _classification_top1_index(row)
        if top1_idx is None or top1_idx not in set(class_ids):
            return []
        return [{"class_index": top1_idx}]
    filtered = filter_task_outputs(row, task_type, conf_min, conf_max)
    return filter_task_outputs_by_class(filtered, class_ids)


def _apply_filtered_outputs_to_row(row: dict[str, Any], task_type: str, filtered: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = task_to_metadata_task_type(row.get("task_type", task_type))
    out = dict(row)
    task_outputs = dict(out.get("task_outputs") if isinstance(out.get("task_outputs"), dict) else {})
    if resolved == "segmentation":
        task_outputs["segments"] = filtered
        out["detections"] = filtered
    elif resolved == "classification":
        cls_obj = dict(task_outputs.get("classification") if isinstance(task_outputs.get("classification"), dict) else {})
        top1 = cls_obj.get("top1")
        if isinstance(top1, dict) and filtered:
            task_outputs["classification"] = {"top1": top1, "top_k": cls_obj.get("top_k", [])}
        else:
            task_outputs["classification"] = {}
        out["detections"] = []
    else:
        task_outputs["detections"] = filtered
        out["detections"] = filtered
    out["task_outputs"] = task_outputs
    return out


def _recompute_report_summary(report: dict[str, Any]) -> None:
    task_type = task_to_metadata_task_type(report.get("task_type"))
    images = report.get("images") if isinstance(report.get("images"), list) else []
    detections_total = 0
    task_outputs_total = 0
    for row in images:
        if not isinstance(row, dict):
            continue
        detections_total += len(row.get("detections") if isinstance(row.get("detections"), list) else [])
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
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    summary["images_processed"] = len([r for r in images if isinstance(r, dict)])
    summary["detections_total"] = detections_total
    summary["task_outputs_total"] = task_outputs_total
    report["summary"] = summary


def filter_inference_report_by_classes(
    report: dict[str, Any],
    *,
    class_ids: set[int] | frozenset[int] | None,
    conf_min: float,
    conf_max: float,
) -> tuple[dict[str, Any], int]:
    """Drop image rows without selected classes; trim outputs to selected classes only."""
    if not class_ids:
        return report, 0
    task_type = task_to_metadata_task_type(report.get("task_type"))
    images = report.get("images") if isinstance(report.get("images"), list) else []
    kept: list[dict[str, Any]] = []
    skipped_no_class = 0
    for row in images:
        if not isinstance(row, dict):
            continue
        filtered = filter_export_task_outputs(
            row,
            task_type,
            conf_min=conf_min,
            conf_max=conf_max,
            class_ids=class_ids,
        )
        if not filtered:
            skipped_no_class += 1
            continue
        kept.append(_apply_filtered_outputs_to_row(row, task_type, filtered))
    out = dict(report)
    out["images"] = kept
    summary = dict(out.get("summary") if isinstance(out.get("summary"), dict) else {})
    summary["images_skipped_no_class"] = int(summary.get("images_skipped_no_class", 0)) + skipped_no_class
    out["summary"] = summary
    _recompute_report_summary(out)
    return out, skipped_no_class


def _allocate_unique_stem(base_stem: str, used: set[str]) -> str:
    if base_stem not in used:
        used.add(base_stem)
        return base_stem
    n = 2
    while True:
        candidate = f"{base_stem}__{n}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        n += 1


def _class_names_from_outputs(items: list[dict[str, Any]]) -> dict[int, str]:
    names: dict[int, str] = {}
    for item in items:
        try:
            cls_id = int(item.get("class_index", item.get("class_id", -1)))
        except Exception:
            continue
        if cls_id < 0:
            continue
        cls_name = str(item.get("class_name") or cls_id)
        names.setdefault(cls_id, cls_name)
    return names


def _write_data_yaml(dataset_dir: Path, class_names: dict[int, str]) -> None:
    ordered = [name for _idx, name in sorted(class_names.items(), key=lambda kv: kv[0])]
    if not ordered:
        ordered = ["obj"]
    content = (
        "train: images\n"
        "val: images\n"
        "test: images\n\n"
        f"nc: {len(ordered)}\n"
        f"names: {ordered}\n"
    )
    (dataset_dir / "data.yaml").write_text(content, encoding="utf-8")


def _model_block(report: dict[str, Any]) -> dict[str, Any]:
    model = report.get("model") if isinstance(report.get("model"), dict) else {}
    provider = model.get("provider") if isinstance(model.get("provider"), dict) else {}
    weights_abs = model.get("weights_absolute") or model.get("weights_value")
    weights_rel = model.get("weights_relative")
    return {
        "source": model.get("source"),
        "name": model.get("name"),
        "weights_absolute": weights_abs,
        "weights_relative": weights_rel,
        "provider": provider,
    }


def _inference_parameters_block(report: dict[str, Any]) -> dict[str, Any]:
    params = report.get("parameters") if isinstance(report.get("parameters"), dict) else {}
    return {
        "conf": params.get("conf"),
        "img_size": params.get("img_size"),
        "device": params.get("device"),
        "half": params.get("half"),
        "batch_size": params.get("batch_size"),
        "task_type": report.get("task_type"),
        "data_mode": params.get("data_mode"),
        "limit": params.get("limit"),
        "roi_pre_detect": params.get("roi_pre_detect"),
    }


def _det_for_render(det: dict[str, Any], img_w: int, img_h: int) -> dict[str, Any]:
    out = dict(det)
    bbox = det.get("bbox_original_xyxy") or det.get("bbox_roi_xyxy") or det.get("bbox_xyxy")
    if isinstance(bbox, list) and len(bbox) >= 4:
        out["bbox_xyxy"] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    poly = det.get("polygon_original_xy") or det.get("polygon_roi_xy") or det.get("polygon_xy")
    if isinstance(poly, list) and poly and img_w > 0 and img_h > 0:
        pts: list[list[float]] = []
        for point in poly:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x = float(point[0])
                y = float(point[1])
            except Exception:
                continue
            if max(abs(x), abs(y)) <= 1.0 + 1e-6:
                pts.append([x, y])
            else:
                pts.append([x / float(img_w), y / float(img_h)])
        if pts:
            out["polygon_xy"] = pts
    return out


def _empty_summary(*, images_skipped_empty: int = 0, images_skipped_no_class: int = 0) -> ExportSummary:
    return ExportSummary(
        dataset_dir=None,
        manifest_path=None,
        images_exported=0,
        labels_total=0,
        images_skipped_empty=images_skipped_empty,
        overlay_paths=(),
        overlay_dir=None,
        parts_count=0,
        files_per_dir=None,
        layout="flat",
        images_skipped_no_class=images_skipped_no_class,
    )


def _write_flat_dataset(
    *,
    dataset_dir: Path,
    pending_exports: list[tuple[str, str, str, list]],
    all_class_names: dict[int, str],
    file_mapping: list[dict[str, Any]],
    report: dict[str, Any],
    report_path: str | Path,
    out_root: str | Path,
    options: ExportOptions,
    images_input: int,
    images_exported: int,
    images_skipped_empty: int,
    labels_total: int,
) -> Path:
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for src, export_image_name, export_label_name, yolo_labels in pending_exports:
        shutil.copy2(src, images_dir / export_image_name)
        write_yolo_labels(str(labels_dir / export_label_name), yolo_labels)

    _write_data_yaml(dataset_dir, all_class_names)

    manifest_path = dataset_dir / MANIFEST_FILENAME
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labeling_type": "autolabel",
        "source": report.get("source") if isinstance(report.get("source"), dict) else {},
        "model": _model_block(report),
        "inference_parameters": _inference_parameters_block(report),
        "export_parameters": {
            "export_label_conf_min": options.label_conf_min,
            "export_label_conf_max": options.label_conf_max,
            "layout": "flat",
            "export_split_dirs": False,
            "export_files_per_dir": options.export_files_per_dir,
            "export_class_ids": sorted(options.export_class_ids) if options.export_class_ids else None,
        },
        "provenance": {
            "inference_report_absolute": str(Path(report_path).resolve()),
            "inference_run_dir_absolute": str(Path(out_root).resolve()),
            "producer": "smartrain.inference",
        },
        "summary": {
            "images_input": images_input,
            "images_exported": images_exported,
            "images_skipped_empty": images_skipped_empty,
            "labels_total": labels_total,
        },
        "file_mapping": file_mapping,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _write_independent_parts(
    *,
    dataset_dir: Path,
    pending_exports: list[tuple[str, str, str, list, dict[int, str]]],
    report: dict[str, Any],
    report_path: str | Path,
    out_root: str | Path,
    options: ExportOptions,
    images_input: int,
    images_exported: int,
    images_skipped_empty: int,
    labels_total: int,
) -> tuple[Path, int]:
    files_per_dir = max(1, int(options.export_files_per_dir))
    parts_meta: list[dict[str, Any]] = []
    created_at = datetime.now(timezone.utc).isoformat()
    model_block = _model_block(report)
    infer_params = _inference_parameters_block(report)
    source_block = report.get("source") if isinstance(report.get("source"), dict) else {}
    provenance = {
        "inference_report_absolute": str(Path(report_path).resolve()),
        "inference_run_dir_absolute": str(Path(out_root).resolve()),
        "producer": "smartrain.inference",
    }

    part_count = int(math.ceil(len(pending_exports) / float(files_per_dir))) if pending_exports else 0
    for part_index in range(part_count):
        start = part_index * files_per_dir
        chunk = pending_exports[start : start + files_per_dir]
        part_id = part_dirname(part_index)
        part_dir = dataset_dir / part_id
        images_dir = part_dir / "images"
        labels_dir = part_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        part_class_names: dict[int, str] = {}
        part_mapping: list[dict[str, Any]] = []
        part_labels_total = 0
        for src, export_image_name, export_label_name, yolo_labels, class_names in chunk:
            shutil.copy2(src, images_dir / export_image_name)
            write_yolo_labels(str(labels_dir / export_label_name), yolo_labels)
            part_class_names.update(class_names)
            part_labels_total += len(yolo_labels)
            part_mapping.append(
                {
                    "source_path_absolute": os.path.abspath(src),
                    "export_stem": Path(export_image_name).stem,
                    "export_image": f"images/{export_image_name}",
                    "export_label": f"labels/{export_label_name}",
                }
            )

        _write_data_yaml(part_dir, part_class_names)
        part_manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": created_at,
            "labeling_type": "autolabel",
            "part_id": part_id,
            "part_index": part_index,
            "source": source_block,
            "model": model_block,
            "inference_parameters": infer_params,
            "export_parameters": {
                "export_label_conf_min": options.label_conf_min,
                "export_label_conf_max": options.label_conf_max,
                "layout": "flat",
                "export_split_dirs": True,
                "export_files_per_dir": files_per_dir,
                "part_id": part_id,
                "part_index": part_index,
                "export_class_ids": sorted(options.export_class_ids) if options.export_class_ids else None,
            },
            "provenance": provenance,
            "summary": {
                "images_exported": len(chunk),
                "labels_total": part_labels_total,
            },
            "file_mapping": part_mapping,
        }
        (part_dir / MANIFEST_FILENAME).write_text(
            json.dumps(part_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        parts_meta.append(
            {
                "id": part_id,
                "path": part_id,
                "images_exported": len(chunk),
                "labels_total": part_labels_total,
            }
        )

    root_manifest_path = dataset_dir / MANIFEST_FILENAME
    root_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "labeling_type": "autolabel",
        "source": source_block,
        "model": model_block,
        "inference_parameters": infer_params,
        "export_parameters": {
            "export_label_conf_min": options.label_conf_min,
            "export_label_conf_max": options.label_conf_max,
            "layout": "independent_parts",
            "export_split_dirs": True,
            "export_files_per_dir": files_per_dir,
            "parts": part_count,
            "export_class_ids": sorted(options.export_class_ids) if options.export_class_ids else None,
        },
        "provenance": provenance,
        "summary": {
            "images_input": images_input,
            "images_exported": images_exported,
            "images_skipped_empty": images_skipped_empty,
            "labels_total": labels_total,
            "parts": part_count,
        },
        "parts": parts_meta,
    }
    root_manifest_path.write_text(json.dumps(root_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return root_manifest_path, part_count


def export_yolo_dataset(
    report: dict[str, Any],
    *,
    out_root: str | Path,
    source_short: str,
    report_path: str | Path,
    options: ExportOptions,
    layout: WorkspaceLayout,
) -> tuple[ExportSummary, set[str]]:
    _ = layout
    task_type = task_to_metadata_task_type(report.get("task_type"))
    if task_type == "classification":
        print(
            "[WARN] Autolabel YOLO export is not supported for classification; skipping dataset export.",
            file=sys.stderr,
        )
        return _empty_summary(), set()

    dataset_dir = resolve_autolabel_dataset_dir(out_root, source_short)
    images = report.get("images") if isinstance(report.get("images"), list) else []
    used_stems: set[str] = set()
    exported_paths: set[str] = set()
    all_class_names: dict[int, str] = {}
    labels_total = 0
    images_exported = 0
    images_skipped_empty = 0
    images_skipped_no_class = 0
    # (src, image_name, label_name, yolo_labels, class_names)
    pending_exports: list[tuple[str, str, str, list, dict[int, str]]] = []

    for row in images:
        if not isinstance(row, dict):
            continue
        src = str(row.get("image_path_absolute") or "")
        if not src or not os.path.isfile(src):
            continue
        conf_filtered = filter_task_outputs(row, task_type, options.label_conf_min, options.label_conf_max)
        filtered = filter_export_task_outputs(
            row,
            task_type,
            conf_min=options.label_conf_min,
            conf_max=options.label_conf_max,
            class_ids=options.export_class_ids,
        )
        img_w, img_h = _image_size(row)
        yolo_labels = task_outputs_to_yolo_labels(task_type, filtered, img_w, img_h)
        if not yolo_labels:
            if conf_filtered and options.export_class_ids:
                images_skipped_no_class += 1
            else:
                images_skipped_empty += 1
            continue

        stem = Path(src).stem
        export_stem = _allocate_unique_stem(stem, used_stems)
        ext = Path(src).suffix.lower() or ".jpg"
        export_image_name = f"{export_stem}{ext}"
        export_label_name = f"{export_stem}.txt"
        class_names = _class_names_from_outputs(filtered)
        pending_exports.append((src, export_image_name, export_label_name, yolo_labels, class_names))
        labels_total += len(yolo_labels)
        images_exported += 1
        exported_paths.add(os.path.abspath(src))
        all_class_names.update(class_names)

    if images_exported == 0:
        print(
            "[INFO] Autolabel dataset export skipped: no images with labels after confidence filter.",
            file=sys.stderr,
        )
        return _empty_summary(
            images_skipped_empty=images_skipped_empty,
            images_skipped_no_class=images_skipped_no_class,
        ), set()

    images_input = len([r for r in images if isinstance(r, dict)])
    split_dirs = bool(options.export_split_dirs)

    if split_dirs:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        manifest_path, parts_count = _write_independent_parts(
            dataset_dir=dataset_dir,
            pending_exports=pending_exports,
            report=report,
            report_path=report_path,
            out_root=out_root,
            options=options,
            images_input=images_input,
            images_exported=images_exported,
            images_skipped_empty=images_skipped_empty,
            labels_total=labels_total,
        )
        print(
            f"[OK] Autolabel split into {parts_count} independent sub-dataset(s) "
            f"(files_per_dir={int(options.export_files_per_dir)}, exported={images_exported})",
            file=sys.stderr,
        )
        return (
            ExportSummary(
                dataset_dir=str(dataset_dir.resolve()),
                manifest_path=str(manifest_path.resolve()),
                images_exported=images_exported,
                labels_total=labels_total,
                images_skipped_empty=images_skipped_empty,
                overlay_paths=(),
                overlay_dir=None,
                parts_count=parts_count,
                files_per_dir=int(options.export_files_per_dir),
                layout="independent_parts",
                images_skipped_no_class=images_skipped_no_class,
            ),
            exported_paths,
        )

    flat_pending = [(s, i, l, y) for s, i, l, y, _c in pending_exports]
    flat_mapping = [
        {
            "source_path_absolute": os.path.abspath(src),
            "export_stem": Path(export_image_name).stem,
            "export_image": f"images/{export_image_name}",
            "export_label": f"labels/{export_label_name}",
        }
        for src, export_image_name, export_label_name, _yolo in flat_pending
    ]
    manifest_path = _write_flat_dataset(
        dataset_dir=dataset_dir,
        pending_exports=flat_pending,
        all_class_names=all_class_names,
        file_mapping=flat_mapping,
        report=report,
        report_path=report_path,
        out_root=out_root,
        options=options,
        images_input=images_input,
        images_exported=images_exported,
        images_skipped_empty=images_skipped_empty,
        labels_total=labels_total,
    )
    return (
        ExportSummary(
            dataset_dir=str(dataset_dir.resolve()),
            manifest_path=str(manifest_path.resolve()),
            images_exported=images_exported,
            labels_total=labels_total,
            images_skipped_empty=images_skipped_empty,
            overlay_paths=(),
            overlay_dir=None,
            parts_count=1,
            files_per_dir=None,
            layout="flat",
            images_skipped_no_class=images_skipped_no_class,
        ),
        exported_paths,
    )


def export_prediction_overlays(
    report: dict[str, Any],
    *,
    out_root: str | Path,
    layout: WorkspaceLayout,
    exported_only: set[str] | None,
    use_export_filter: bool,
    options: ExportOptions,
) -> tuple[list[str], str | None]:
    overlay_dir = Path(out_root) / "pred_overlays"
    task_type = task_to_metadata_task_type(report.get("task_type"))
    images = report.get("images") if isinstance(report.get("images"), list) else []
    color_registry = LabelColorRegistry(Path(layout.root))
    saved: list[str] = []
    class_names: dict[int, str] = {}
    split_dirs = bool(options.export_split_dirs)
    files_per_dir = max(1, int(options.export_files_per_dir))
    rendered_index = 0

    for row in images:
        if not isinstance(row, dict):
            continue
        src = str(row.get("image_path_absolute") or "")
        if not src or not os.path.isfile(src):
            continue
        src_abs = os.path.abspath(src)
        if exported_only is not None and src_abs not in exported_only:
            continue
        if use_export_filter:
            preds = filter_export_task_outputs(
                row,
                task_type,
                conf_min=options.label_conf_min,
                conf_max=options.label_conf_max,
                class_ids=options.export_class_ids,
            )
        else:
            task_outputs = row.get("task_outputs") if isinstance(row.get("task_outputs"), dict) else {}
            if task_type == "segmentation":
                raw = task_outputs.get("segments")
                preds = [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []
            elif task_type == "classification":
                preds = []
            else:
                raw = task_outputs.get("detections")
                if isinstance(raw, list) and raw:
                    preds = [x for x in raw if isinstance(x, dict)]
                else:
                    legacy = row.get("detections")
                    preds = [x for x in legacy if isinstance(x, dict)] if isinstance(legacy, list) else []
        class_names.update(_class_names_from_outputs(preds))
        img_w, img_h = _image_size(row)
        render_rows = [_det_for_render(det, img_w, img_h) for det in preds]
        with Image.open(src) as im:
            canvas = im.convert("RGBA")
            original_format = im.format
        palette = {name: color_registry.ensure(name) for name in class_names.values()}
        rendered = render_pred_overlay(canvas, render_rows, class_names, label_colors=palette)

        if split_dirs:
            target_dir = overlay_dir / part_dirname(rendered_index // files_per_dir)
        else:
            target_dir = overlay_dir
        if not saved:
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir.mkdir(parents=True, exist_ok=True)

        out_path = target_dir / Path(src).name
        if out_path.exists():
            out_path = target_dir / f"{Path(src).stem}__{len(saved) + 1}{Path(src).suffix.lower() or '.jpg'}"
        save_rendered_image(rendered, out_path, original_format=original_format)
        saved.append(str(out_path.resolve()))
        rendered_index += 1
    color_registry.save()
    if not saved:
        return [], None
    return saved, str(overlay_dir.resolve())


def patch_inference_report_artifacts(
    report_path: str | Path,
    *,
    layout: WorkspaceLayout,
    options: ExportOptions,
    summary: ExportSummary,
) -> None:
    path = Path(report_path)
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
    params.update(
        {
            "export_dataset": options.export_dataset,
            "export_label_conf_min": options.label_conf_min,
            "export_label_conf_max": options.label_conf_max,
            "export_visualize": options.export_visualize,
            "export_split_dirs": options.export_split_dirs,
            "export_files_per_dir": options.export_files_per_dir,
            "export_class_ids": sorted(options.export_class_ids) if options.export_class_ids else None,
        }
    )
    payload["parameters"] = params
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    if summary.dataset_dir:
        artifacts["autolabel_dataset"] = {
            "path_absolute": summary.dataset_dir,
            "path_relative": relativize_if_under(layout.root, summary.dataset_dir) or summary.dataset_dir,
            "manifest_absolute": summary.manifest_path,
            "manifest_relative": (
                relativize_if_under(layout.root, summary.manifest_path) if summary.manifest_path else None
            ),
            "images_exported": summary.images_exported,
            "labels_total": summary.labels_total,
            "layout": summary.layout,
            "parts_count": summary.parts_count,
            "files_per_dir": summary.files_per_dir,
        }
    if summary.overlay_dir:
        artifacts["pred_overlays"] = {
            "path_absolute": summary.overlay_dir,
            "path_relative": relativize_if_under(layout.root, summary.overlay_dir) or summary.overlay_dir,
            "images_rendered": len(summary.overlay_paths),
        }
    payload["artifacts"] = artifacts
    write_report(str(path), payload)


def run_inference_exports(
    *,
    report_path: str | Path,
    out_root: str | Path,
    source_short: str,
    args: argparse.Namespace,
    layout: WorkspaceLayout,
) -> ExportSummary:
    options = resolve_export_options(args)
    if not options.export_dataset and not options.export_visualize:
        return _empty_summary()

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    exported_paths: set[str] = set()
    summary = _empty_summary()

    if options.export_dataset:
        summary, exported_paths = export_yolo_dataset(
            report,
            out_root=out_root,
            source_short=source_short,
            report_path=report_path,
            options=options,
            layout=layout,
        )
        if summary.dataset_dir:
            parts_txt = (
                f", parts={summary.parts_count}, files_per_dir={summary.files_per_dir}"
                if summary.layout == "independent_parts"
                else ""
            )
            print(
                f"[OK] Autolabel dataset: {summary.dataset_dir} "
                f"(images={summary.images_exported}, labels={summary.labels_total}{parts_txt})"
            )

    if options.export_visualize:
        overlay_only = exported_paths if options.export_dataset else None
        use_filter = bool(options.export_dataset)
        overlay_paths, overlay_dir = export_prediction_overlays(
            report,
            out_root=out_root,
            layout=layout,
            exported_only=overlay_only,
            use_export_filter=use_filter,
            options=options,
        )
        summary = ExportSummary(
            dataset_dir=summary.dataset_dir,
            manifest_path=summary.manifest_path,
            images_exported=summary.images_exported,
            labels_total=summary.labels_total,
            images_skipped_empty=summary.images_skipped_empty,
            overlay_paths=tuple(overlay_paths),
            overlay_dir=overlay_dir,
            parts_count=summary.parts_count,
            files_per_dir=summary.files_per_dir,
            layout=summary.layout,
            images_skipped_no_class=summary.images_skipped_no_class,
        )
        if overlay_paths:
            print(f"[OK] Saved {len(overlay_paths)} prediction overlay image(s) to {overlay_dir}")

    patch_inference_report_artifacts(report_path, layout=layout, options=options, summary=summary)
    return summary
