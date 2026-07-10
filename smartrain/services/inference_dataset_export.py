"""Post-inference export: YOLO autolabel dataset and prediction overlays."""

from __future__ import annotations

import argparse
import json
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


@dataclass(frozen=True)
class ExportSummary:
    dataset_dir: str | None
    manifest_path: str | None
    images_exported: int
    labels_total: int
    images_skipped_empty: int
    overlay_paths: tuple[str, ...]
    overlay_dir: str | None


def resolve_export_options(args: argparse.Namespace) -> ExportOptions:
    export_dataset = bool(getattr(args, "export_dataset", True))
    raw_visualize = getattr(args, "export_visualize", None)
    if raw_visualize is None:
        export_visualize = export_dataset
    else:
        export_visualize = bool(raw_visualize)
    return ExportOptions(
        export_dataset=export_dataset,
        export_visualize=export_visualize,
        label_conf_min=float(getattr(args, "export_label_conf_min", 0.25)),
        label_conf_max=float(getattr(args, "export_label_conf_max", 1.0)),
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


def resolve_autolabel_dataset_dir(out_root: str | Path, source_short: str) -> Path:
    base = sanitize_segment(str(source_short))
    return Path(out_root) / f"{base}_autolabeled"


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


def export_yolo_dataset(
    report: dict[str, Any],
    *,
    out_root: str | Path,
    source_short: str,
    report_path: str | Path,
    options: ExportOptions,
    layout: WorkspaceLayout,
) -> tuple[ExportSummary, set[str]]:
    task_type = task_to_metadata_task_type(report.get("task_type"))
    if task_type == "classification":
        print(
            "[WARN] Autolabel YOLO export is not supported for classification; skipping dataset export.",
            file=sys.stderr,
        )
        return (
            ExportSummary(
                dataset_dir=None,
                manifest_path=None,
                images_exported=0,
                labels_total=0,
                images_skipped_empty=0,
                overlay_paths=(),
                overlay_dir=None,
            ),
            set(),
        )

    dataset_dir = resolve_autolabel_dataset_dir(out_root, source_short)
    images = report.get("images") if isinstance(report.get("images"), list) else []
    used_stems: set[str] = set()
    file_mapping: list[dict[str, Any]] = []
    exported_paths: set[str] = set()
    all_class_names: dict[int, str] = {}
    labels_total = 0
    images_exported = 0
    images_skipped_empty = 0
    pending_exports: list[tuple[str, str, str, list]] = []

    for row in images:
        if not isinstance(row, dict):
            continue
        src = str(row.get("image_path_absolute") or "")
        if not src or not os.path.isfile(src):
            continue
        filtered = filter_task_outputs(row, task_type, options.label_conf_min, options.label_conf_max)
        img_w, img_h = _image_size(row)
        yolo_labels = task_outputs_to_yolo_labels(task_type, filtered, img_w, img_h)
        if not yolo_labels:
            images_skipped_empty += 1
            continue

        stem = Path(src).stem
        export_stem = _allocate_unique_stem(stem, used_stems)
        ext = Path(src).suffix.lower() or ".jpg"
        export_image_name = f"{export_stem}{ext}"
        export_label_name = f"{export_stem}.txt"
        pending_exports.append((src, export_image_name, export_label_name, yolo_labels))
        labels_total += len(yolo_labels)
        images_exported += 1
        exported_paths.add(os.path.abspath(src))
        all_class_names.update(_class_names_from_outputs(filtered))
        file_mapping.append(
            {
                "source_path_absolute": os.path.abspath(src),
                "export_stem": export_stem,
                "export_image": f"images/{export_image_name}",
                "export_label": f"labels/{export_label_name}",
            }
        )

    if images_exported == 0:
        print(
            "[INFO] Autolabel dataset export skipped: no images with labels after confidence filter.",
            file=sys.stderr,
        )
        return (
            ExportSummary(
                dataset_dir=None,
                manifest_path=None,
                images_exported=0,
                labels_total=0,
                images_skipped_empty=images_skipped_empty,
                overlay_paths=(),
                overlay_dir=None,
            ),
            set(),
        )

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
        },
        "provenance": {
            "inference_report_absolute": str(Path(report_path).resolve()),
            "inference_run_dir_absolute": str(Path(out_root).resolve()),
            "producer": "smartrain.inference",
        },
        "summary": {
            "images_input": len([r for r in images if isinstance(r, dict)]),
            "images_exported": images_exported,
            "images_skipped_empty": images_skipped_empty,
            "labels_total": labels_total,
        },
        "file_mapping": file_mapping,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return (
        ExportSummary(
            dataset_dir=str(dataset_dir.resolve()),
            manifest_path=str(manifest_path.resolve()),
            images_exported=images_exported,
            labels_total=labels_total,
            images_skipped_empty=images_skipped_empty,
            overlay_paths=(),
            overlay_dir=None,
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
            preds = filter_task_outputs(row, task_type, options.label_conf_min, options.label_conf_max)
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
        if not saved:
            overlay_dir.mkdir(parents=True, exist_ok=True)
        out_path = overlay_dir / Path(src).name
        if out_path.exists():
            out_path = overlay_dir / f"{Path(src).stem}__{len(saved) + 1}{Path(src).suffix.lower() or '.jpg'}"
        save_rendered_image(rendered, out_path, original_format=original_format)
        saved.append(str(out_path.resolve()))
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
        return ExportSummary(
            dataset_dir=None,
            manifest_path=None,
            images_exported=0,
            labels_total=0,
            images_skipped_empty=0,
            overlay_paths=(),
            overlay_dir=None,
        )

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    exported_paths: set[str] = set()
    summary = ExportSummary(
        dataset_dir=None,
        manifest_path=None,
        images_exported=0,
        labels_total=0,
        images_skipped_empty=0,
        overlay_paths=(),
        overlay_dir=None,
    )

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
            print(
                f"[OK] Autolabel dataset: {summary.dataset_dir} "
                f"(images={summary.images_exported}, labels={summary.labels_total})"
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
        )
        if overlay_paths:
            print(f"[OK] Saved {len(overlay_paths)} prediction overlay image(s) to {overlay_dir}")

    patch_inference_report_artifacts(report_path, layout=layout, options=options, summary=summary)
    return summary
