"""Inference execution pipeline (non-CLI): resolve model, run external or local backend, write report."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


class InferenceJobOutcome(NamedTuple):
    """CLI maps this to SystemExit; external batch historically always raised SystemExit(rc)."""

    exit_code: int
    exit_via_sysexit_always: bool
from PIL import Image
from tqdm import tqdm

from smartrain.core.runtime.environment_profile import collect_environment_profile, write_environment_profile
from smartrain.backends.external_provider_adapter import ExternalProviderAdapter
from smartrain.backends.ultralytics_adapter import UltralyticsAdapter
from smartrain.core.training.external_model_ref import parse_external_model_ref, validate_external_model_ref
from smartrain.backends.train_test_registry import resolve_infer_backend
from smartrain.external_providers.registry import list_provider_specs
from smartrain.workflows.inference.inference_perf import DualPerfProfiler
from smartrain.core.runtime.path_portable import relativize_if_under
from smartrain.providers.core.global_index import get_provider_location
from smartrain.core.training.train_model_catalog import TrainModelCatalog, is_supported_external_provider_model
from smartrain.core.training.train_profile import task_to_metadata_task_type
from smartrain.core.runtime.ultralytics_ephemeral import ultralytics_sidecar_dir
from smartrain.core.runtime.workspace_paths import WorkspaceLayout


def _backend_name_matches_capability(runtime_name: str | None, capability_backend: str) -> bool:
    actual = str(runtime_name or "").strip().lower()
    expected = str(capability_backend or "").strip().lower()
    if not actual or not expected:
        return False
    if actual == expected or actual.startswith(f"{expected}:"):
        return True
    # Capability backends are logical IDs, while runtime backends may expose
    # implementation-specific names (for example "ultralytics:engine").
    alias_accept: dict[str, tuple[str, ...]] = {
        "tensorrt": ("ultralytics:engine", "ultralytics:trt"),
        "onnxruntime": ("ultralytics:onnx",),
    }
    return any(actual == token or actual.startswith(f"{token}:") for token in alias_accept.get(expected, ()))


def _offset_bbox(bbox_roi_xyxy: list[Any], roi_box: tuple[int, int, int, int] | None) -> tuple[list[float], list[float]]:
    x1, y1, x2, y2 = [float(v) for v in (bbox_roi_xyxy[:4] + [0.0, 0.0, 0.0, 0.0])[:4]]
    if roi_box is None:
        return [x1, y1, x2, y2], [x1, y1, x2, y2]
    ox1 = x1 + float(roi_box[0])
    oy1 = y1 + float(roi_box[1])
    ox2 = x2 + float(roi_box[0])
    oy2 = y2 + float(roi_box[1])
    return [x1, y1, x2, y2], [ox1, oy1, ox2, oy2]


def _offset_polygon(
    polygon_roi_xy: list[list[Any]] | Any,
    roi_box: tuple[int, int, int, int] | None,
) -> list[list[float]]:
    if not isinstance(polygon_roi_xy, list):
        return []
    out: list[list[float]] = []
    for point in polygon_roi_xy:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except Exception:
            continue
        if roi_box is not None:
            x += float(roi_box[0])
            y += float(roi_box[1])
        out.append([x, y])
    return out


def _build_task_outputs_payload(
    task_type: str,
    outputs: dict[str, Any],
    *,
    roi_box: tuple[int, int, int, int] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detections_payload: list[dict[str, Any]] = []
    task_payload: dict[str, Any] = {}
    if task_type == "classification":
        cls = outputs.get("classification")
        task_payload["classification"] = cls if isinstance(cls, dict) else {}
        return detections_payload, task_payload
    if task_type == "segmentation":
        segments_raw = outputs.get("segments")
        segments_payload: list[dict[str, Any]] = []
        if isinstance(segments_raw, list):
            for seg in segments_raw:
                if not isinstance(seg, dict):
                    continue
                bbox_roi, bbox_original = _offset_bbox(seg.get("bbox_roi_xyxy", []), roi_box)
                segments_payload.append(
                    {
                        "bbox_roi_xyxy": bbox_roi,
                        "bbox_original_xyxy": bbox_original,
                        "class_index": int(seg.get("class_index", 0)),
                        "class_name": str(seg.get("class_name", seg.get("class_index", ""))),
                        "confidence": float(seg.get("confidence", 0.0)),
                        "polygon_roi_xy": _offset_polygon(seg.get("polygon_roi_xy", []), None),
                        "polygon_original_xy": _offset_polygon(seg.get("polygon_roi_xy", []), roi_box),
                    }
                )
        task_payload["segments"] = segments_payload
        return detections_payload, task_payload
    detections_raw = outputs.get("detections")
    if isinstance(detections_raw, list):
        for det in detections_raw:
            if not isinstance(det, dict):
                continue
            bbox_roi, bbox_original = _offset_bbox(det.get("bbox_roi_xyxy", []), roi_box)
            detections_payload.append(
                {
                    "bbox_roi_xyxy": bbox_roi,
                    "bbox_original_xyxy": bbox_original,
                    "class_index": int(det.get("class_index", 0)),
                    "class_name": str(det.get("class_name", det.get("class_index", ""))),
                    "confidence": float(det.get("confidence", 0.0)),
                }
            )
    task_payload["detections"] = detections_payload
    return detections_payload, task_payload


def _normalize_external_batch_result(raw_result: Any) -> tuple[int, list[dict[str, Any]]]:
    if isinstance(raw_result, dict):
        rc_raw = raw_result.get("return_code", raw_result.get("rc", raw_result.get("code", 0)))
        try:
            rc = int(rc_raw)
        except Exception:
            rc = 1
        images_raw = raw_result.get("images")
        images = images_raw if isinstance(images_raw, list) else []
        return rc, [img for img in images if isinstance(img, dict)]
    try:
        return int(raw_result), []
    except Exception:
        return 1, []


def _task_outputs_count(task_type: str, task_outputs: dict[str, Any]) -> int:
    if task_type == "classification":
        cls = task_outputs.get("classification")
        return 1 if isinstance(cls, dict) and cls else 0
    if task_type == "segmentation":
        segs = task_outputs.get("segments")
        return len(segs) if isinstance(segs, list) else 0
    dets = task_outputs.get("detections")
    return len(dets) if isinstance(dets, list) else 0


def _normalize_task_outputs_for_task(task_type: str, task_outputs: dict[str, Any], detections: list[dict[str, Any]]) -> dict[str, Any]:
    if task_type == "classification":
        cls = task_outputs.get("classification")
        return {"classification": cls if isinstance(cls, dict) else {}}
    if task_type == "segmentation":
        segs = task_outputs.get("segments")
        return {"segments": segs if isinstance(segs, list) else []}
    dets = task_outputs.get("detections")
    if isinstance(dets, list):
        return {"detections": dets}
    return {"detections": detections}


def _normalize_external_image_rows(task_type: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        task_outputs = row.get("task_outputs")
        if not isinstance(task_outputs, dict):
            task_outputs = {}
        detections = row.get("detections")
        if not isinstance(detections, list):
            detections = []
        # Bridge legacy external payloads: allow flat "detections" without task_outputs.
        if not task_outputs and detections:
            task_outputs = {"detections": detections}
        normalized_task = _normalize_task_outputs_for_task(task_type, task_outputs, detections)
        capability_gap = False
        if task_type == "classification":
            capability_gap = not bool(normalized_task.get("classification"))
        elif task_type == "segmentation":
            capability_gap = len(normalized_task.get("segments", [])) == 0
        normalized.append(
            {
                "image_path_absolute": row.get("image_path_absolute"),
                "image_path_relative": row.get("image_path_relative"),
                "image_size": row.get("image_size"),
                "roi_xyxy": row.get("roi_xyxy"),
                "task_type": task_to_metadata_task_type(row.get("task_type", task_type)),
                "detections": detections,
                "task_outputs": normalized_task,
                "capability_gap": capability_gap,
            }
        )
    return normalized


def _write_local_inference_report(
    *,
    ic_module: Any,
    report_path: str,
    args: argparse.Namespace,
    layout: WorkspaceLayout,
    model_source: str,
    model_name: str,
    model_path: Path,
    source_abs: str,
    source_short: str,
    out_root: str,
    images_input_count: int,
    image_rows: list[dict[str, Any]],
    skipped: int,
    performance_payload: dict[str, Any],
    environment_artifact_path: str,
) -> None:
    ic_module._write_report(
        report_path,
        ic_module._build_report(
            args=args,
            layout=layout,
            model_source=model_source,
            model_name=model_name,
            model_path=model_path,
            source_abs=source_abs,
            source_short=source_short,
            out_root=out_root,
            report_path=report_path,
            images_input_count=images_input_count,
            image_rows=image_rows,
            skipped=skipped,
            performance=performance_payload,
            environment_artifact_path=environment_artifact_path,
        ),
    )


def _apply_external_provider_inference_from_refs(
    args: argparse.Namespace,
    *,
    known_provider_ids: set[str],
) -> tuple[int, bool] | None:
    try:
        parsed_weights_ref = validate_external_model_ref(
            parse_external_model_ref(getattr(args, "weights", None)),
            known_provider_ids=known_provider_ids,
        )
        parsed_model_name_ref = validate_external_model_ref(
            parse_external_model_ref(getattr(args, "model_name", None)),
            known_provider_ids=known_provider_ids,
        )
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2, False
    if parsed_weights_ref.is_external and parsed_weights_ref.provider_id and not getattr(args, "external_provider", None):
        args.external_provider = parsed_weights_ref.provider_id
        args.weights = parsed_weights_ref.model_ref
        print(f"[INFO] External provider inferred from --weights: {parsed_weights_ref.provider_id}")
    if parsed_model_name_ref.is_external and parsed_model_name_ref.provider_id and not getattr(args, "external_provider", None):
        args.external_provider = parsed_model_name_ref.provider_id
        args.model_name = None
        args.weights = parsed_model_name_ref.model_ref
        print(f"[INFO] External provider inferred from --model-name: {parsed_model_name_ref.provider_id}")
    return None


def _validate_external_inference_model_or_fail(
    *,
    ext_provider: str,
    raw_model_value: str,
    repo_path: str,
) -> tuple[int, bool] | None:
    maybe_file = Path(raw_model_value).expanduser()
    if maybe_file.is_file():
        return None
    is_supported = is_supported_external_provider_model(
        ext_provider,
        raw_model_value,
        provider_repo_path=repo_path or None,
    )
    if is_supported:
        return None
    aliases = TrainModelCatalog(
        provider=ext_provider,
        provider_repo_path=repo_path or None,
    ).supported_aliases()
    known = ", ".join(aliases) if aliases else "<none>"
    print(
        f"[ERROR] Model {raw_model_value!r} is not supported by external provider "
        f"{ext_provider!r}. Supported aliases: {known}",
        file=sys.stderr,
    )
    return 2, True


def run_inference_job(args: argparse.Namespace, layout: WorkspaceLayout) -> tuple[int, bool]:
    """
    Run inference after CLI validated workspace, device, and interactive/non-interactive args.

    Returns ``(exit_code, always_system_exit)``. External-provider runs historically always
    finished with ``SystemExit(code)`` (including code 0); ``always_system_exit`` is True for
    those. Local Ultralytics/backend runs use ``(code, False)`` so the CLI can return normally
    on success (code 0) and only raise on error.
    """
    # Late import: helpers live in inference_cli; avoids import cycle with cli module.
    from smartrain.workflows.inference import inference_cli as ic

    known_provider_ids = {spec.id for spec in list_provider_specs()}
    external_ref_outcome = _apply_external_provider_inference_from_refs(
        args, known_provider_ids=known_provider_ids
    )
    if external_ref_outcome is not None:
        return external_ref_outcome

    ext_provider = str(getattr(args, "external_provider", "") or "").strip()
    task_type = task_to_metadata_task_type(getattr(args, "task", None))
    if ext_provider and args.weights:
        raw_weight = str(args.weights).strip()
        maybe_path = Path(raw_weight).expanduser()
        if maybe_path.is_file():
            model_path = maybe_path.resolve()
            model_name = model_path.stem
            model_source = "weights"
        else:
            model_path = Path(raw_weight)
            model_name = ic._sanitize_segment(model_path.name or raw_weight)
            model_source = "external-model"
    else:
        try:
            model_path, model_name, model_source = ic._resolve_model(args, layout)
        except Exception as e:
            print(f"[ERROR] Failed to resolve model: {e}", file=sys.stderr)
            return 1, False
    if args.img_size is None:
        inferred = ic._infer_img_size_from_model_context(model_path) if isinstance(model_path, Path) else None
        args.img_size = int(inferred) if inferred is not None else 640

    if ext_provider:
        location = get_provider_location(ext_provider)
        if location is None and not getattr(args, "external_repo", None):
            print(
                f"[ERROR] External provider {ext_provider!r} is not installed. "
                "Use `smartrain providers install` or pass --external-repo.",
                file=sys.stderr,
            )
            return 1, True
        repo_path = str(getattr(args, "external_repo", "") or "").strip() or (location.repo_path if location else "")
        venv_path = location.venv_path if location else os.path.join(repo_path, "venv")
        if not venv_path:
            print(f"[ERROR] Missing venv for external provider {ext_provider!r}.", file=sys.stderr)
            return 1, True
        raw_model_value = str(getattr(args, "weights", "") or "")
        model_validation_outcome = _validate_external_inference_model_or_fail(
            ext_provider=ext_provider,
            raw_model_value=raw_model_value,
            repo_path=repo_path,
        )
        if model_validation_outcome is not None:
            return model_validation_outcome
        source_for_external = ic._resolve_external_source(args, layout)
        source_short = (
            os.path.basename(os.path.abspath(os.path.expanduser(str(args.source_dir))).rstrip(os.sep)) or "folder"
            if args.data_mode == "folder"
            else f"{args.dataset}-{args.split}"
        )
        out_root = ic._resolve_output_root(layout, model_name, source_short)
        report_path = os.path.join(out_root, "inference_results.json")
        env_profile = collect_environment_profile()
        env_path = os.path.join(out_root, "environment_profile.json")
        write_environment_profile(env_path, env_profile)
        ext_adapter = ExternalProviderAdapter(
            provider_id=ext_provider,
            repo_path=repo_path,
            venv_path=venv_path,
        )
        raw_result = ext_adapter.run_batch(
            model_path=str(model_path),
            source_path=source_for_external,
            conf=float(args.conf),
            imgsz=int(args.img_size),
            device=str(args.device) if args.device else None,
            task_type=task_type,
        )
        rc, ext_images = _normalize_external_batch_result(raw_result)
        ext_result_diagnostics = raw_result.get("diagnostics") if isinstance(raw_result, dict) else None
        ext_image_rows = _normalize_external_image_rows(task_type, ext_images)
        detections_total = sum(len(x.get("detections", [])) for x in ext_image_rows)
        task_outputs_total = sum(
            _task_outputs_count(task_type, x.get("task_outputs") if isinstance(x.get("task_outputs"), dict) else {})
            for x in ext_image_rows
        )
        capability_gap_images = sum(1 for x in ext_image_rows if bool(x.get("capability_gap")))
        if capability_gap_images > 0 and task_type in {"classification", "segmentation"}:
            print(
                "[WARN] External provider did not expose full task-specific outputs "
                + f"for {capability_gap_images} images (task={task_type}); degraded contract applied."
            )
        external_report = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "task_type": task_type,
            "workspace": {"root_absolute": layout.root},
            "model": {
                "source": "external",
                "name": model_name,
                "provider": {"type": "external", "id": ext_provider},
                "weights_value": str(model_path),
            },
            "parameters": {
                "conf": args.conf,
                "img_size": int(args.img_size),
                "device": args.device,
                "data_mode": args.data_mode,
            },
            "source": ic._source_descriptor(args, source_for_external, source_short, layout),
            "output": {
                "dir_absolute": out_root,
                "dir_relative": relativize_if_under(layout.root, out_root) or out_root,
                "json_absolute": report_path,
                "json_relative": relativize_if_under(layout.root, report_path) or report_path,
            },
            "external_execution": {
                "provider_id": ext_provider,
                "repo_path": repo_path,
                "venv_path": venv_path,
                "return_code": int(rc),
                "diagnostics": ext_result_diagnostics if isinstance(ext_result_diagnostics, dict) else {},
            },
            "summary": {
                "images_input": len(ext_image_rows),
                "images_processed": len(ext_image_rows),
                "images_skipped": 0,
                "detections_total": detections_total,
                "task_outputs_total": task_outputs_total,
                "capability_gap_images": capability_gap_images,
            },
            "performance": {
                "end_to_end": None,
                "infer_only": None,
                "stage_breakdown_ms": {},
                "methodology": {
                    "profile_mode": "dual",
                    "caveats": ["External provider currently does not expose per-image performance telemetry."],
                },
            },
            "artifacts": {
                "environment_profile": {
                    "path_absolute": env_path,
                    "path_relative": relativize_if_under(layout.root, env_path) or env_path,
                }
            },
            "images": ext_image_rows,
        }
        ic._write_report(report_path, external_report)
        print(f"[OK] External inference report: {report_path}")
        return int(rc), True

    model_format = str(model_path.suffix).lower().lstrip(".")
    if model_format not in {"pt", "onnx", "engine", "trt"}:
        print(f"[ERROR] Unsupported model format for inference: {model_format}", file=sys.stderr)
        return 1, False
    try:
        expected_caps = resolve_infer_backend(task_type=task_type, model_format=model_format)
    except Exception as e:
        print(f"[ERROR] No registered inference backend capability for format {model_format!r}: {e}", file=sys.stderr)
        return 1, False
    adapter = UltralyticsAdapter()
    try:
        backend = adapter.create_inference_backend(model_format=model_format, model_path=str(model_path), task_type=task_type)
    except Exception as e:
        print(f"[ERROR] Failed to initialize inference backend: {e}", file=sys.stderr)
        return 1, False
    if not _backend_name_matches_capability(getattr(backend, "name", None), expected_caps.backend):
        print(
            f"[ERROR] Inference backend mismatch: capability resolver expects {expected_caps.backend!r} "
            f"for format {model_format!r}, got {getattr(backend, 'name', None)!r}. "
            "Aborting to keep capability routing deterministic.",
            file=sys.stderr,
        )
        return 1, False
    roi_model = None
    if args.roi_pre_detect:
        from ultralytics import YOLO

        if args.data_mode != "folder":
            print("[ERROR] --roi-pre-detect is supported only for --data-mode folder.", file=sys.stderr)
            return 1, False
        roi_w = args.roi_weights or str(model_path)
        roi_model = YOLO(str(roi_w))
        args.roi_weights = roi_w
        args._ultralytics_roi_project = ultralytics_sidecar_dir(layout.root, ".cache", "ultralytics_roi_infer")

    try:
        if args.data_mode == "folder":
            images = ic._collect_folder_images(str(args.source_dir), int(args.limit))
            source_abs = os.path.abspath(os.path.expanduser(str(args.source_dir)))
            source_short = os.path.basename(source_abs.rstrip(os.sep)) or "folder"
        else:
            images, split_dir = ic._collect_split_images_for_dataset(
                layout,
                str(args.dataset),
                str(args.split),
                int(args.limit),
            )
            source_abs = split_dir
            source_short = f"{args.dataset}-{args.split}"
    except Exception as e:
        print(f"[ERROR] Failed to resolve inference source: {e}", file=sys.stderr)
        return 1, False
    if not images:
        print("[ERROR] No images found for inference.", file=sys.stderr)
        return 1, False

    out_root = ic._resolve_output_root(layout, model_name, source_short)
    report_path = os.path.join(out_root, "inference_results.json")
    env_profile = collect_environment_profile()
    env_path = os.path.join(out_root, "environment_profile.json")
    write_environment_profile(env_path, env_profile)

    image_rows: list[dict[str, Any]] = []
    skipped = 0
    perf = DualPerfProfiler(warmup_images=int(max(0, args.perf_warmup_images)))
    perf_methodology = {
        "profile_mode": "dual",
        "warmup_images": int(max(0, args.perf_warmup_images)),
        "end_to_end_includes": ["image_io", "roi_preprocess", "model_infer", "postprocess", "report_update"],
        "infer_only_includes": ["model_infer_call"],
        "backend": backend.name,
        "model_format": model_format,
    }
    _write_local_inference_report(
        ic_module=ic,
        report_path=report_path,
        args=args,
        layout=layout,
        model_source=model_source,
        model_name=model_name,
        model_path=model_path,
        source_abs=source_abs,
        source_short=source_short,
        out_root=out_root,
        images_input_count=len(images),
        image_rows=image_rows,
        skipped=skipped,
        performance_payload=perf.to_payload(methodology=perf_methodology),
        environment_artifact_path=env_path,
    )
    progress_desc = f"inference:{args.data_mode}"
    for image_path in tqdm(images, desc=progress_desc, unit="img"):
        loop_t0 = time.perf_counter_ns()
        image_path_abs = os.path.abspath(image_path)
        with Image.open(image_path_abs) as im:
            im_rgb = im.convert("RGB")
            iw, ih = im_rgb.size
            roi_box: tuple[int, int, int, int] | None = None
            src_for_predict: Any = image_path_abs
            if roi_model is not None:
                rb = ic._predict_roi_crop(roi_model, image_path_abs, args)
                if rb[0] < 0:
                    skipped += 1
                    _write_local_inference_report(
                        ic_module=ic,
                        report_path=report_path,
                        args=args,
                        layout=layout,
                        model_source=model_source,
                        model_name=model_name,
                        model_path=model_path,
                        source_abs=source_abs,
                        source_short=source_short,
                        out_root=out_root,
                        images_input_count=len(images),
                        image_rows=image_rows,
                        skipped=skipped,
                        performance_payload=perf.to_payload(methodology=perf_methodology),
                        environment_artifact_path=env_path,
                    )
                    perf.record_end_to_end(int(time.perf_counter_ns() - loop_t0))
                    continue
                roi_box = rb
                crop = im_rgb.crop((roi_box[0], roi_box[1], roi_box[2], roi_box[3]))
                src_for_predict = np.asarray(crop)
            elif args.data_mode == "folder":
                roi_box = (0, 0, iw, ih)

        pred_result = backend.predict(
            src_for_predict,
            conf=float(args.conf),
            imgsz=int(args.img_size),
            device=str(args.device),
            half=bool(args.half),
            task_type=task_type,
        )
        perf.record_infer_only(int(pred_result.infer_only_ns))
        for stage, dt in pred_result.stage_ns.items():
            perf.record_stage(stage, int(dt))
        resolved_pred_task = task_to_metadata_task_type(getattr(pred_result, "task_type", task_type))
        detections_payload, task_outputs_payload = _build_task_outputs_payload(
            resolved_pred_task,
            pred_result.outputs if isinstance(pred_result.outputs, dict) else {},
            roi_box=roi_box,
        )
        image_rows.append(
            {
                "image_path_absolute": image_path_abs,
                "image_path_relative": relativize_if_under(layout.root, image_path_abs) or image_path_abs,
                "image_size": {"width": iw, "height": ih},
                "roi_xyxy": list(roi_box) if roi_box is not None else None,
                "task_type": resolved_pred_task,
                "detections": detections_payload,
                "task_outputs": task_outputs_payload,
            }
        )
        perf.record_end_to_end(int(time.perf_counter_ns() - loop_t0))
        _write_local_inference_report(
            ic_module=ic,
            report_path=report_path,
            args=args,
            layout=layout,
            model_source=model_source,
            model_name=model_name,
            model_path=model_path,
            source_abs=source_abs,
            source_short=source_short,
            out_root=out_root,
            images_input_count=len(images),
            image_rows=image_rows,
            skipped=skipped,
            performance_payload=perf.to_payload(methodology=perf_methodology),
            environment_artifact_path=env_path,
        )

    print(f"[OK] Inference done: {len(image_rows)} images, skipped={skipped}")
    print(f"[OK] Report: {report_path}")
    return 0, False
