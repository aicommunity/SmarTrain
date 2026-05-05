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

from smartrain.environment_profile import collect_environment_profile, write_environment_profile
from smartrain.external_model_ref import parse_external_model_ref, validate_external_model_ref
from smartrain.backends.train_test_registry import resolve_infer_backend
from smartrain.external_providers.registry import list_provider_specs
from smartrain.inference_backends import ExternalProviderBackend, InferenceBackendRegistry
from smartrain.inference_perf import DualPerfProfiler
from smartrain.path_portable import relativize_if_under
from smartrain.provider_global_index import get_provider_location
from smartrain.train_model_catalog import TrainModelCatalog, is_supported_external_provider_model
from smartrain.ultralytics_ephemeral import ultralytics_sidecar_dir
from smartrain.workspace_paths import WorkspaceLayout


def _backend_name_matches_capability(runtime_name: str | None, capability_backend: str) -> bool:
    actual = str(runtime_name or "").strip().lower()
    expected = str(capability_backend or "").strip().lower()
    if not actual or not expected:
        return False
    return actual == expected or actual.startswith(f"{expected}:")


def run_inference_job(args: argparse.Namespace, layout: WorkspaceLayout) -> tuple[int, bool]:
    """
    Run inference after CLI validated workspace, device, and interactive/non-interactive args.

    Returns ``(exit_code, always_system_exit)``. External-provider runs historically always
    finished with ``SystemExit(code)`` (including code 0); ``always_system_exit`` is True for
    those. Local Ultralytics/backend runs use ``(code, False)`` so the CLI can return normally
    on success (code 0) and only raise on error.
    """
    # Late import: helpers live in inference_cli; avoids import cycle with cli module.
    from smartrain import inference_cli as ic

    known_provider_ids = {spec.id for spec in list_provider_specs()}
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

    ext_provider = str(getattr(args, "external_provider", "") or "").strip()
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
        maybe_file = Path(raw_model_value).expanduser()
        if not maybe_file.is_file():
            is_supported = is_supported_external_provider_model(
                ext_provider,
                raw_model_value,
                provider_repo_path=repo_path or None,
            )
            if not is_supported:
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
        ext_backend = ExternalProviderBackend(ext_provider, repo_path, venv_path)
        rc = ext_backend.run_batch(
            model_path=str(model_path),
            source_path=source_for_external,
            conf=float(args.conf),
            imgsz=int(args.img_size),
            device=str(args.device) if args.device else None,
        )
        external_report = {
            "created_at": datetime.utcnow().isoformat() + "Z",
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
            },
            "summary": {"images_input": None, "images_processed": None, "images_skipped": None, "detections_total": None},
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
            "images": [],
        }
        ic._write_report(report_path, external_report)
        print(f"[OK] External inference report: {report_path}")
        return int(rc), True

    registry = InferenceBackendRegistry()
    model_format = str(model_path.suffix).lower().lstrip(".")
    if model_format not in {"pt", "onnx", "engine", "trt"}:
        print(f"[ERROR] Unsupported model format for inference: {model_format}", file=sys.stderr)
        return 1, False
    try:
        expected_caps = resolve_infer_backend(task_type="detection", model_format=model_format)
    except Exception as e:
        print(f"[ERROR] No registered inference backend capability for format {model_format!r}: {e}", file=sys.stderr)
        return 1, False
    try:
        backend = registry.create_local_backend(model_format=model_format, model_path=str(model_path))
    except Exception as e:
        print(f"[ERROR] Failed to initialize inference backend: {e}", file=sys.stderr)
        return 1, False
    if not _backend_name_matches_capability(getattr(backend, "name", None), expected_caps.backend):
        print(
            f"[WARN] Inference backend mismatch: capability resolver expects {expected_caps.backend!r} "
            f"for format {model_format!r}, got {getattr(backend, 'name', None)!r}. "
            "Continuing with runtime-selected backend.",
            file=sys.stderr,
        )
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
    ic._write_report(
        report_path,
        ic._build_report(
            args=args,
            layout=layout,
            model_source=model_source,
            model_name=model_name,
            model_path=model_path,
            source_abs=source_abs,
            source_short=source_short,
            out_root=out_root,
            report_path=report_path,
            images_input_count=len(images),
            image_rows=image_rows,
            skipped=skipped,
            performance=perf.to_payload(methodology=perf_methodology),
            environment_artifact_path=env_path,
        ),
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
                    ic._write_report(
                        report_path,
                        ic._build_report(
                            args=args,
                            layout=layout,
                            model_source=model_source,
                            model_name=model_name,
                            model_path=model_path,
                            source_abs=source_abs,
                            source_short=source_short,
                            out_root=out_root,
                            report_path=report_path,
                            images_input_count=len(images),
                            image_rows=image_rows,
                            skipped=skipped,
                            performance=perf.to_payload(methodology=perf_methodology),
                            environment_artifact_path=env_path,
                        ),
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
        )
        perf.record_infer_only(int(pred_result.infer_only_ns))
        for stage, dt in pred_result.stage_ns.items():
            perf.record_stage(stage, int(dt))
        boxes_payload: list[dict[str, Any]] = []
        for det in pred_result.detections:
            x1, y1, x2, y2 = [float(v) for v in det.get("bbox_roi_xyxy", [0.0, 0.0, 0.0, 0.0])]
            if roi_box is not None:
                ox1 = x1 + float(roi_box[0])
                oy1 = y1 + float(roi_box[1])
                ox2 = x2 + float(roi_box[0])
                oy2 = y2 + float(roi_box[1])
            else:
                ox1, oy1, ox2, oy2 = x1, y1, x2, y2
            boxes_payload.append(
                {
                    "bbox_roi_xyxy": [x1, y1, x2, y2],
                    "bbox_original_xyxy": [ox1, oy1, ox2, oy2],
                    "class_index": int(det.get("class_index", 0)),
                    "class_name": str(det.get("class_name", det.get("class_index", ""))),
                    "confidence": float(det.get("confidence", 0.0)),
                }
            )
        image_rows.append(
            {
                "image_path_absolute": image_path_abs,
                "image_path_relative": relativize_if_under(layout.root, image_path_abs) or image_path_abs,
                "image_size": {"width": iw, "height": ih},
                "roi_xyxy": list(roi_box) if roi_box is not None else None,
                "detections": boxes_payload,
            }
        )
        perf.record_end_to_end(int(time.perf_counter_ns() - loop_t0))
        ic._write_report(
            report_path,
            ic._build_report(
                args=args,
                layout=layout,
                model_source=model_source,
                model_name=model_name,
                model_path=model_path,
                source_abs=source_abs,
                source_short=source_short,
                out_root=out_root,
                report_path=report_path,
                images_input_count=len(images),
                image_rows=image_rows,
                skipped=skipped,
                performance=perf.to_payload(methodology=perf_methodology),
                environment_artifact_path=env_path,
            ),
        )

    print(f"[OK] Inference done: {len(image_rows)} images, skipped={skipped}")
    print(f"[OK] Report: {report_path}")
    return 0, False
