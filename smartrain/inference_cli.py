from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from smartrain.cli_support.cli_argparse import CliArgumentParser
from smartrain.cli_support.cli_replay import print_replay_command  # backward-compatible symbol for tests/mocks
from smartrain.cli_support.cli_prompts import print_numbered_options, prompt_choice, prompt_text, prompt_yes_no
from smartrain.cli_support.cli_contracts import emit_replay, make_command_request
from smartrain.workflows.datasets.dataset_access import resolve_dataset_root_for_entry
from smartrain.workflows.datasets.dataset_roi_yolo import ON_EMPTY_MODES, ROI_POLICIES, _clamp_crop, _full_image_crop, _select_roi_boxes
from smartrain.datasets_json_former import find_yaml_file
from smartrain.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.path_portable import relativize_if_under
from smartrain.results_analyzer import find_run_directories
from smartrain.core.runtime.ultralytics_ephemeral import best_effort_prune_workspace_runs_detect, ultralytics_sidecar_dir
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.core.runtime.device_selector import (
    default_device_value,
    device_display_name,
    prompt_device_selection,
    resolve_device_request,
    validate_device_available,
)
from smartrain.workflows.models.model_context import infer_img_size_from_model_context
from smartrain.core.runtime.run_artifacts import is_internal_conversion_artifact
from smartrain.canonical.schema import wrap_inference_report_v2
from smartrain.core.training.train_profile import task_to_metadata_task_type
from smartrain.services.inference_service import run_inference_job
from smartrain.canonical.policy import emit_legacy_read_deprecation_warnings
from smartrain.canonical.refs import canonical_target_from_model_dir

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MANIFEST_NAME = "model_manifest.json"
DATA_MODES = ("folder", "dataset-split")
SUPPORTED_INFERENCE_EXTS = {".pt", ".onnx", ".engine", ".trt"}


def build_inference_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Run object detection inference and save JSON report (empty call starts interactive mode)."
    )
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR}).")
    p.add_argument("--model-name", type=str, default=None, help="Promoted model directory name from workspace/models.")
    p.add_argument("--run", type=str, default=None, help="Run path or run index from workspace/runs list.")
    p.add_argument("--weights", type=str, default=None, help="Explicit model weights path (.pt/.onnx/.engine/.trt).")
    p.add_argument("--data-mode", choices=DATA_MODES, default="folder", help="Data source mode.")
    p.add_argument("--source-dir", type=str, default=None, help="Folder with images (recursive).")
    p.add_argument("--dataset", type=str, default=None, help="Dataset key from datasets/datasets_info.json.")
    p.add_argument("--split", choices=("train", "val", "test"), default="test", help="Dataset split for dataset-split mode.")
    p.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all).")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for inference model.")
    p.add_argument("--img-size", type=int, default=None, help="Inference input resolution (imgsz).")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Ultralytics device (cpu, 0, etc). Default: GPU 0 if available, otherwise cpu.",
    )
    p.add_argument("--half", action="store_true", help="Enable FP16 where supported.")
    p.add_argument("--perf-warmup-images", type=int, default=5, help="Warmup images excluded from steady perf statistics.")
    p.add_argument("--roi-pre-detect", action="store_true", help="Pre-detect ROI before inference (folder mode only).")
    p.add_argument("--roi-weights", type=str, default=None, help="ROI detector weights path (.pt/.onnx).")
    p.add_argument("--roi-conf", type=float, default=0.25, help="Confidence threshold for ROI detector.")
    p.add_argument("--roi-policy", choices=ROI_POLICIES, default="largest", help="ROI selection policy.")
    p.add_argument("--roi-pad-px", type=int, default=0, help="Padding in pixels around selected ROI.")
    p.add_argument("--roi-on-empty", choices=ON_EMPTY_MODES, default="full_image", help="Behavior when ROI detector has no detections.")
    p.add_argument("--roi-class-ids", type=str, default=None, help="CSV class ids for ROI detector (empty=all).")
    p.add_argument("--external-provider", type=str, default=None, help="External provider id for inference.")
    p.add_argument("--external-repo", type=str, default=None, help="Override external provider repository path.")
    p.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["detect", "segment", "classify", "detection", "segmentation", "classification"],
        help="Task type hint for task-aware backend routing (default: detection).",
    )
    return p


def _sanitize_segment(value: str) -> str:
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


def _discover_model_names(layout: WorkspaceLayout) -> list[str]:
    root = Path(layout.models)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _discover_model_entries(layout: WorkspaceLayout) -> list[tuple[str, str, str]]:
    """
    Returns tuples: (display_label, model_name_arg_value, model_dir_name).
    model_name_arg_value can be either directory name or a relative file path under models/.
    """
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


def _pick_preferred_model_path(candidates: list[Path], *, prefer_onnx_variant: bool = False) -> Path | None:
    filtered = [p for p in candidates if p.is_file() and not is_internal_conversion_artifact(p)]
    if not filtered:
        return None

    ext_rank = {".pt": 4, ".onnx": 3, ".engine": 2, ".trt": 1}

    def _score(p: Path) -> tuple[int, int, float, str]:
        variant_hint = 1 if "_imgsz" in p.name.lower() and "_b" in p.name.lower() else 0
        onnx_variant_bonus = 2 if (prefer_onnx_variant and p.suffix.lower() == ".onnx" and variant_hint) else 0
        return (onnx_variant_bonus, ext_rank.get(p.suffix.lower(), 0), p.stat().st_mtime, p.name.lower())

    return max(filtered, key=_score)


def _resolve_model_from_name(layout: WorkspaceLayout, name: str) -> tuple[Path, str]:
    """
    Resolve promoted model directory name into a resolved weights path.

    This helper is intentionally canonical-only (no env-driven legacy fallback).
    """
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

    canonical = canonical_target_from_model_dir(mdir)
    return canonical.model_path.resolve(), name


def _resolve_model(args: argparse.Namespace, layout: WorkspaceLayout) -> tuple[Path, str, str]:
    def _resolve_and_validate_canonical_weights(model: Any) -> Path:
        p = Path(str(model.weights_path)).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Canonical weights not found: {p}")
        if p.suffix.lower() not in SUPPORTED_INFERENCE_EXTS:
            raise FileNotFoundError(f"Unsupported canonical weights format: {p.suffix}")
        return p

    # Canonical-only behavior:
    # legacy canonical-read env flags are deprecated/ignored for routing and only
    # trigger deterministic [DEPRECATION] warnings via `deprecation_policy`.
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


def _infer_img_size_from_model_context(model_path: Path) -> int | None:
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


def _collect_folder_images(source_dir: str, limit: int) -> list[str]:
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


def _collect_split_images_for_dataset(layout: WorkspaceLayout, dataset: str, split: str, limit: int) -> tuple[list[str], str]:
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


def _class_name_from_names(names: Any, idx: int) -> str:
    if isinstance(names, list) and 0 <= idx < len(names):
        return str(names[idx])
    if isinstance(names, dict):
        if idx in names:
            return str(names[idx])
        key = str(idx)
        if key in names:
            return str(names[key])
    return str(idx)


def _predict_roi_crop(roi_model: Any, image_path: str, args: argparse.Namespace) -> tuple[int, int, int, int]:
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


def _resolve_output_root(layout: WorkspaceLayout, model_name: str, source_short: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{ts}-{_sanitize_segment(source_short)}"
    out = os.path.join(layout.root, "inference", _sanitize_segment(model_name), run_name)
    os.makedirs(out, exist_ok=True)
    return out


def _interactive_fill(args: argparse.Namespace, layout: WorkspaceLayout) -> bool:
    print("[INFO] Interactive inference mode (Enter = default).")
    model_entries = _discover_model_entries(layout)
    model_options = ["models", "runs", "weights"]
    model_source = "models" if model_entries else "weights"
    model_source = prompt_choice("Model source", model_options, default=model_source)
    args.model_name = None
    args.run = None
    args.weights = None
    if model_source == "models":
        if not model_entries:
            print("[ERROR] No model files found in workspace/models.")
            return False
        labels = [x[0] for x in model_entries]
        print_numbered_options("models", labels)
        selected_label = prompt_choice(
            "Select model file from models",
            labels,
            default=labels[0],
            show_options=False,
        )
        selected = next((x for x in model_entries if x[0] == selected_label), None)
        if selected is None:
            print("[ERROR] Internal error: selected model not found.")
            return False
        args.model_name = selected[1]
    elif model_source == "runs":
        runs = find_run_directories(layout.runs)
        if not runs:
            print("[ERROR] No runs found in workspace/runs.")
            return False
        pretty: list[str] = []
        opts: list[str] = []
        for i, rd in enumerate(runs, start=1):
            rel = os.path.relpath(rd, layout.root)
            pretty.append(f"{i}. {rel}")
            opts.append(str(i))
        print("[INFO] Available runs:")
        for row in pretty:
            print(f"  {row}")
        args.run = prompt_choice("Select run index", opts, default=opts[0], show_options=False)
    else:
        args.weights = prompt_text("Weights path", default="models").strip()

    inferred_imgsz = None
    try:
        mpath, _mname, _msrc = _resolve_model(args, layout)
        inferred_imgsz = _infer_img_size_from_model_context(mpath)
    except Exception:
        inferred_imgsz = None

    args.data_mode = prompt_choice("Data mode", list(DATA_MODES), default=args.data_mode)
    if args.data_mode == "folder":
        args.source_dir = prompt_text("Source directory", default=args.source_dir or "datasets").strip()
        args.roi_pre_detect = prompt_yes_no("Enable ROI pre-detect", default=bool(args.roi_pre_detect))
        if args.roi_pre_detect:
            args.roi_weights = prompt_text("ROI weights (empty = main model)", default=str(args.roi_weights or "")).strip() or None
            args.roi_conf = float(prompt_text("ROI conf", default=str(args.roi_conf)).strip() or str(args.roi_conf))
            args.roi_policy = prompt_choice("ROI policy", list(ROI_POLICIES), default=args.roi_policy)
            args.roi_pad_px = int(prompt_text("ROI pad px", default=str(args.roi_pad_px)).strip() or str(args.roi_pad_px))
            args.roi_on_empty = prompt_choice("ROI on empty", list(ON_EMPTY_MODES), default=args.roi_on_empty)
            args.roi_class_ids = prompt_text("ROI class ids CSV (empty=all)", default=str(args.roi_class_ids or "")).strip() or None
    else:
        catalog = _load_catalog(layout)
        ds_names = sorted(catalog.keys())
        if not ds_names:
            print("[ERROR] datasets_info.json has no datasets.")
            return False
        print_numbered_options("datasets", ds_names)
        args.dataset = prompt_choice("Select dataset", ds_names, default=ds_names[0], show_options=False)
        args.split = prompt_choice("Split", ["train", "val", "test"], default=args.split)
        args.roi_pre_detect = False
        args.source_dir = None

    args.limit = int(prompt_text("Images limit (0=all)", default=str(args.limit)).strip() or str(args.limit))
    img_default = inferred_imgsz if inferred_imgsz is not None else (args.img_size if args.img_size is not None else 640)
    args.img_size = int(prompt_text("Input resolution (--img-size)", default=str(img_default)).strip() or str(img_default))
    args.conf = float(prompt_text("Inference conf", default=str(args.conf)).strip() or str(args.conf))
    args.device = prompt_device_selection(
        title="inference devices",
        default_device=str(args.device or default_device_value()),
    )
    args.half = prompt_yes_no("Use FP16 (--half)", default=bool(args.half))
    return True


def _ensure_device_available_or_exit(device: str | None) -> None:
    try:
        validate_device_available(device)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)


def _source_descriptor(args: argparse.Namespace, source_abs: str, source_short: str, layout: WorkspaceLayout) -> dict[str, Any]:
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


def _validate_non_interactive_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.data_mode == "folder" and not args.source_dir:
        parser.error("incomplete arguments: --source-dir is required for --data-mode folder.")
    if args.data_mode == "dataset-split" and not args.dataset:
        parser.error("incomplete arguments: --dataset is required for --data-mode dataset-split.")
    if not args.model_name and not args.run and not args.weights:
        parser.error("incomplete arguments: specify --model-name, --run or --weights.")


def _build_report(
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
        "source": _source_descriptor(args, source_abs, source_short, layout),
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


def _write_report(path: str, report: dict[str, Any]) -> None:
    report = wrap_inference_report_v2(report)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    request = make_command_request("inference", argv, interactive_allowed=is_interactive_allowed(argv))
    parser = build_inference_arg_parser()
    args = parser.parse_args(argv)
    args.device = resolve_device_request(args.device or default_device_value())

    try:
        workspace_root = resolve_workspace_root(args.workspace)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)
    layout = WorkspaceLayout(workspace_root)
    atexit.register(lambda wr=workspace_root: best_effort_prune_workspace_runs_detect(wr))
    os.makedirs(os.path.join(layout.root, "inference"), exist_ok=True)
    interactive_allowed = request.interactive_allowed
    interactive_used = False
    if len(argv) == 0 and interactive_allowed:
        if not sys.stdin.isatty():
            print(
                "[ERROR] Interactive inference mode requires a terminal (TTY). "
                "Run with explicit arguments in non-interactive environments.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if not _interactive_fill(args, layout):
            raise SystemExit(1)
        interactive_used = True
        request.interactive_used = True
        emit_replay(command_name="inference", parser=parser, args=args, stage="before launch")
    else:
        _validate_non_interactive_args(parser, args)
    _ensure_device_available_or_exit(args.device)
    print(f"[INFO] Inference device: {device_display_name(args.device)}")

    code, exit_via_sysexit = run_inference_job(args, layout)
    if exit_via_sysexit:
        raise SystemExit(code)
    if code != 0:
        raise SystemExit(code)
    if interactive_used:
        emit_replay(command_name="inference", parser=parser, args=args, stage="after execution")


def _resolve_external_source(args: argparse.Namespace, layout: WorkspaceLayout) -> str:
    if args.data_mode == "folder":
        return os.path.abspath(os.path.expanduser(str(args.source_dir)))
    _, split_dir = _collect_split_images_for_dataset(
        layout,
        str(args.dataset),
        str(args.split),
        int(args.limit),
    )
    return split_dir


if __name__ == "__main__":
    main()
