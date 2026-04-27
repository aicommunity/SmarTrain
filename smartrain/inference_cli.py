from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from smartrain.cli_argparse import CliArgumentParser
from smartrain.cli_prompts import print_numbered_options, prompt_choice, prompt_text, prompt_yes_no
from smartrain.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.dataset_access import resolve_dataset_root_for_entry
from smartrain.dataset_roi_yolo import ON_EMPTY_MODES, ROI_POLICIES, _clamp_crop, _full_image_crop, _select_roi_boxes
from smartrain.datasets_json_former import find_yaml_file
from smartrain.interactive_contract import is_interactive_allowed
from smartrain.path_portable import relativize_if_under
from smartrain.results_analyzer import find_run_directories
from smartrain.provider_global_index import get_provider_location
from smartrain.external_providers.runner import run_external_infer
from smartrain.external_model_ref import parse_external_model_ref, validate_external_model_ref
from smartrain.external_providers.registry import list_provider_specs
from smartrain.train_model_catalog import is_supported_external_provider_model, TrainModelCatalog
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.device_selector import default_device_value, discover_device_options, is_cuda_device
from smartrain.model_context import infer_img_size_from_model_context

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MANIFEST_NAME = "model_manifest.json"
DATA_MODES = ("folder", "dataset-split")


def build_inference_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Run object detection inference and save JSON report (empty call starts interactive mode)."
    )
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR}).")
    p.add_argument("--model-name", type=str, default=None, help="Promoted model directory name from workspace/models.")
    p.add_argument("--run", type=str, default=None, help="Run path or run index from workspace/runs list.")
    p.add_argument("--weights", type=str, default=None, help="Explicit model weights path (.pt/.onnx).")
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
    p.add_argument("--roi-pre-detect", action="store_true", help="Pre-detect ROI before inference (folder mode only).")
    p.add_argument("--roi-weights", type=str, default=None, help="ROI detector weights path (.pt/.onnx).")
    p.add_argument("--roi-conf", type=float, default=0.25, help="Confidence threshold for ROI detector.")
    p.add_argument("--roi-policy", choices=ROI_POLICIES, default="largest", help="ROI selection policy.")
    p.add_argument("--roi-pad-px", type=int, default=0, help="Padding in pixels around selected ROI.")
    p.add_argument("--roi-on-empty", choices=ON_EMPTY_MODES, default="full_image", help="Behavior when ROI detector has no detections.")
    p.add_argument("--roi-class-ids", type=str, default=None, help="CSV class ids for ROI detector (empty=all).")
    p.add_argument("--external-provider", type=str, default=None, help="External provider id for inference.")
    p.add_argument("--external-repo", type=str, default=None, help="Override external provider repository path.")
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
            if p.is_file() and p.suffix.lower() in {".pt", ".onnx"}
        )
        if not files:
            out.append((f"{d.name}/(no model files)", d.name, d.name))
            continue
        for fp in files:
            rel = fp.relative_to(root).as_posix()
            out.append((rel, rel, d.name))
    return out


def _resolve_model_from_name(layout: WorkspaceLayout, name: str) -> tuple[Path, str]:
    # Support direct file selection under models/ (used by interactive "models" mode).
    models_root = Path(layout.models).resolve()
    candidate_rel = Path(name)
    if candidate_rel.suffix.lower() in {".pt", ".onnx"} and not candidate_rel.is_absolute():
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
    preferred = sorted([p for p in mdir.glob("*.pt") if p.is_file()]) + sorted([p for p in mdir.glob("*.onnx") if p.is_file()])
    if preferred:
        return preferred[0].resolve(), name
    any_weight = sorted(p for p in mdir.rglob("*") if p.is_file() and p.suffix.lower() in {".pt", ".onnx"})
    if any_weight:
        return any_weight[0].resolve(), name
    raise FileNotFoundError(f"No .pt/.onnx model files found in: {mdir}")


def _resolve_model(args: argparse.Namespace, layout: WorkspaceLayout) -> tuple[Path, str, str]:
    if args.model_name:
        p, model_key = _resolve_model_from_name(layout, str(args.model_name).strip())
        return p, model_key, "models"
    if args.run:
        run_dir = _resolve_run_ref(layout, str(args.run))
        best = run_dir / "train" / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(f"best.pt not found in run: {run_dir}")
        return best.resolve(), run_dir.name, "runs"
    if args.weights:
        w = Path(str(args.weights)).expanduser()
        if not w.is_absolute():
            w = (Path(layout.root) / w).resolve()
        if not w.is_file():
            raise FileNotFoundError(f"Weights not found: {w}")
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
    roi_pred = roi_model.predict(source=image_path, conf=float(args.roi_conf), verbose=False)
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
    args.device = _prompt_inference_device(default=str(args.device or default_device_value()))
    args.half = prompt_yes_no("Use FP16 (--half)", default=bool(args.half))
    return True


def _prompt_inference_device(default: str = "cpu") -> str:
    options = discover_device_options()
    labels = [o.label for o in options]
    value_by_label = {o.label: o.value for o in options}
    default_value = default if any(o.value == default for o in options) else default_device_value()
    default_label = next((o.label for o in options if o.value == default_value), labels[0])
    print_numbered_options("inference devices", labels)
    picked = prompt_choice("Select inference device", labels, default=default_label, show_options=False)
    return value_by_label[picked]


def _ensure_device_available_or_exit(device: str | None) -> None:
    if not is_cuda_device(device):
        return
    try:
        import torch
    except Exception as exc:
        print(f"[ERROR] CUDA device requested ({device}), but torch import failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    if not torch.cuda.is_available():
        print(
            "[ERROR] CUDA device requested "
            f"({device}), but torch.cuda.is_available()=False. "
            f"torch={getattr(torch, '__version__', 'unknown')} "
            f"cuda_runtime={getattr(torch.version, 'cuda', 'unknown')}",
            file=sys.stderr,
        )
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
) -> dict[str, Any]:
    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
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
            "detections_total": sum(len(x.get("detections", [])) for x in image_rows),
        },
        "images": image_rows,
    }


def _write_report(path: str, report: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    parser = build_inference_arg_parser()
    args = parser.parse_args(argv)
    if args.device is None:
        args.device = default_device_value()

    try:
        workspace_root = resolve_workspace_root(args.workspace)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)
    layout = WorkspaceLayout(workspace_root)
    os.makedirs(os.path.join(layout.root, "inference"), exist_ok=True)
    interactive_allowed = is_interactive_allowed(argv)
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
        replay_cmd = build_non_interactive_command("inference", parser, args)
        print_replay_command("before launch", replay_cmd)
    else:
        _validate_non_interactive_args(parser, args)
    _ensure_device_available_or_exit(args.device)

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
        raise SystemExit(2)
    if parsed_weights_ref.is_external and parsed_weights_ref.provider_id and not getattr(args, "external_provider", None):
        args.external_provider = parsed_weights_ref.provider_id
        args.weights = parsed_weights_ref.model_ref
        print(f"[INFO] External provider inferred from --weights: {parsed_weights_ref.provider_id}")
    if parsed_model_name_ref.is_external and parsed_model_name_ref.provider_id and not getattr(args, "external_provider", None):
        args.external_provider = parsed_model_name_ref.provider_id
        args.model_name = None
        args.weights = parsed_model_name_ref.model_ref
        print(f"[INFO] External provider inferred from --model-name: {parsed_model_name_ref.provider_id}")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[ERROR] Failed to import ultralytics: {e}", file=sys.stderr)
        raise SystemExit(1)

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
            model_name = _sanitize_segment(model_path.name or raw_weight)
            model_source = "external-model"
    else:
        try:
            model_path, model_name, model_source = _resolve_model(args, layout)
        except Exception as e:
            print(f"[ERROR] Failed to resolve model: {e}", file=sys.stderr)
            raise SystemExit(1)
    if args.img_size is None:
        inferred = _infer_img_size_from_model_context(model_path) if isinstance(model_path, Path) else None
        args.img_size = int(inferred) if inferred is not None else 640

    if ext_provider:
        location = get_provider_location(ext_provider)
        if location is None and not getattr(args, "external_repo", None):
            print(
                f"[ERROR] External provider {ext_provider!r} is not installed. "
                "Use `smartrain providers install` or pass --external-repo.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        repo_path = str(getattr(args, "external_repo", "") or "").strip() or (location.repo_path if location else "")
        venv_path = location.venv_path if location else os.path.join(repo_path, "venv")
        if not venv_path:
            print(f"[ERROR] Missing venv for external provider {ext_provider!r}.", file=sys.stderr)
            raise SystemExit(1)
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
                raise SystemExit(2)
        source_for_external = _resolve_external_source(args, layout)
        source_short = (
            os.path.basename(os.path.abspath(os.path.expanduser(str(args.source_dir))).rstrip(os.sep)) or "folder"
            if args.data_mode == "folder"
            else f"{args.dataset}-{args.split}"
        )
        out_root = _resolve_output_root(layout, model_name, source_short)
        report_path = os.path.join(out_root, "inference_results.json")
        rc = run_external_infer(
            ext_provider,
            repo_path,
            venv_path,
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
            "source": _source_descriptor(args, source_for_external, source_short, layout),
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
            "images": [],
        }
        _write_report(report_path, external_report)
        print(f"[OK] External inference report: {report_path}")
        raise SystemExit(rc)
    model = YOLO(str(model_path))
    roi_model = None
    if args.roi_pre_detect:
        if args.data_mode != "folder":
            print("[ERROR] --roi-pre-detect is supported only for --data-mode folder.", file=sys.stderr)
            raise SystemExit(1)
        roi_w = args.roi_weights or str(model_path)
        roi_model = YOLO(str(roi_w))
        args.roi_weights = roi_w

    try:
        if args.data_mode == "folder":
            images = _collect_folder_images(str(args.source_dir), int(args.limit))
            source_abs = os.path.abspath(os.path.expanduser(str(args.source_dir)))
            source_short = os.path.basename(source_abs.rstrip(os.sep)) or "folder"
        else:
            images, split_dir = _collect_split_images_for_dataset(
                layout,
                str(args.dataset),
                str(args.split),
                int(args.limit),
            )
            source_abs = split_dir
            source_short = f"{args.dataset}-{args.split}"
    except Exception as e:
        print(f"[ERROR] Failed to resolve inference source: {e}", file=sys.stderr)
        raise SystemExit(1)
    if not images:
        print("[ERROR] No images found for inference.", file=sys.stderr)
        raise SystemExit(1)

    out_root = _resolve_output_root(layout, model_name, source_short)
    report_path = os.path.join(out_root, "inference_results.json")

    image_rows: list[dict[str, Any]] = []
    skipped = 0
    # Initialize output file early to make progress durable.
    _write_report(
        report_path,
        _build_report(
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
        ),
    )
    progress_desc = f"inference:{args.data_mode}"
    for image_path in tqdm(images, desc=progress_desc, unit="img"):
        image_path_abs = os.path.abspath(image_path)
        with Image.open(image_path_abs) as im:
            im_rgb = im.convert("RGB")
            iw, ih = im_rgb.size
            roi_box: tuple[int, int, int, int] | None = None
            src_for_predict: Any = image_path_abs
            if roi_model is not None:
                rb = _predict_roi_crop(roi_model, image_path_abs, args)
                if rb[0] < 0:
                    skipped += 1
                    _write_report(
                        report_path,
                        _build_report(
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
                        ),
                    )
                    continue
                roi_box = rb
                crop = im_rgb.crop((roi_box[0], roi_box[1], roi_box[2], roi_box[3]))
                src_for_predict = np.asarray(crop)
            elif args.data_mode == "folder":
                roi_box = (0, 0, iw, ih)

        preds = model.predict(
            source=src_for_predict,
            conf=float(args.conf),
            imgsz=int(args.img_size),
            verbose=False,
            device=str(args.device),
            half=bool(args.half),
        )
        boxes_payload: list[dict[str, Any]] = []
        if preds:
            r = preds[0]
            boxes_obj = getattr(r, "boxes", None)
            if boxes_obj is not None and len(boxes_obj) > 0:
                xyxy = boxes_obj.xyxy.cpu().numpy()
                cls = boxes_obj.cls.cpu().numpy()
                confs = boxes_obj.conf.cpu().numpy()
                names = getattr(model, "names", {})
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                    cls_idx = int(cls[i])
                    conf_v = float(confs[i])
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
                            "class_index": cls_idx,
                            "class_name": _class_name_from_names(names, cls_idx),
                            "confidence": conf_v,
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
        _write_report(
            report_path,
            _build_report(
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
            ),
        )

    print(f"[OK] Inference done: {len(image_rows)} images, skipped={skipped}")
    print(f"[OK] Report: {report_path}")
    if interactive_used:
        replay_cmd = build_non_interactive_command("inference", parser, args)
        print_replay_command("after execution", replay_cmd)


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
