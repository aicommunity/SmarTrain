#!/usr/bin/env python3
"""
Trimming the YOLO dataset by ROI from a pre-trained Ultralytics model (detect/segment).
Recalculates normalized detection and segmentation marks for a new frame.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from smartrain.cli_support.cli_argparse import CliArgumentParser
from smartrain.cli_support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.services.datasets.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.services.datasets.dataset_passport import write_dataset_passport
from smartrain.services.datasets.dataset_scan import find_yaml_file
from smartrain.services.datasets.datasets_json_scan_core_service import (
    IMAGE_EXTS_FLAT,
    load_yaml,
)
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.ultralytics_ephemeral import best_effort_prune_workspace_runs_detect, ultralytics_sidecar_dir
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout

sys.stdout.reconfigure(encoding="utf-8")

DATASETS_INFO_FILE = "datasets_info.json"

ROI_POLICIES = ("union", "largest", "best_conf", "per_box")
ON_EMPTY_MODES = ("full_image", "skip", "fail")
MODES = ("yolo_detect", "yolo_segment")


def _is_image_file(name: str) -> bool:
    low = name.lower()
    return any(low.endswith(ext) for ext in IMAGE_EXTS_FLAT)


def _parse_class_ids(raw: Any) -> Optional[List[int]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [int(x) for x in raw]
    raise ValueError("class_ids in roi_auto must be null or a list of ints")


def _bbox_pixels_from_yolo_norm(
    cx: float, cy: float, w: float, h: float, iw: int, ih: int
) -> Tuple[float, float, float, float]:
    x1 = (cx - w / 2.0) * iw
    x2 = (cx + w / 2.0) * iw
    y1 = (cy - h / 2.0) * ih
    y2 = (cy + h / 2.0) * ih
    return x1, y1, x2, y2


def _intersect_bbox(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> Optional[Tuple[float, float, float, float]]:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _clamp_crop(
    x1: float, y1: float, x2: float, y2: float, pad: int, iw: int, ih: int
) -> Tuple[int, int, int, int]:
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad
    x1c = max(0, int(round(x1)))
    y1c = max(0, int(round(y1)))
    x2c = min(iw, int(round(x2)))
    y2c = min(ih, int(round(y2)))
    if x2c <= x1c:
        x2c = min(iw, x1c + 1)
    if y2c <= y1c:
        y2c = min(ih, y1c + 1)
    return x1c, y1c, x2c, y2c


def _bbox_to_yolo_line(
    x1: float, y1: float, x2: float, y2: float, cw: int, ch: int
) -> Optional[str]:
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return f"{cx / cw:.6f} {cy / ch:.6f} {w / cw:.6f} {h / ch:.6f}"


def _transform_detection_line(
    parts: List[str],
    crop: Tuple[int, int, int, int],
    iw: int,
    ih: int,
) -> Optional[str]:
    cls = int(float(parts[0]))
    cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    x1, y1, x2, y2 = _bbox_pixels_from_yolo_norm(cx, cy, w, h, iw, ih)
    cx0, cy0, cx1, cy1 = crop
    cw = cx1 - cx0
    ch = cy1 - cy0
    inter = _intersect_bbox((x1, y1, x2, y2), (cx0, cy0, cx1, cy1))
    if inter is None:
        return None
    ix1, iy1, ix2, iy2 = inter
    nx1, ny1 = ix1 - cx0, iy1 - cy0
    nx2, ny2 = ix2 - cx0, iy2 - cy0
    body = _bbox_to_yolo_line(nx1, ny1, nx2, ny2, cw, ch)
    if body is None:
        return None
    return f"{cls} {body}"


def _transform_segment_line(
    parts: List[str],
    crop: Tuple[int, int, int, int],
    iw: int,
    ih: int,
) -> Optional[str]:
    cls = int(float(parts[0]))
    nums = [float(x) for x in parts[1:]]
    if len(nums) < 6 or len(nums) % 2 != 0:
        return None
    cx0, cy0, cx1, cy1 = crop
    cw = cx1 - cx0
    ch = cy1 - cy0
    pts: List[Tuple[float, float]] = []
    for i in range(0, len(nums), 2):
        px = nums[i] * iw - cx0
        py = nums[i + 1] * ih - cy0
        pts.append((px, py))
    min_x = min(p[0] for p in pts)
    max_x = max(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    max_y = max(p[1] for p in pts)
    if max_x < 0 or max_y < 0 or min_x > cw or min_y > ch:
        return None
    out_coords: List[str] = []
    for px, py in pts:
        px = max(0.0, min(float(cw), px))
        py = max(0.0, min(float(ch), py))
        out_coords.append(f"{px / cw:.6f}")
        out_coords.append(f"{py / ch:.6f}")
    return f"{cls} " + " ".join(out_coords)


def _transform_label_line(
    line: str,
    crop: Tuple[int, int, int, int],
    iw: int,
    ih: int,
) -> Optional[str]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 5:
        return None
    if len(parts) == 5:
        return _transform_detection_line(parts, crop, iw, ih)
    n_rest = len(parts) - 1
    if n_rest % 2 == 0 and n_rest >= 6:
        return _transform_segment_line(parts, crop, iw, ih)
    return None


def _read_label_lines(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def _write_label_lines(path: str, lines: List[str]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _select_roi_boxes(
    xyxy: Any,
    cls: Any,
    confs: Any,
    class_ids: Optional[Sequence[int]],
    policy: str,
    iw: int,
    ih: int,
) -> List[Tuple[float, float, float, float]]:
    if xyxy is None or len(xyxy) == 0:
        return []
    xyxy = np.asarray(xyxy, dtype=np.float64)
    cls = np.asarray(cls, dtype=np.int64).reshape(-1)
    confs = np.asarray(confs, dtype=np.float64).reshape(-1)
    boxes: List[Tuple[float, float, float, float, int, float]] = []
    for i in range(xyxy.shape[0]):
        c = int(cls[i])
        if class_ids is not None and c not in class_ids:
            continue
        x1, y1, x2, y2 = xyxy[i]
        x1 = max(0, min(iw, x1))
        x2 = max(0, min(iw, x2))
        y1 = max(0, min(ih, y1))
        y2 = max(0, min(ih, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((float(x1), float(y1), float(x2), float(y2), c, float(confs[i])))

    if not boxes:
        return []

    if policy == "per_box":
        boxes.sort(key=lambda t: -t[5])
        return [(b[0], b[1], b[2], b[3]) for b in boxes]

    if policy == "largest":
        one = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return [(one[0], one[1], one[2], one[3])]

    if policy == "best_conf":
        one = max(boxes, key=lambda b: b[5])
        return [(one[0], one[1], one[2], one[3])]

    # union
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [(x1, y1, x2, y2)]


def _full_image_crop(iw: int, ih: int) -> Tuple[float, float, float, float]:
    return 0.0, 0.0, float(iw), float(ih)


def _copy_and_patch_yaml(dataset_root: str, output_root: str) -> None:
    ypath = find_yaml_file(dataset_root)
    if not ypath:
        return
    rel = os.path.relpath(ypath, dataset_root)
    data = load_yaml(ypath)
    if not data or not isinstance(data, dict):
        return
    data = dict(data)
    data.pop("path", None)
    dest = os.path.join(output_root, rel)
    dest_dir = os.path.dirname(dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def _infer_structure_from_dataset_root(dataset_root: str) -> str:
    """
    Infer dataset structure when there is no datasets_info.json.
    Returns one of: 'split' or 'flat'.
    """
    root = os.path.abspath(os.path.expanduser(dataset_root))
    # split layout
    if any(
        os.path.isdir(os.path.join(root, s, "images")) or os.path.isdir(os.path.join(root, s, "labels"))
        for s in ("train", "val", "test")
    ):
        return "split"
    # flat layout
    if os.path.isdir(os.path.join(root, "images")) and os.path.isdir(os.path.join(root, "labels")):
        return "flat"
    # fallback: treat as flat (iter_image_label_buckets will validate)
    return "flat"


def _infer_classes_from_data_yaml(dataset_root: str) -> dict[str, int]:
    y = find_yaml_file(dataset_root)
    if not y:
        return {}
    data = load_yaml(y)
    if not isinstance(data, dict):
        return {}
    names = data.get("names")
    if isinstance(names, list):
        return {str(v): int(i) for i, v in enumerate(names)}
    if isinstance(names, dict):
        out: dict[str, int] = {}
        for k, v in names.items():
            try:
                out[str(v)] = int(k)
            except Exception:
                continue
        # normalize to 0..N-1 if keys are not contiguous is out of scope here
        return out
    return {}


def _iter_images_only(dataset_root: str) -> list[tuple[str, str]]:
    """
    All images under dataset_root (recursive).
    Returns (abs_image_path, rel_dir_under_root) where rel_dir_under_root == '' for root.
    """
    root = os.path.abspath(os.path.expanduser(dataset_root))
    out: list[tuple[str, str]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""
        for fn in sorted(filenames):
            if _is_image_file(fn):
                out.append((os.path.join(dirpath, fn), rel_dir))
    return out

def _ensure_data_yaml_after_roi(output_root: str, entry: Dict[str, Any]) -> None:
    """
    If the source didn't have data.yaml (often cvat11/zip), scan skips the ROI output.
    We write a minimal flat-YOLO yaml from the classes field of the datasets_info record.
    """
    if find_yaml_file(output_root):
        return
    classes = entry.get("classes")
    names_list: List[str] = []
    if isinstance(classes, dict) and classes:
        pairs = sorted(classes.items(), key=lambda kv: int(kv[1]))
        names_list = [str(k) for k, _ in pairs]
    nc = len(names_list)
    blob: Dict[str, Any] = {
        "train": "images",
        "val": "images",
        "test": "images",
        "nc": nc,
        "names": names_list,
    }
    dest = os.path.join(output_root, "data.yaml")
    os.makedirs(output_root, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(blob, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def build_roi_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Cropping the YOLO dataset by ROI (Ultralytics YOLO detect/segment)"
    )
    p.add_argument("--dataset-name", required=False, default=None, help="Dataset key in datasets_info.json")
    p.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Input dataset name (can be repeated).",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="CSV list of input datasets (for example ds1,ds2).",
    )
    p.add_argument(
        "--workspace",
        default=None,
        help=f"Workspace root (aka {WORKSPACE_ENV_VAR}); main mode without --source-path",
    )
    p.add_argument(
        "--source-path",
        default=None,
        help="Legacy: parent directory, dataset in {source-path}/{dataset-name}/",
    )
    p.add_argument(
        "--output-path",
        default=None,
        help="The root of the new dataset (by default in workspace: datasets/<dataset-name>_roi)",
    )
    p.add_argument(
        "--datasets-info-path",
        default=None,
        help="The file datasets_info.json or the directory with it (default: datasets/ in workspace or --source-path)",
    )
    p.add_argument(
        "--tmp-dir",
        default=None,
        help="Directory for temporary cvat11 tags (default: <workspace>/tmp)",
    )
    p.add_argument("--weights", default=None, help="Override weights from roi_auto")
    p.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    p.add_argument("--pad-px", type=int, default=None, help="Crop indentation on each side")
    p.add_argument(
        "--roi-policy",
        choices=ROI_POLICIES,
        default=None,
        help="ROI policy (default largest, if not set in roi_auto): union | largest | best_conf | per_box",
    )
    p.add_argument("--mode", choices=MODES, default=None, help="yolo_detect | yolo_segment")
    p.add_argument(
        "--on-empty",
        choices=ON_EMPTY_MODES,
        default="full_image",
        help="Behavior in the absence of detections for ROI",
    )
    p.add_argument(
        "--require-roi-auto",
        action="store_true",
        help="Require roi_auto block in datasets_info (otherwise --weights, etc. is sufficient)",
    )
    p.add_argument(
        "--images-only",
        action="store_true",
        help=(
            "Allow input without labels (images-only). ROI is applied to images and empty label files "
            "are created in the output."
        ),
    )
    return p


def parse_args(argv=None) -> argparse.Namespace:
    return build_roi_arg_parser().parse_args(argv)


def _parse_selected_datasets(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    single = (args.dataset_name or "").strip()
    if single:
        out.append(single)
    if args.dataset:
        for item in args.dataset:
            name = str(item).strip()
            if name:
                out.append(name)
    if args.datasets:
        for part in str(args.datasets).split(","):
            name = part.strip()
            if name:
                out.append(name)
    uniq: list[str] = []
    seen: set[str] = set()
    for name in out:
        if name not in seen:
            seen.add(name)
            uniq.append(name)
    return uniq


def _prompt_input(label: str, default: str = "", completer=None, show_default_hint: bool = True) -> str:
    from prompt_toolkit import prompt

    prompt_label = f"{label} [default: {default}]: " if (default != "" and show_default_hint) else label
    value = str(prompt(prompt_label, default="", completer=completer, complete_while_typing=True)).strip()
    if value:
        return value
    if default != "":
        if sys.stdin.isatty():
            try:
                sys.stdout.write("\x1b[1A\r")
                sys.stdout.write(f"{prompt_label}{default}\n")
                sys.stdout.flush()
            except Exception:
                print(default)
        else:
            print(default)
    return str(default)


def _prompt_yes_no(label: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    default_text = "y" if default else "n"
    raw = _prompt_input(f"{label} [{suffix}]: ", default=default_text, show_default_hint=False).strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true", "yes", "d")


def _list_workspace_detector_models(workspace_root: str) -> list[str]:
    exts = {".pt", ".onnx"}
    root = Path(workspace_root)
    if not root.is_dir():
        return []
    out: list[str] = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p.name)
    return out


def _run_interactive_roi_setup(args: argparse.Namespace) -> bool:
    from prompt_toolkit.completion import WordCompleter

    print("[INFO] Interactive roi mode (Enter = default).")
    ws = (args.workspace or "").strip() or (os.environ.get(WORKSPACE_ENV_VAR) or "").strip()
    if not ws:
        ws = _prompt_input("Workspace path: ", default=os.getcwd()).strip()
        if not ws:
            print("[ERROR] Workspace not set.")
            return False
        args.workspace = ws
    layout = WorkspaceLayout(os.path.abspath(os.path.expanduser(ws)))
    info_path = layout.work_datasets_info_path()
    if not os.path.isfile(info_path):
        print(f"[ERROR] {info_path} not found")
        return False
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read {info_path}: {e}")
        return False
    if not isinstance(catalog, dict) or not catalog:
        print("[ERROR] There are no available datasets in datasets_info.json.")
        return False
    names = sorted(str(k) for k in catalog.keys())
    from smartrain.cli_support.cli_prompts import prompt_multi_choice_csv

    print("[INFO] Datasets for ROI: CSV of numbers or names (empty = first in list).")
    picked = prompt_multi_choice_csv(
        "Input datasets for ROI",
        names,
        default_values=[names[0]] if names else [],
    )
    args.dataset = picked
    args.dataset_name = picked[0]
    default_out = str(args.output_path or (layout.datasets if len(args.dataset) > 1 else os.path.join(layout.datasets, f"{args.dataset_name}_roi")))
    out_raw = _prompt_input(
        "Output directory (--output-path)",
        default=default_out,
    ).strip()
    args.output_path = out_raw or default_out
    mode_comp = WordCompleter(list(MODES), ignore_case=True)
    mode_val = _prompt_input(
        "Mode (--mode)",
        default=str(args.mode or ""),
        completer=mode_comp,
    ).strip()
    args.mode = mode_val or None
    rp_comp = WordCompleter(list(ROI_POLICIES), ignore_case=True)
    rp_val = _prompt_input(
        "ROI policy (--roi-policy)",
        default=str(args.roi_policy or ""),
        completer=rp_comp,
    ).strip()
    args.roi_policy = rp_val or None
    oe_comp = WordCompleter(list(ON_EMPTY_MODES), ignore_case=True)
    oe_val = _prompt_input(
        "On empty (--on-empty)",
        default=str(args.on_empty or "full_image"),
        completer=oe_comp,
    ).strip()
    args.on_empty = oe_val or "full_image"
    models = _list_workspace_detector_models(layout.root)
    if models:
        print("[INFO] ROI detectors in the workspace root:")
        for m in models:
            print(f"  - {m}")
    weights_completer = WordCompleter(models, ignore_case=True) if models else None
    w_raw = _prompt_input(
        "Weights (--weights)",
        default=str(args.weights or ""),
        completer=weights_completer,
    ).strip()
    args.weights = w_raw or None
    c_raw = _prompt_input(
        "Confidence (--conf)",
        default="" if args.conf is None else str(args.conf),
    ).strip()
    args.conf = float(c_raw) if c_raw else args.conf
    p_raw = _prompt_input(
        "Pad px (--pad-px)",
        default="" if args.pad_px is None else str(args.pad_px),
    ).strip()
    args.pad_px = int(p_raw) if p_raw else args.pad_px
    req_default = bool(getattr(args, "require_roi_auto", False))
    args.require_roi_auto = _prompt_yes_no(
        "Require roi_auto (--require-roi-auto)",
        default=req_default,
    )
    return True


def _resolve_catalog_dataset_key(catalog: Dict[str, Any], requested: str) -> str:
    """
    The key in datasets_info for zip in datasets is set without the .zip suffix
    (as in scan). We also accept the full name of the archive file.
    """
    if requested in catalog:
        return requested
    if requested.lower().endswith(".zip"):
        stem = requested[:-4]
        if stem in catalog:
            return stem
    sys.exit(
        f"[ERROR] The key {requested!r} is not in the directory."
        " Usually in datasets_info the name matches the archive name without '.zip', "
        "for example 'job_2512_dataset_2026_03_27_15_31_43_cvat_for_images_1_1'."
    )


def _resolve_datasets_info_file(datasets_info_path_arg: Optional[str], default_json_path: str) -> str:
    if not datasets_info_path_arg or not str(datasets_info_path_arg).strip():
        return default_json_path
    p = os.path.abspath(os.path.expanduser(str(datasets_info_path_arg).strip()))
    if os.path.isfile(p):
        return p
    return os.path.join(p, DATASETS_INFO_FILE)


def _relpath_under_dataset(path: str, dataset_root: str, fallback: str) -> str:
    """Relative path for mirroring the structure to output; if outside dataset_root - fallback."""
    ap = os.path.abspath(path)
    ar = os.path.abspath(dataset_root)
    try:
        rel = os.path.relpath(ap, ar)
    except ValueError:
        return fallback
    if rel == ".." or rel.startswith(".." + os.sep):
        return fallback
    return rel


def _validate_layout_legacy(
    source_path: str, dataset_name: str, datasets_info_path_override: Optional[str]
) -> str:
    dr = os.path.join(source_path, dataset_name)
    if not os.path.isdir(dr):
        sys.exit(f"[ERROR] No dataset directory: {dr}")
    if datasets_info_path_override is not None:
        alt = os.path.join(datasets_info_path_override, dataset_name)
        if not os.path.isdir(alt):
            sys.exit(f"[ERROR] If --datasets-info-path is expected, directory {alt}")
        if os.path.realpath(dr) != os.path.realpath(alt):
            sys.exit(
                "[ERROR] --source-path and --datasets-info-path must point to the same dataset"
                f"({dr} vs {alt})"
            )
    return dr


def _load_roi_config(args: argparse.Namespace, entry: Dict[str, Any]) -> Dict[str, Any]:
    ra = entry.get("roi_auto")
    if args.require_roi_auto and not isinstance(ra, dict):
        sys.exit("[ERROR] There is no roi_auto object in the dataset record (--require-roi-auto).")

    ra = ra if isinstance(ra, dict) else {}
    weights = args.weights or ra.get("weights")
    if not weights:
        sys.exit("[ERROR] Specify weights in roi_auto or via --weights.")

    conf = args.conf if args.conf is not None else float(ra.get("conf", 0.25))
    pad_px = args.pad_px if args.pad_px is not None else int(ra.get("pad_px", 0))
    roi_policy = args.roi_policy or ra.get("roi_policy", "largest")
    mode = args.mode or ra.get("mode", "yolo_detect")

    if roi_policy not in ROI_POLICIES:
        sys.exit(f"[ERROR] Invalid roi_policy: {roi_policy}")
    if mode not in MODES:
        sys.exit(f"[ERROR] Invalid mode: {mode}")

    try:
        class_ids = _parse_class_ids(ra.get("class_ids"))
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")

    return {
        "weights": weights,
        "conf": conf,
        "pad_px": pad_px,
        "roi_policy": roi_policy,
        "mode": mode,
        "class_ids": class_ids,
    }


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_roi_arg_parser()
    args = parser.parse_args(argv)
    interactive_allowed = is_interactive_allowed(argv)
    interactive_used = False
    selected_dataset_names = _parse_selected_datasets(args)
    legacy_source = (args.source_path or "").strip()
    direct_legacy_single = False
    if legacy_source and (args.output_path or "").strip() and not selected_dataset_names and not (args.datasets_info_path or "").strip():
        # Direct legacy mode: treat --source-path as dataset root (no datasets_info.json required).
        direct_legacy_single = True
        selected_dataset_names = [Path(os.path.abspath(os.path.expanduser(legacy_source))).name]
    if not selected_dataset_names:
        if not interactive_allowed:
            sys.exit("[ERROR] Incomplete arguments: specify --dataset-name/--dataset/--datasets.")
        if not sys.stdin.isatty():
            sys.exit(
                "[ERROR] Interactive roi mode requires a terminal (TTY). "
                "Run without arguments in TTY or pass complete flags."
            )
        if not _run_interactive_roi_setup(args):
            return
        interactive_used = True
        selected_dataset_names = _parse_selected_datasets(args)
    replay_cmd = None
    if interactive_used:
        replay_cmd = build_non_interactive_command("roi", parser, args)
        print_replay_command("before launch", replay_cmd)
    workspace_root: Optional[str] = None
    layout: Optional[WorkspaceLayout] = None

    need_default_roi_output = False
    if legacy_source:
        if len(selected_dataset_names) > 1:
            sys.exit("[ERROR] Legacy mode does not support processing multiple datasets in one run.")
        if not args.output_path or not str(args.output_path).strip():
            sys.exit("[ERROR] In legacy mode (--source-path), specify --output-path")
        source_path_abs = os.path.abspath(os.path.expanduser(legacy_source))
        if direct_legacy_single:
            info_path = None
            temp_root = os.path.join(source_path_abs, "tmp")
        else:
            info_path = _resolve_datasets_info_file(
                args.datasets_info_path, os.path.join(source_path_abs, DATASETS_INFO_FILE)
            )
            temp_root = os.path.join(source_path_abs, "tmp")
        os.makedirs(temp_root, exist_ok=True)
    else:
        ws = (args.workspace or "").strip() or (os.environ.get(WORKSPACE_ENV_VAR) or "").strip()
        if not ws:
            sys.exit(
                f"[ERROR] Specify --source-path (legacy) or --workspace / variable {WORKSPACE_ENV_VAR}"
            )
        workspace_root = os.path.abspath(os.path.expanduser(ws))
        layout = WorkspaceLayout(workspace_root)
        info_path = _resolve_datasets_info_file(
            args.datasets_info_path, layout.work_datasets_info_path()
        )
        if args.tmp_dir and str(args.tmp_dir).strip():
            temp_root = os.path.abspath(os.path.expanduser(str(args.tmp_dir).strip()))
        else:
            temp_root = os.path.join(workspace_root, "tmp")
        os.makedirs(temp_root, exist_ok=True)
        if not (args.output_path or "").strip():
            need_default_roi_output = True

    if direct_legacy_single:
        datasets_info = {
            selected_dataset_names[0]: {
                "structure": _infer_structure_from_dataset_root(source_path_abs),
                "classes": _infer_classes_from_data_yaml(source_path_abs),
                "roi_auto": {},
            }
        }
    else:
        assert info_path is not None
        if not os.path.isfile(info_path):
            sys.exit(f"[ERROR] {info_path} not found")
        with open(info_path, "r", encoding="utf-8") as f:
            datasets_info = json.load(f)

    resolved_keys: list[str] = []
    for req_name in selected_dataset_names:
        key = _resolve_catalog_dataset_key(datasets_info, req_name)
        if key not in resolved_keys:
            resolved_keys.append(key)
        if key != req_name:
            print(f"[INFO] The dataset key in the directory is used: {key!r}")

    output_base = os.path.abspath(os.path.expanduser(str(args.output_path).strip())) if (args.output_path or "").strip() else None
    for dataset_key in resolved_keys:
        entry = datasets_info[dataset_key]
        structure = entry.get("structure")
        if not structure:
            sys.exit(f"[ERROR] There is no structure field in the dataset record {dataset_key!r}")
        if legacy_source:
            source_path_abs = os.path.abspath(os.path.expanduser(legacy_source))
            if direct_legacy_single:
                dataset_root = source_path_abs
            else:
                dataset_root = _validate_layout_legacy(
                    source_path_abs, dataset_key, args.datasets_info_path
                )
            output_root = os.path.abspath(os.path.expanduser(str(args.output_path).strip()))
        else:
            assert workspace_root is not None and layout is not None
            dataset_root = resolve_dataset_root_for_entry(
                dataset_key,
                entry,
                workspace_root=workspace_root,
                source_catalog_dir=layout.datasets,
                legacy_source_parent=layout.datasets,
            )
            if len(resolved_keys) == 1:
                output_root = output_base or os.path.join(layout.datasets, f"{dataset_key}_roi")
            else:
                base_dir = output_base or layout.datasets
                output_root = os.path.join(base_dir, f"{dataset_key}_roi")
        cfg = _load_roi_config(args, entry)
        os.makedirs(output_root, exist_ok=True)

        model = YOLO(cfg["weights"])
        pred_proj = ultralytics_sidecar_dir(
            workspace_root if workspace_root is not None else temp_root,
            ".ultralytics_roi_predict",
        )
        if cfg["mode"] == "yolo_segment" and getattr(model, "task", None) != "segment":
            print(
                f"[WARNING] mode=yolo_segment, but model task={getattr(model, 'task', None)}; "
                "detect/mask bounding boxes are used."
            )

        buckets = iter_image_label_buckets(
            dataset_root,
            structure,
            entry,
            dataset_name=dataset_key,
            temp_root=temp_root,
            exclude_test=False,
        )
        images_only_items: list[tuple[str, str]] = []
        if args.images_only:
            if len(resolved_keys) > 1:
                sys.exit("[ERROR] --images-only does not support processing multiple datasets in one run.")
            images_only_items = _iter_images_only(dataset_root)
            if not images_only_items:
                sys.exit("[ERROR] No images found for images-only mode.")
        if not buckets:
            # Auto-fallback for direct legacy single-dataset runs (no datasets_info.json).
            images_only_allowed = bool(args.images_only) or bool(direct_legacy_single)
            if images_only_allowed:
                images_only_items = _iter_images_only(dataset_root)
            if not images_only_items:
                sys.exit(f"[ERROR] No images/labels pairs for structure={structure}")
            print(
                "[WARNING] No labels found; running in images-only mode. "
                "Only images will be saved to the output directory."
            )

        stats = {"images": 0, "skipped": 0}

        def _process_one_image(
            *,
            src_img: str,
            out_img_dir: str,
            out_lbl_dir: Optional[str],
            label_lines: List[str],
            write_labels: bool,
            stem_prefix: str,
        ) -> None:
            fname = os.path.basename(src_img)
            stem, _ext = os.path.splitext(fname)
            base_stem = f"{stem_prefix}{stem}" if stem_prefix else stem

            with Image.open(src_img) as im:
                im = im.convert("RGB")
                iw, ih = im.size

            results = model.predict(
                source=src_img,
                conf=cfg["conf"],
                verbose=False,
                save=False,
                project=pred_proj,
                name="dataset-roi",
                exist_ok=True,
            )
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                xyxy, cls, confs = [], [], []
            else:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()

            roi_list = _select_roi_boxes(
                xyxy, cls, confs, cfg["class_ids"], cfg["roi_policy"], iw, ih
            )

            if not roi_list:
                if args.on_empty == "fail":
                    sys.exit(f"[ERROR] No detections for ROI: {src_img}")
                if args.on_empty == "skip":
                    stats["skipped"] += 1
                    return
                roi_list = [_full_image_crop(iw, ih)]

            def process_one_crop(
                crop_unpadded: Tuple[float, float, float, float],
                out_stem: str,
            ) -> None:
                x1, y1, x2, y2 = crop_unpadded
                crop = _clamp_crop(x1, y1, x2, y2, cfg["pad_px"], iw, ih)
                cx0, cy0, cx1, cy1 = crop
                with Image.open(src_img) as im2:
                    im2 = im2.convert("RGB")
                    cropped = im2.crop((cx0, cy0, cx1, cy1))
                out_image_path = os.path.join(out_img_dir, out_stem + os.path.splitext(fname)[1])
                os.makedirs(out_img_dir, exist_ok=True)
                cropped.save(out_image_path)

                if write_labels:
                    if not out_lbl_dir:
                        sys.exit("[ERROR] Internal: out_lbl_dir is required when write_labels=True.")
                    out_lines: List[str] = []
                    for raw in label_lines:
                        new_l = _transform_label_line(raw, crop, iw, ih)
                        if new_l is not None:
                            out_lines.append(new_l + "\n")
                    out_lbl_path = os.path.join(out_lbl_dir, out_stem + ".txt")
                    _write_label_lines(out_lbl_path, out_lines)

            if cfg["roi_policy"] == "per_box":
                for idx, box in enumerate(roi_list, start=1):
                    process_one_crop(box, f"{base_stem}_split_{idx}")
                stats["images"] += len(roi_list)
            else:
                process_one_crop(roi_list[0], base_stem)
                stats["images"] += 1

        if images_only_items:
            for src_img, rel_dir in tqdm(images_only_items, desc=f"{dataset_key}:images-only"):
                # In images-only mode we write ONLY images directly into output_root,
                # with no extra folders, labels or data.yaml.
                stem_prefix = ""
                if rel_dir:
                    stem_prefix = rel_dir.replace(os.sep, "__").strip("_") + "__"
                out_img_dir = output_root
                _process_one_image(
                    src_img=src_img,
                    out_img_dir=out_img_dir,
                    out_lbl_dir=None,
                    label_lines=[],
                    write_labels=False,
                    stem_prefix=stem_prefix,
                )
        else:
            for img_dir, lbl_dir in buckets:
                rel_base = _relpath_under_dataset(img_dir, dataset_root, "images")
                out_img_dir = os.path.join(output_root, rel_base)
                rel_lbl = _relpath_under_dataset(lbl_dir, dataset_root, "labels")
                out_lbl_dir = os.path.join(output_root, rel_lbl)

                files = [f for f in sorted(os.listdir(img_dir)) if _is_image_file(f)]
                for fname in tqdm(files, desc=f"{dataset_key}:{rel_base}"):
                    src_img = os.path.join(img_dir, fname)
                    stem, _ext = os.path.splitext(fname)
                    src_lbl = os.path.join(lbl_dir, stem + ".txt")
                    label_lines = _read_label_lines(src_lbl)
                    _process_one_image(
                        src_img=src_img,
                        out_img_dir=out_img_dir,
                        out_lbl_dir=out_lbl_dir,
                        label_lines=label_lines,
                        write_labels=True,
                        stem_prefix="",
                    )

        if not images_only_items:
            _copy_and_patch_yaml(dataset_root, output_root)
            _ensure_data_yaml_after_roi(output_root, entry)
        try:
            passport_path = write_dataset_passport(
                output_dataset_dir=output_root,
                command="roi",
                source_datasets=[
                    {
                        "name": dataset_key,
                        "path": dataset_root,
                        "dataset_hash": entry.get("dataset_hash") if isinstance(entry, dict) else None,
                    }
                ],
                parameters=vars(args),
                workspace_root=layout.root if layout is not None else None,
                transformations=[
                    {
                        "mode": cfg["mode"],
                        "roi_policy": cfg["roi_policy"],
                        "class_ids": cfg["class_ids"] or [],
                        "conf": cfg["conf"],
                        "pad_px": cfg["pad_px"],
                        "on_empty": args.on_empty,
                    }
                ],
                random_seed=None,
                stats_before={},
                stats_after={"output_images": stats["images"], "skipped": stats["skipped"]},
            )
            print(f"[OK] Passport: {passport_path}")
        except Exception as e:
            print(f"[WARNING] Failed to write dataset_passport.json: {e}")
        print(
            f"[OK] Done for {dataset_key!r}: {stats['images']} output frames, "
            f"skipped {stats['skipped']}, directory: {output_root}"
        )
    if replay_cmd:
        print_replay_command("after execution", replay_cmd)
    if workspace_root:
        best_effort_prune_workspace_runs_detect(workspace_root)


if __name__ == "__main__":
    main()
