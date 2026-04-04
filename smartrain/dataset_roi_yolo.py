#!/usr/bin/env python3
"""
Обрезка датасета YOLO по ROI из предобученной модели Ultralytics (detect/segment).
Пересчитывает нормализованные метки детекции и сегментации под новый кадр.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from smartrain.cli_argparse import CliArgumentParser
from smartrain.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.datasets_json_former import (
    IMAGE_EXTS_FLAT,
    find_yaml_file,
    load_yaml,
)
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout

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
    raise ValueError("class_ids в roi_auto должен быть null или списком int")


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
    data["path"] = os.path.abspath(output_root)
    dest = os.path.join(output_root, rel)
    dest_dir = os.path.dirname(dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def _ensure_data_yaml_after_roi(output_root: str, entry: Dict[str, Any]) -> None:
    """
    Если в источнике не было data.yaml (часто cvat11 / zip), scan пропускает выход ROI.
    Пишем минимальный flat-YOLO yaml из поля classes записи datasets_info.
    """
    if find_yaml_file(output_root):
        return
    classes = entry.get("classes")
    names_list: List[str] = []
    if isinstance(classes, dict) and classes:
        pairs = sorted(classes.items(), key=lambda kv: int(kv[1]))
        names_list = [str(k) for k, _ in pairs]
    nc = len(names_list)
    root_abs = os.path.abspath(output_root)
    blob: Dict[str, Any] = {
        "path": root_abs,
        "train": "images",
        "val": "images",
        "nc": nc,
        "names": names_list,
    }
    dest = os.path.join(output_root, "data.yaml")
    os.makedirs(output_root, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(blob, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def build_roi_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Кроп датасета YOLO по ROI (Ultralytics YOLO detect/segment)"
    )
    p.add_argument("--dataset-name", required=True, help="Ключ датасета в datasets_info.json")
    p.add_argument(
        "--workspace",
        default=None,
        help=f"Корень workspace (иначе {WORKSPACE_ENV_VAR}); основной режим без --source-path",
    )
    p.add_argument(
        "--source-path",
        default=None,
        help="Legacy: родительский каталог, датасет в {source-path}/{dataset-name}/",
    )
    p.add_argument(
        "--output-path",
        default=None,
        help="Корень нового датасета (по умолчанию в workspace: source_datasets/<dataset-name>_roi)",
    )
    p.add_argument(
        "--datasets-info-path",
        default=None,
        help="Файл datasets_info.json или каталог с ним (по умолчанию: source_datasets/ в workspace или --source-path)",
    )
    p.add_argument(
        "--tmp-dir",
        default=None,
        help="Каталог для временных cvat11-меток (по умолчанию: <workspace>/tmp)",
    )
    p.add_argument("--weights", default=None, help="Переопределить weights из roi_auto")
    p.add_argument("--conf", type=float, default=None, help="Порог confidence")
    p.add_argument("--pad-px", type=int, default=None, help="Отступ кропа с каждой стороны")
    p.add_argument(
        "--roi-policy",
        choices=ROI_POLICIES,
        default=None,
        help="Политика ROI (по умолчанию largest, если не задано в roi_auto): union | largest | best_conf | per_box",
    )
    p.add_argument("--mode", choices=MODES, default=None, help="yolo_detect | yolo_segment")
    p.add_argument(
        "--on-empty",
        choices=ON_EMPTY_MODES,
        default="full_image",
        help="Поведение при отсутствии детекций для ROI",
    )
    p.add_argument(
        "--require-roi-auto",
        action="store_true",
        help="Требовать блок roi_auto в datasets_info (иначе достаточно --weights и т.д.)",
    )
    return p


def parse_args(argv=None) -> argparse.Namespace:
    return build_roi_arg_parser().parse_args(argv)


def _resolve_catalog_dataset_key(catalog: Dict[str, Any], requested: str) -> str:
    """
    Ключ в datasets_info для zip в source_datasets задаётся без суффикса .zip
    (как в scan). Принимаем и полное имя файла архива.
    """
    if requested in catalog:
        return requested
    if requested.lower().endswith(".zip"):
        stem = requested[:-4]
        if stem in catalog:
            return stem
    sys.exit(
        f"[ERROR] Ключ {requested!r} отсутствует в каталоге. "
        "Обычно в datasets_info имя совпадает с именем архива без «.zip», "
        "например «job_2512_dataset_2026_03_27_15_31_43_cvat for images 1.1»."
    )


def _resolve_datasets_info_file(datasets_info_path_arg: Optional[str], default_json_path: str) -> str:
    if not datasets_info_path_arg or not str(datasets_info_path_arg).strip():
        return default_json_path
    p = os.path.abspath(os.path.expanduser(str(datasets_info_path_arg).strip()))
    if os.path.isfile(p):
        return p
    return os.path.join(p, DATASETS_INFO_FILE)


def _relpath_under_dataset(path: str, dataset_root: str, fallback: str) -> str:
    """Относительный путь для зеркалирования структуры в output; если вне dataset_root — fallback."""
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
        sys.exit(f"[ERROR] Нет каталога датасета: {dr}")
    if datasets_info_path_override is not None:
        alt = os.path.join(datasets_info_path_override, dataset_name)
        if not os.path.isdir(alt):
            sys.exit(f"[ERROR] При --datasets-info-path ожидается каталог {alt}")
        if os.path.realpath(dr) != os.path.realpath(alt):
            sys.exit(
                "[ERROR] --source-path и --datasets-info-path должны указывать на один и тот же датасет "
                f"({dr} vs {alt})"
            )
    return dr


def _load_roi_config(args: argparse.Namespace, entry: Dict[str, Any]) -> Dict[str, Any]:
    ra = entry.get("roi_auto")
    if args.require_roi_auto and not isinstance(ra, dict):
        sys.exit("[ERROR] В записи датасета нет объекта roi_auto (--require-roi-auto).")

    ra = ra if isinstance(ra, dict) else {}
    weights = args.weights or ra.get("weights")
    if not weights:
        sys.exit("[ERROR] Укажите weights в roi_auto или через --weights.")

    conf = args.conf if args.conf is not None else float(ra.get("conf", 0.25))
    pad_px = args.pad_px if args.pad_px is not None else int(ra.get("pad_px", 0))
    roi_policy = args.roi_policy or ra.get("roi_policy", "largest")
    mode = args.mode or ra.get("mode", "yolo_detect")

    if roi_policy not in ROI_POLICIES:
        sys.exit(f"[ERROR] Недопустимый roi_policy: {roi_policy}")
    if mode not in MODES:
        sys.exit(f"[ERROR] Недопустимый mode: {mode}")

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
    args = parse_args(argv)
    legacy_source = (args.source_path or "").strip()
    workspace_root: Optional[str] = None
    layout: Optional[WorkspaceLayout] = None

    need_default_roi_output = False
    if legacy_source:
        if not args.output_path or not str(args.output_path).strip():
            sys.exit("[ERROR] В legacy-режиме (--source-path) укажите --output-path")
        source_path_abs = os.path.abspath(os.path.expanduser(legacy_source))
        info_path = _resolve_datasets_info_file(
            args.datasets_info_path, os.path.join(source_path_abs, DATASETS_INFO_FILE)
        )
        temp_root = os.path.join(source_path_abs, "tmp")
        os.makedirs(temp_root, exist_ok=True)
    else:
        ws = (args.workspace or "").strip() or (os.environ.get(WORKSPACE_ENV_VAR) or "").strip()
        if not ws:
            sys.exit(
                f"[ERROR] Укажите --source-path (legacy) или --workspace / переменную {WORKSPACE_ENV_VAR}"
            )
        workspace_root = os.path.abspath(os.path.expanduser(ws))
        layout = WorkspaceLayout(workspace_root)
        info_path = _resolve_datasets_info_file(
            args.datasets_info_path, layout.source_datasets_info_path()
        )
        if args.tmp_dir and str(args.tmp_dir).strip():
            temp_root = os.path.abspath(os.path.expanduser(str(args.tmp_dir).strip()))
        else:
            temp_root = os.path.join(workspace_root, "tmp")
        os.makedirs(temp_root, exist_ok=True)
        if not (args.output_path or "").strip():
            need_default_roi_output = True

    if not os.path.isfile(info_path):
        sys.exit(f"[ERROR] Не найден {info_path}")

    with open(info_path, "r", encoding="utf-8") as f:
        datasets_info = json.load(f)

    dataset_key = _resolve_catalog_dataset_key(datasets_info, args.dataset_name)
    if dataset_key != args.dataset_name:
        print(f"[INFO] Используется ключ датасета в каталоге: {dataset_key!r}")

    entry = datasets_info[dataset_key]

    if need_default_roi_output:
        assert layout is not None
        args.output_path = os.path.join(layout.source_datasets, f"{dataset_key}_roi")
    structure = entry.get("structure")
    if not structure:
        sys.exit("[ERROR] В записи датасета нет поля structure")

    if legacy_source:
        source_path_abs = os.path.abspath(os.path.expanduser(legacy_source))
        dataset_root = _validate_layout_legacy(
            source_path_abs, dataset_key, args.datasets_info_path
        )
    else:
        assert workspace_root is not None and layout is not None
        dataset_root = resolve_dataset_root_for_entry(
            dataset_key,
            entry,
            workspace_root=workspace_root,
            source_catalog_dir=layout.source_datasets,
            legacy_source_parent=layout.source_datasets,
        )

    cfg = _load_roi_config(args, entry)
    output_root = os.path.abspath(os.path.expanduser(str(args.output_path).strip()))
    os.makedirs(output_root, exist_ok=True)

    model = YOLO(cfg["weights"])
    if cfg["mode"] == "yolo_segment" and getattr(model, "task", None) != "segment":
        print(
            f"[WARNING] mode=yolo_segment, но модель task={getattr(model, 'task', None)}; "
            "используются ограничивающие прямоугольники детекций/масок."
        )

    buckets = iter_image_label_buckets(
        dataset_root,
        structure,
        entry,
        dataset_name=dataset_key,
        temp_root=temp_root,
        exclude_test=False,
    )
    if not buckets:
        sys.exit(f"[ERROR] Нет пар images/labels для structure={structure}")

    stats = {"images": 0, "skipped": 0}

    for img_dir, lbl_dir in buckets:
        rel_base = _relpath_under_dataset(img_dir, dataset_root, "images")
        out_img_dir = os.path.join(output_root, rel_base)
        rel_lbl = _relpath_under_dataset(lbl_dir, dataset_root, "labels")
        out_lbl_dir = os.path.join(output_root, rel_lbl)

        files = [f for f in sorted(os.listdir(img_dir)) if _is_image_file(f)]
        for fname in tqdm(files, desc=rel_base):
            src_img = os.path.join(img_dir, fname)
            stem, _ext = os.path.splitext(fname)
            src_lbl = os.path.join(lbl_dir, stem + ".txt")
            label_lines = _read_label_lines(src_lbl)

            with Image.open(src_img) as im:
                im = im.convert("RGB")
                iw, ih = im.size

            results = model.predict(
                source=src_img,
                conf=cfg["conf"],
                verbose=False,
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
                    sys.exit(f"[ERROR] Нет детекций для ROI: {src_img}")
                if args.on_empty == "skip":
                    stats["skipped"] += 1
                    continue
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

                out_lines: List[str] = []
                for raw in label_lines:
                    new_l = _transform_label_line(raw, crop, iw, ih)
                    if new_l is not None:
                        out_lines.append(new_l + "\n")
                out_lbl_path = os.path.join(out_lbl_dir, out_stem + ".txt")
                _write_label_lines(out_lbl_path, out_lines)

            if cfg["roi_policy"] == "per_box":
                for idx, box in enumerate(roi_list, start=1):
                    process_one_crop(box, f"{stem}_split_{idx}")
                stats["images"] += len(roi_list)
            else:
                process_one_crop(roi_list[0], stem)
                stats["images"] += 1

    _copy_and_patch_yaml(dataset_root, output_root)
    _ensure_data_yaml_after_roi(output_root, entry)
    print(
        f"[OK] Готово: {stats['images']} выходных кадров, пропущено {stats['skipped']}, "
        f"каталог: {output_root}"
    )


if __name__ == "__main__":
    main()
