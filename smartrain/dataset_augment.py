from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import zlib
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from tqdm import tqdm
from ultralytics import YOLO

from smartrain.cli_argparse import CliArgumentParser
from smartrain.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.dataset_hash import calculate_dataset_hash
from smartrain.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "test": "test"}
_BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
_ROI_MODEL_CACHE: dict[str, YOLO] = {}


def build_augment_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Офлайн-аугментация датасета в новый datasets/<name>")
    p.add_argument("--workspace", type=str, default=None, help=f"Корень workspace (иначе {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Имя исходного датасета из datasets_info.json")
    p.add_argument("--output-name", type=str, default=None, help="Имя выходного датасета (по умолчанию <dataset>_aug)")
    p.add_argument("--enable-flip", action="store_true", help="Включить flip-аугментацию")
    p.add_argument("--enable-photometric", action="store_true", help="Включить brightness/contrast")
    p.add_argument("--enable-conveyor", action="store_true", help="Включить конвейерные шум/blur/shift/rotate")
    p.add_argument("--enable-center-rotate", action="store_true", default=True, help="Включить поворот кадра вокруг центра")
    p.add_argument("--center-rotate-deg", type=float, default=5.0, help="Максимальный угол поворота в обе стороны")
    p.add_argument("--enable-bbox-copy", action="store_true", help="Включить bbox-copy аугментацию")
    p.add_argument("--flip", choices=("horizontal", "vertical", "both", "none"), default="horizontal")
    p.add_argument("--brightness-limit", type=float, default=0.1, help="Для policy=basic: диапазон brightness")
    p.add_argument("--contrast-limit", type=float, default=0.1, help="Для policy=basic: диапазон contrast")
    p.add_argument("--copy-paste-count", type=int, default=1, help="Для policy=bbox_copy: число вставок на изображение")
    p.add_argument("--copy-paste-rotation", type=float, default=12.0, help="Для policy=bbox_copy: max |угол| для поворота вставки")
    p.add_argument("--copy-paste-scale-min", type=float, default=0.85, help="Для policy=bbox_copy: min scale (inner)")
    p.add_argument("--copy-paste-scale-max", type=float, default=1.2, help="Для policy=bbox_copy: max scale (inner)")
    p.add_argument("--copy-paste-max-iou", type=float, default=0.0, help="Макс. IoU вставки с существующими bbox")
    p.add_argument("--copy-paste-tries", type=int, default=25, help="Число попыток подобрать валидное размещение")
    p.add_argument(
        "--class-balance",
        choices=("on", "off"),
        default="on",
        help="Для bbox_copy: балансировать выбор доноров по классам",
    )
    p.add_argument(
        "--color-match",
        choices=("meanstd", "off"),
        default="meanstd",
        help="Для bbox_copy: выравнивание яркости/контраста патча относительно целевой области",
    )
    p.add_argument(
        "--blend-feather",
        type=float,
        default=0.16,
        help="Для bbox_copy: сила сглаживания шва [0..0.5], 0=без feather",
    )
    p.add_argument(
        "--placement-mode",
        choices=("none", "bbox", "detector"),
        default="detector",
        help="Режим ROI-размещения для bbox_copy: none|bbox|detector (по умолчанию detector)",
    )
    p.add_argument("--placement-roi", action="store_true", help="Legacy: то же, что --placement-mode bbox")
    p.add_argument("--roi-model", type=str, default="yolo11n.pt", help="Модель детектора ROI для --placement-mode detector")
    p.add_argument("--roi-conf", type=float, default=0.25, help="Порог confidence для ROI-детектора")
    p.add_argument("--roi-class-ids", type=str, default=None, help="CSV class ids для ROI-детектора (пусто=все)")
    p.add_argument("--side-tolerance-px", type=float, default=3.0, help="Допуск в px для классификации стороны ROI")
    p.add_argument("--multiplier", type=int, default=1, help="Сколько аугментированных копий на исходное изображение")
    p.add_argument("--splits", type=str, default="train", help="CSV: train,val,test")
    p.add_argument("--classes", type=str, default=None, help="Ограничить аугментацию классами CSV")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-legend", action="store_true")
    return p


def _load_catalog(layout: WorkspaceLayout) -> dict:
    info_path = layout.work_datasets_info_path()
    if not os.path.isfile(info_path):
        return {}
    with open(info_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _detect_split(images_path: str) -> str:
    low = images_path.lower()
    if "/train/" in low:
        return "train"
    if "/val/" in low or "/valid/" in low:
        return "val"
    if "/test/" in low:
        return "test"
    return "train"


def _read_yolo_classes(label_path: str) -> set[int]:
    out: set[int] = set()
    if not os.path.isfile(label_path):
        return out
    with open(label_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                out.add(int(float(parts[0])))
            except ValueError:
                continue
    return out


def _parse_yolo_labels(label_path: str) -> list[tuple[int, float, float, float, float]]:
    out: list[tuple[int, float, float, float, float]] = []
    if not os.path.isfile(label_path):
        return out
    for raw in Path(label_path).read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            out.append((int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        except ValueError:
            continue
    return out


def _serialize_yolo_labels(labels: list[tuple[int, float, float, float, float]]) -> str:
    return "".join(f"{cls} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for cls, x, y, w, h in labels)


def _compose_for_basic(args) -> A.Compose:
    t: list[A.BasicTransform] = []
    if args.enable_flip:
        if args.flip == "horizontal":
            t.append(A.HorizontalFlip(p=1.0))
        elif args.flip == "vertical":
            t.append(A.VerticalFlip(p=1.0))
        elif args.flip == "both":
            t.append(A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]))
    if args.enable_conveyor:
        t.extend(
            [
                A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=5, border_mode=0, p=0.8),
                A.GaussNoise(p=0.3),
                A.MotionBlur(blur_limit=3, p=0.15),
            ]
        )
    if args.enable_photometric:
        t.append(
            A.RandomBrightnessContrast(
                brightness_limit=float(args.brightness_limit),
                contrast_limit=float(args.contrast_limit),
                p=1.0,
            )
        )
    if args.enable_center_rotate:
        t.append(A.Affine(rotate=(-float(args.center_rotate_deg), float(args.center_rotate_deg)), p=1.0))
    return A.Compose(
        t,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], clip=True),
    )


def _sanitize_yolo_box(
    label: tuple[int, float, float, float, float],
    *,
    min_size: float = 1e-6,
) -> tuple[int, float, float, float, float] | None:
    cls_id, cx, cy, bw, bh = label
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    x1 = min(1.0, max(0.0, x1))
    y1 = min(1.0, max(0.0, y1))
    x2 = min(1.0, max(0.0, x2))
    y2 = min(1.0, max(0.0, y2))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    nx = (x1 + x2) / 2.0
    ny = (y1 + y2) / 2.0
    nw = x2 - x1
    nh = y2 - y1
    return (cls_id, nx, ny, nw, nh)


def _base36(num: int) -> str:
    if num <= 0:
        return "0"
    x = num
    out = []
    while x:
        x, r = divmod(x, 36)
        out.append(_BASE36_ALPHABET[r])
    return "".join(reversed(out))


def _variant_code(args) -> str:
    if args.enable_conveyor:
        return "s"
    if args.enable_photometric:
        return "l"
    return "n"


def _flip_code(args) -> str:
    if args.enable_bbox_copy:
        return "n"
    if not args.enable_flip:
        return "n"
    return {"horizontal": "h", "vertical": "v", "both": "b", "none": "n"}[args.flip]


def _aug_stem(stem: str, args, idx: int) -> str:
    mode = "p" if args.enable_bbox_copy else ("c" if args.enable_conveyor else "b")
    return f"{stem}__a-{mode}{_flip_code(args)}{_variant_code(args)}{_base36(idx)}"


def _to_xyxy(box: tuple[int, float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    _, cx, cy, bw, bh = box
    x1 = int(round((cx - bw / 2.0) * w))
    y1 = int(round((cy - bh / 2.0) * h))
    x2 = int(round((cx + bw / 2.0) * w))
    y2 = int(round((cy + bh / 2.0) * h))
    return x1, y1, x2, y2


def _to_yolo(cls_id: int, xyxy: tuple[int, int, int, int], w: int, h: int) -> tuple[int, float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return (cls_id, cx / w, cy / h, bw / w, bh / h)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def _roi_from_labels(labels: list[tuple[int, float, float, float, float]], img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    if not labels:
        return None
    boxes = [_to_xyxy(x, img_w, img_h) for x in labels]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def _classify_side(roi: tuple[int, int, int, int], box: tuple[int, int, int, int], tol: float) -> str:
    rx1, ry1, rx2, ry2 = roi
    bx1, by1, bx2, by2 = box
    d = {
        "left": abs(bx1 - rx1),
        "right": abs(rx2 - bx2),
        "top": abs(by1 - ry1),
        "bottom": abs(ry2 - by2),
    }
    first = min(d, key=d.get)
    first_v = d[first]
    sorted_vals = sorted(d.items(), key=lambda it: it[1])
    second, second_v = sorted_vals[1]
    if second_v - first_v <= tol:
        pair = sorted([first, second])
        return f"corner:{pair[0]}-{pair[1]}"
    return first


def _inside(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 >= bx1 and ay1 >= by1 and ax2 <= bx2 and ay2 <= by2


def _parse_roi_class_ids(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    out: list[int] = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out or None


def _detect_roi_box(image_path: str, args) -> tuple[int, int, int, int] | None:
    model_path = str(getattr(args, "roi_model", "") or "yolo11n.pt")
    if model_path not in _ROI_MODEL_CACHE:
        _ROI_MODEL_CACHE[model_path] = YOLO(model_path)
    model = _ROI_MODEL_CACHE[model_path]
    class_ids = _parse_roi_class_ids(getattr(args, "roi_class_ids", None))
    results = model.predict(
        source=image_path,
        conf=float(getattr(args, "roi_conf", 0.25)),
        classes=class_ids,
        verbose=False,
    )
    if not results:
        return None
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    idx = int(np.argmax(confs)) if confs is not None and len(confs) else 0
    x1, y1, x2, y2 = xyxy[idx].tolist()
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def _apply_geom_aug(image_path: str, label_path: str, out_img: str, out_lbl: str, args) -> None:
    image = np.array(Image.open(image_path).convert("RGB"))
    raw_labels = _parse_yolo_labels(label_path)
    labels = [x for x in (_sanitize_yolo_box(lb) for lb in raw_labels) if x is not None]
    bboxes = [(x, y, w, h) for _, x, y, w, h in labels]
    class_labels = [cls for cls, *_ in labels]
    pipeline = _compose_for_basic(args)
    transformed = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
    new_img = transformed["image"]
    new_labels_raw = [
        (int(cls), float(x), float(y), float(w), float(h))
        for cls, (x, y, w, h) in zip(transformed["class_labels"], transformed["bboxes"])
    ]
    new_labels = [x for x in (_sanitize_yolo_box(lb) for lb in new_labels_raw) if x is not None]
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    os.makedirs(os.path.dirname(out_lbl), exist_ok=True)
    Image.fromarray(new_img).save(out_img)
    Path(out_lbl).write_text(_serialize_yolo_labels(new_labels), encoding="utf-8")


def _build_donor_pool(items: list[dict], args) -> list[dict]:
    donors: list[dict] = []
    progress = tqdm(
        items,
        total=len(items),
        desc="augment:donors",
        unit="img",
        disable=bool(args.no_legend),
    )
    for it in progress:
        img = Image.open(it["img"]).convert("RGB")
        w, h = img.size
        labels = _parse_yolo_labels(it["lbl"])
        roi = _roi_from_labels(labels, w, h)
        for lb in labels:
            cls_id = lb[0]
            xyxy = _to_xyxy(lb, w, h)
            x1, y1, x2, y2 = xyxy
            if x2 <= x1 or y2 <= y1:
                continue
            patch = img.crop((x1, y1, x2, y2))
            if roi is None:
                side = "inner"
            else:
                side_key = _classify_side(roi, xyxy, float(args.side_tolerance_px))
                rx1, ry1, rx2, ry2 = roi
                touches = abs(x1 - rx1) <= args.side_tolerance_px or abs(x2 - rx2) <= args.side_tolerance_px or abs(y1 - ry1) <= args.side_tolerance_px or abs(y2 - ry2) <= args.side_tolerance_px
                side = f"edge:{side_key}" if touches else "inner"
            donors.append({"class_id": cls_id, "patch": patch, "w": x2 - x1, "h": y2 - y1, "side": side})
        progress.set_postfix(donors=len(donors), refresh=False)
    progress.close()
    return donors


def _match_patch_to_region(patch_arr: np.ndarray, region_arr: np.ndarray) -> np.ndarray:
    patch = patch_arr.astype(np.float32)
    region = region_arr.astype(np.float32)
    p_mean = patch.mean(axis=(0, 1), keepdims=True)
    r_mean = region.mean(axis=(0, 1), keepdims=True)
    p_std = patch.std(axis=(0, 1), keepdims=True) + 1e-6
    r_std = region.std(axis=(0, 1), keepdims=True) + 1e-6
    matched = (patch - p_mean) * (r_std / p_std) + r_mean
    return np.clip(matched, 0, 255).astype(np.uint8)


def _feather_alpha(width: int, height: int, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return np.ones((height, width, 1), dtype=np.float32)
    strength = min(0.5, max(0.01, float(strength)))
    edge = max(2, int(min(width, height) * strength))
    x = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float32)
    y = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float32)
    fx = np.clip(x / edge, 0.0, 1.0)
    fy = np.clip(y / edge, 0.0, 1.0)
    return np.outer(fy, fx)[..., None]


def _pick_donor_balanced(donors: list[dict], class_usage: dict[int, int], rng: random.Random) -> dict:
    by_class: dict[int, list[dict]] = {}
    for d in donors:
        by_class.setdefault(int(d["class_id"]), []).append(d)
    min_used = min(class_usage.get(c, 0) for c in by_class.keys())
    candidate_classes = [c for c in by_class.keys() if class_usage.get(c, 0) == min_used]
    cls = candidate_classes[rng.randrange(0, len(candidate_classes))]
    pool = by_class[cls]
    return pool[rng.randrange(0, len(pool))]


def _pick_donor_any(donors: list[dict], rng: random.Random) -> dict:
    return donors[rng.randrange(0, len(donors))]


def _apply_copy_paste(
    image_path: str,
    label_path: str,
    out_img: str,
    out_lbl: str,
    args,
    donors: list[dict],
    class_usage: dict[int, int],
) -> bool:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    labels = _parse_yolo_labels(label_path)
    placement_mode = str(getattr(args, "placement_mode", "detector"))
    roi: tuple[int, int, int, int] | None = None
    if placement_mode == "bbox":
        roi = _roi_from_labels(labels, w, h)
    elif placement_mode == "detector":
        roi = _detect_roi_box(image_path, args)
    if placement_mode in ("bbox", "detector") and roi is None:
        return False
    # Опциональный поворот кадра перед copy-paste.
    if bool(getattr(args, "enable_center_rotate", False)):
        path_seed = zlib.crc32(image_path.encode("utf-8")) & 0xFFFFFFFF
        rng_rot = random.Random(int(args.seed) + 101 + path_seed)
        angle = rng_rot.uniform(-float(getattr(args, "center_rotate_deg", 5.0)), float(getattr(args, "center_rotate_deg", 5.0)))
        img_arr = np.array(img.convert("RGB"))
        if placement_mode in ("bbox", "detector") and roi is not None:
            cx = (roi[0] + roi[2]) / 2.0
            cy = (roi[1] + roi[3]) / 2.0
        else:
            cx = w / 2.0
            cy = h / 2.0
        m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot = cv2.warpAffine(img_arr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        img = Image.fromarray(rot)
        new_labels: list[tuple[int, float, float, float, float]] = []
        for lb in labels:
            cls, x, y, bw, bh = lb
            x1, y1, x2, y2 = _to_xyxy(lb, w, h)
            corners = np.array(
                [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
                dtype=np.float32,
            )
            tr = (m @ corners.T).T
            nx1 = int(np.floor(np.min(tr[:, 0])))
            ny1 = int(np.floor(np.min(tr[:, 1])))
            nx2 = int(np.ceil(np.max(tr[:, 0])))
            ny2 = int(np.ceil(np.max(tr[:, 1])))
            nx1 = max(0, min(w, nx1))
            ny1 = max(0, min(h, ny1))
            nx2 = max(0, min(w, nx2))
            ny2 = max(0, min(h, ny2))
            if nx2 <= nx1 or ny2 <= ny1:
                continue
            new_labels.append(_to_yolo(int(cls), (nx1, ny1, nx2, ny2), w, h))
        labels = new_labels
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            rc = np.array(
                [[rx1, ry1, 1.0], [rx2, ry1, 1.0], [rx2, ry2, 1.0], [rx1, ry2, 1.0]],
                dtype=np.float32,
            )
            rr = (m @ rc.T).T
            rx1n = int(np.floor(np.min(rr[:, 0])))
            ry1n = int(np.floor(np.min(rr[:, 1])))
            rx2n = int(np.ceil(np.max(rr[:, 0])))
            ry2n = int(np.ceil(np.max(rr[:, 1])))
            roi = (
                max(0, min(w, rx1n)),
                max(0, min(h, ry1n)),
                max(0, min(w, rx2n)),
                max(0, min(h, ry2n)),
            )
            if roi[2] <= roi[0] or roi[3] <= roi[1]:
                return False
    existing = [_to_xyxy(lb, w, h) for lb in labels]
    rng = random.Random(int(args.seed))
    total_placed = 0
    for _ in range(int(args.copy_paste_count)):
        if not donors:
            break
        placed = False
        for _try in range(int(args.copy_paste_tries)):
            if str(getattr(args, "class_balance", "on")) == "on":
                d = _pick_donor_balanced(donors, class_usage, rng)
            else:
                d = _pick_donor_any(donors, rng)
            patch = d["patch"].copy()
            edge_mode = d["side"].startswith("edge:")
            if not edge_mode:
                angle = rng.uniform(-float(args.copy_paste_rotation), float(args.copy_paste_rotation))
                patch = patch.rotate(angle, resample=Image.BICUBIC, expand=True)
                s = rng.uniform(float(args.copy_paste_scale_min), float(args.copy_paste_scale_max))
                nw = max(2, int(round(patch.size[0] * s)))
                nh = max(2, int(round(patch.size[1] * s)))
                patch = patch.resize((nw, nh), Image.BICUBIC)
            pw, ph = patch.size
            if pw >= w or ph >= h:
                continue
            if placement_mode in ("bbox", "detector") and roi is not None:
                rx1, ry1, rx2, ry2 = roi
                if edge_mode:
                    side = d["side"].split(":", 1)[1]
                    if side.startswith("corner:"):
                        c = side.split(":", 1)[1]
                        if c == "left-top":
                            x, y = rx1, ry1
                        elif c == "left-bottom":
                            x, y = rx1, max(ry1, ry2 - ph)
                        elif c == "right-top":
                            x, y = max(rx1, rx2 - pw), ry1
                        else:
                            x, y = max(rx1, rx2 - pw), max(ry1, ry2 - ph)
                    elif side == "left":
                        x = rx1
                        y = rng.randint(ry1, max(ry1, ry2 - ph))
                    elif side == "right":
                        x = max(rx1, rx2 - pw)
                        y = rng.randint(ry1, max(ry1, ry2 - ph))
                    elif side == "top":
                        x = rng.randint(rx1, max(rx1, rx2 - pw))
                        y = ry1
                    else:
                        x = rng.randint(rx1, max(rx1, rx2 - pw))
                        y = max(ry1, ry2 - ph)
                else:
                    x = rng.randint(rx1, max(rx1, rx2 - pw))
                    y = rng.randint(ry1, max(ry1, ry2 - ph))
            else:
                x = rng.randint(0, w - pw)
                y = rng.randint(0, h - ph)
            cand = (x, y, x + pw, y + ph)
            if placement_mode in ("bbox", "detector") and roi is not None and not _inside(cand, roi):
                continue
            if any(_iou(cand, ex) > float(args.copy_paste_max_iou) for ex in existing):
                continue
            region = img.crop((x, y, x + pw, y + ph))
            patch_arr = np.array(patch.convert("RGB"))
            region_arr = np.array(region.convert("RGB"))
            if str(getattr(args, "color_match", "meanstd")) == "meanstd":
                patch_arr = _match_patch_to_region(patch_arr, region_arr)
            alpha = _feather_alpha(pw, ph, float(getattr(args, "blend_feather", 0.16)))
            blended = (patch_arr.astype(np.float32) * alpha + region_arr.astype(np.float32) * (1.0 - alpha)).astype(
                np.uint8
            )
            img.paste(Image.fromarray(blended), (x, y))
            existing.append(cand)
            labels.append(_to_yolo(int(d["class_id"]), cand, w, h))
            class_usage[int(d["class_id"])] = class_usage.get(int(d["class_id"]), 0) + 1
            placed = True
            total_placed += 1
            break
        if not placed:
            continue
    if total_placed == 0:
        return False
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    os.makedirs(os.path.dirname(out_lbl), exist_ok=True)
    img.save(out_img)
    Path(out_lbl).write_text(_serialize_yolo_labels(labels), encoding="utf-8")
    return True


def _write_data_yaml(out_dir: str, names: list[str]) -> None:
    p = Path(out_dir) / "data.yaml"
    p.write_text(
        "train: ./train/images\nval: ./val/images\ntest: ./test/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )


def _update_datasets_sidecar(
    layout: WorkspaceLayout,
    output_key: str,
    class_map: dict[str, int],
    target_dir: str,
    output_hash: str,
) -> None:
    os.makedirs(layout.datasets, exist_ok=True)
    rel = os.path.relpath(os.path.abspath(target_dir), layout.root)
    entry = {
        "classes": {str(k): int(v) for k, v in sorted(class_map.items(), key=lambda kv: int(kv[1]))},
        "structure": "split",
        "elements_count": None,
        "data_path": rel,
        "dataset_hash": output_hash,
        "modified": False,
    }
    info_path = layout.work_datasets_info_path()
    previous: dict = {}
    if os.path.isfile(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            previous = loaded
    previous[output_key] = entry
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(previous, f, ensure_ascii=False, indent=4)

    cn_path = layout.work_class_names_path()
    class_names_out: dict[str, str] = {}
    if os.path.isfile(cn_path):
        with open(cn_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            class_names_out = {str(k): str(v) for k, v in loaded.items()}
    for c in class_map.keys():
        class_names_out[str(c)] = str(c)
    with open(cn_path, "w", encoding="utf-8") as f:
        json.dump(class_names_out, f, ensure_ascii=False, indent=4)


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


def _interactive_fill(args, dataset_names: list[str], classes: list[str], workspace_root: str) -> None:
    print("[INFO] Интерактивный режим augment")
    print("[INFO] Доступные датасеты:")
    for n in dataset_names:
        print(f"  - {n}")
    print("[INFO] Доступные классы:")
    for c in classes:
        print(f"  - {c}")
    args.dataset = prompt("Датасет: ", completer=WordCompleter(dataset_names, ignore_case=True)).strip()
    args.output_name = prompt("Имя выходного датасета (пусто=авто): ", default=(args.output_name or "")).strip() or None
    args.enable_flip = (
        prompt("Включить flip? [Y/n]: ", default=("y" if args.enable_flip else "n")).strip().lower()
        in ("y", "yes", "1", "true")
    )
    if args.enable_flip:
        args.flip = (
            prompt(
                "Flip (horizontal/vertical/both/none): ",
                default=args.flip,
                completer=WordCompleter(["horizontal", "vertical", "both", "none"], ignore_case=True),
            ).strip()
            or args.flip
        )
    args.enable_photometric = (
        prompt("Включить brightness/contrast? [y/N]: ", default=("y" if args.enable_photometric else "n")).strip().lower()
        in ("y", "yes", "1", "true")
    )
    args.enable_conveyor = (
        prompt("Включить conveyor шум/blur/shift/rotate? [y/N]: ", default=("y" if args.enable_conveyor else "n")).strip().lower()
        in ("y", "yes", "1", "true")
    )
    args.enable_center_rotate = (
        prompt("Включить поворот кадра вокруг центра? [y/N]: ", default=("y" if args.enable_center_rotate else "n")).strip().lower()
        in ("y", "yes", "1", "true")
    )
    if args.enable_center_rotate:
        args.center_rotate_deg = float(
            prompt("Предел угла поворота (градусы, +-): ", default=str(getattr(args, "center_rotate_deg", 5.0))).strip()
            or str(getattr(args, "center_rotate_deg", 5.0))
        )
    args.enable_bbox_copy = (
        prompt("Включить bbox_copy? [y/N]: ", default=("y" if args.enable_bbox_copy else "n")).strip().lower()
        in ("y", "yes", "1", "true")
    )
    args.multiplier = int(prompt("Multiplier: ", default=str(args.multiplier)).strip() or str(args.multiplier))
    if args.enable_bbox_copy:
        args.placement_mode = (
            prompt(
                "ROI mode (none/bbox/detector): ",
                default=str(getattr(args, "placement_mode", "detector")),
                completer=WordCompleter(["none", "bbox", "detector"], ignore_case=True),
            ).strip()
            or str(getattr(args, "placement_mode", "detector"))
        )
        if args.placement_mode == "detector":
            models = _list_workspace_detector_models(workspace_root)
            if models:
                print("[INFO] ROI-детекторы в корне workspace:")
                for m in models:
                    print(f"  - {m}")
            model_completer = WordCompleter(models, ignore_case=True) if models else None
            args.roi_model = (
                prompt(
                    "ROI detector model (--roi-model): ",
                    default=str(getattr(args, "roi_model", "yolo11n.pt")),
                    completer=model_completer,
                    complete_while_typing=True,
                ).strip()
                or str(getattr(args, "roi_model", "yolo11n.pt"))
            )
            args.roi_conf = float(prompt("ROI conf (--roi-conf): ", default=str(getattr(args, "roi_conf", 0.25))).strip() or str(getattr(args, "roi_conf", 0.25)))
            args.roi_class_ids = (
                prompt("ROI class ids CSV (--roi-class-ids, пусто=все): ", default=str(getattr(args, "roi_class_ids", "") or "")).strip()
                or None
            )
        args.class_balance = (
            prompt(
                "Class balance (on/off): ",
                default=str(getattr(args, "class_balance", "on")),
                completer=WordCompleter(["on", "off"], ignore_case=True),
            ).strip()
            or str(getattr(args, "class_balance", "on"))
        )
        args.color_match = (
            prompt(
                "Color match (meanstd/off): ",
                default=str(getattr(args, "color_match", "meanstd")),
                completer=WordCompleter(["meanstd", "off"], ignore_case=True),
            ).strip()
            or str(getattr(args, "color_match", "meanstd"))
        )
        args.blend_feather = float(
            prompt("Blend feather [0..0.5]: ", default=str(getattr(args, "blend_feather", 0.16))).strip()
            or str(getattr(args, "blend_feather", 0.16))
        )
        args.copy_paste_count = int(
            prompt("Copy-paste count: ", default=str(args.copy_paste_count)).strip() or str(args.copy_paste_count)
        )
    args.splits = prompt("Splits CSV (train,val,test): ", default=args.splits).strip() or args.splits
    args.classes = (
        prompt(
            "Классы CSV (пусто=все): ",
            default="",
            completer=WordCompleter(classes, ignore_case=True),
            complete_while_typing=True,
        ).strip()
        or None
    )
    args.dry_run = (prompt("Dry-run? [y/N]: ", default="n").strip().lower() in ("y", "yes", "1", "true"))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = build_augment_arg_parser().parse_args(argv)
    root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(root)
    catalog = _load_catalog(layout)
    if not catalog:
        print("[ERROR] Не найдено datasets_info.json или он пуст.")
        return

    if args.dataset is None and not argv:
        # на всякий случай, но у нас подкоманда вызывается с аргументами от cli
        pass
    if args.dataset is None and sys.stdin.isatty():
        all_classes = sorted({k for v in catalog.values() if isinstance(v, dict) for k in (v.get("classes") or {}).keys()})
        _interactive_fill(args, sorted(catalog.keys()), all_classes, layout.root)

    if not args.dataset:
        print("[ERROR] Укажите --dataset или используйте интерактивный режим.")
        return
    if args.dataset not in catalog:
        print(f"[ERROR] Неизвестный датасет: {args.dataset}")
        return
    if bool(getattr(args, "placement_roi", False)):
        args.placement_mode = "bbox"

    entry = catalog[args.dataset]
    src_root = resolve_dataset_root_for_entry(
        args.dataset,
        entry,
        workspace_root=layout.root,
        source_catalog_dir=layout.datasets,
        legacy_source_parent=layout.datasets,
    )
    class_map = entry.get("classes", {})
    names_by_id = {int(v): str(k) for k, v in class_map.items()} if isinstance(class_map, dict) else {}
    allowed_classes = None
    if args.classes:
        allowed_classes = {x.strip() for x in args.classes.split(",") if x.strip()}
    split_filter = {SPLIT_ALIASES.get(x.strip().lower(), x.strip().lower()) for x in args.splits.split(",") if x.strip()}
    out_base = args.output_name or f"{args.dataset}_aug"
    out_name = next_dataset_name(layout.datasets, out_base)
    out_dir = os.path.join(layout.datasets, out_name)

    buckets = iter_image_label_buckets(
        src_root,
        str(entry.get("structure", "split")),
        entry,
        dataset_name=args.dataset,
        temp_root=os.path.join(layout.root, "tmp"),
        exclude_test=False,
    )
    copied = 0
    augmented = 0
    skipped_roi_missing = 0
    items: list[dict[str, str]] = []
    for images_path, labels_path in buckets:
        split = _detect_split(images_path)
        for file_name in os.listdir(images_path):
            stem, ext = os.path.splitext(file_name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            items.append(
                {
                    "split": split,
                    "stem": stem,
                    "ext": ext,
                    "img": os.path.join(images_path, file_name),
                    "lbl": os.path.join(labels_path, f"{stem}.txt"),
                }
            )
    if not args.dry_run:
        for split in ("train", "val", "test"):
            os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, split, "labels"), exist_ok=True)
    donors = _build_donor_pool(items, args) if args.enable_bbox_copy else []
    class_usage: dict[int, int] = {}
    progress = tqdm(
        items,
        total=len(items),
        desc="augment",
        unit="img",
        disable=bool(args.no_legend),
    )
    for it in progress:
            split = it["split"]
            stem = it["stem"]
            ext = it["ext"]
            img_src = it["img"]
            lbl_src = it["lbl"]
            classes_in_image = _read_yolo_classes(lbl_src)
            class_names = {names_by_id.get(i, f"id_{i}") for i in classes_in_image}
            if allowed_classes and class_names.isdisjoint(allowed_classes):
                continue
            if not args.dry_run:
                dst_img = os.path.join(out_dir, split, "images", f"{stem}{ext}")
                dst_lbl = os.path.join(out_dir, split, "labels", f"{stem}.txt")
                shutil.copy2(img_src, dst_img)
                if os.path.isfile(lbl_src):
                    shutil.copy2(lbl_src, dst_lbl)
                else:
                    Path(dst_lbl).write_text("", encoding="utf-8")
            copied += 1
            if split not in split_filter:
                continue
            for i in range(args.multiplier):
                aug_stem = _aug_stem(stem, args, i + 1)
                if not args.dry_run:
                    out_img = os.path.join(out_dir, split, "images", f"{aug_stem}{ext}")
                    out_lbl = os.path.join(out_dir, split, "labels", f"{aug_stem}.txt")
                    if args.enable_bbox_copy:
                        ok = _apply_copy_paste(img_src, lbl_src, out_img, out_lbl, args, donors, class_usage)
                        if ok:
                            augmented += 1
                        elif args.placement_mode in ("bbox", "detector"):
                            skipped_roi_missing += 1
                    else:
                        _apply_geom_aug(img_src, lbl_src, out_img, out_lbl, args)
                        augmented += 1
                else:
                    augmented += 1
            progress.set_postfix(
                copied=copied,
                augmented=augmented,
                skipped_roi=skipped_roi_missing,
                refresh=False,
            )
    progress.close()

    if args.dry_run:
        print(f"[OK] dry-run: copied={copied}, augmented={augmented}, output={out_name}")
        return

    all_names = [str(x) for _, x in sorted(names_by_id.items())]
    _write_data_yaml(out_dir, all_names)
    out_hash = calculate_dataset_hash(out_dir)
    _update_datasets_sidecar(layout, out_name, class_map if isinstance(class_map, dict) else {}, out_dir, out_hash)
    passport_path = write_dataset_passport(
        output_dataset_dir=out_dir,
        command="augment",
        source_datasets=[
            {
                "name": args.dataset,
                "path": src_root,
                "dataset_hash": entry.get("dataset_hash"),
            }
        ],
        parameters=vars(args),
        transformations=[
            {
                "enable_flip": bool(args.enable_flip),
                "flip": args.flip,
                "enable_photometric": bool(args.enable_photometric),
                "enable_conveyor": bool(args.enable_conveyor),
                "enable_bbox_copy": bool(args.enable_bbox_copy),
                "placement_mode": args.placement_mode,
                "multiplier": args.multiplier,
                "splits": sorted(split_filter),
            }
        ],
        random_seed=args.seed,
        stats_before={"copied_input_images": copied},
        stats_after={
            "copied_images": copied,
            "augmented_images": augmented,
            "skipped_roi_missing": skipped_roi_missing,
            "output_hash": out_hash,
        },
    )
    print(f"[OK] Создан датасет: {out_dir}")
    print(f"[OK] Passport: {passport_path}")


if __name__ == "__main__":
    main()

