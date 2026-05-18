from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ultralytics.data.utils import img2label_paths, verify_image_label


@dataclass
class UnifiedGtRow:
    image_path: str
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def _load_data_cfg(data_yaml_path: str) -> dict[str, Any]:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return payload if isinstance(payload, dict) else {}


def _resolve_split_image_paths(data_yaml_path: str, split_name: str) -> list[str]:
    data = _load_data_cfg(data_yaml_path)
    split_value = data.get(split_name)
    if split_value is None:
        raise ValueError(f"data.yaml has no split={split_name!r}")
    root = Path(data_yaml_path).resolve().parent
    candidates = split_value if isinstance(split_value, list) else [split_value]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: list[str] = []
    for candidate in candidates:
        p = Path(str(candidate))
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.is_file() and p.suffix.lower() == ".txt":
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    q = Path(line)
                    if not q.is_absolute():
                        q = (root / q).resolve()
                    out.append(str(q.resolve()))
            continue
        if p.is_file() and p.suffix.lower() in exts:
            out.append(str(p.resolve()))
            continue
        if p.is_dir():
            for dirpath, _dirnames, filenames in os.walk(str(p)):
                for name in sorted(filenames):
                    if Path(name).suffix.lower() in exts:
                        out.append(str(Path(dirpath, name).resolve()))
    return list(dict.fromkeys(out))


def collect_ultralytics_style_gt(
    data_yaml_path: str,
    split_name: str,
    names: list[str],
) -> tuple[list[UnifiedGtRow], list[str], list[str]]:
    image_paths = _resolve_split_image_paths(data_yaml_path, split_name)
    label_paths = img2label_paths(image_paths)
    gt_rows: list[UnifiedGtRow] = []
    issues: list[str] = []
    for im_file, lb_file in zip(image_paths, label_paths):
        checked = verify_image_label((im_file, lb_file, "", False, max(len(names), 1), 0, 0, False))
        msg = str(checked[9] or "")
        if msg:
            issues.append(msg)
        if checked[0] is None or checked[1] is None or checked[2] is None:
            continue
        labels = np.asarray(checked[1], dtype=np.float32)
        if labels.size == 0:
            continue
        img_h, img_w = checked[2]
        for row in labels:
            cls_id = int(row[0])
            xc, yc, w, h = [float(v) for v in row[1:5]]
            bw = w * float(img_w)
            bh = h * float(img_h)
            x1 = (xc * float(img_w)) - bw / 2.0
            y1 = (yc * float(img_h)) - bh / 2.0
            x2 = x1 + bw
            y2 = y1 + bh
            gt_rows.append(
                UnifiedGtRow(
                    image_path=str(Path(im_file).resolve()),
                    cls_id=cls_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
    return gt_rows, image_paths, issues

