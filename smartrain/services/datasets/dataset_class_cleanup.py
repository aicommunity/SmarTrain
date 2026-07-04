from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from smartrain.services.datasets.cvsdcldet_converter import (
    _detection_label,
    collect_cvsdcldet_pairs,
)
from smartrain.services.datasets.dataset_access import find_dataset_paths
from smartrain.services.datasets.dataset_scan import (
    find_obj_data_file,
    find_obj_names_file,
    find_yaml_file,
    load_obj_names,
)
from smartrain.services.datasets.datasets_json_scan_core_service import (
    detect_structure,
    yolo_flat_image_label_buckets,
)
from smartrain.services.datasets.yolo_labels import (
    YoloBBox,
    YoloLabel,
    YoloSegment,
    read_yolo_labels,
    write_yolo_labels,
)

YOLO_STRUCTURES = frozenset({"split", "flat", "subset_flat", "nested_split", "darknet", "cvat11"})
CVSDCLDET_STRUCTURE = "cvsdcldet"


@dataclass
class StripUnusedClassesStats:
    classes_before: int = 0
    classes_after: int = 0
    removed_class_names: list[str] = field(default_factory=list)
    new_class_map: dict[str, int] = field(default_factory=dict)
    labels_remapped: int = 0
    unknown_class_ids: int = 0
    skipped: bool = False
    skip_reason: str | None = None


def normalize_class_name(name: str, class_names_map: dict[str, str] | None) -> str:
    if not class_names_map:
        return name
    return str(class_names_map.get(name, name))


def load_class_names_map(path: str | None) -> dict[str, str]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def load_id_to_name(dataset_dir: str, *, info: dict[str, Any] | None = None) -> dict[int, str]:
    yaml_path = find_yaml_file(dataset_dir)
    if yaml_path:
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception:
            raw = {}
        if isinstance(raw, dict):
            names = raw.get("names")
            if isinstance(names, dict):
                out: dict[int, str] = {}
                for k, v in names.items():
                    try:
                        out[int(k)] = str(v)
                    except (TypeError, ValueError):
                        continue
                if out:
                    return out
            if isinstance(names, list):
                return {i: str(v) for i, v in enumerate(names)}

    obj_names_path = find_obj_names_file(dataset_dir)
    if obj_names_path:
        names = load_obj_names(obj_names_path)
        if names:
            return {i: str(n) for i, n in enumerate(names)}

    if info and isinstance(info.get("classes"), dict):
        return {int(v): str(k) for k, v in info["classes"].items()}

    return {}


def _canonical_to_output_name(canonical: str, class_names_map: dict[str, str]) -> str:
    for alias, target in class_names_map.items():
        if target == canonical:
            return canonical
    return canonical


def build_compact_class_map(
    id_to_name: dict[int, str],
    used_canonical: set[str],
    class_names_map: dict[str, str] | None,
) -> tuple[dict[str, int], dict[int, int], list[str]]:
    cn_map = class_names_map or {}
    ids_by_canonical: dict[str, list[int]] = defaultdict(list)
    for old_id, raw_name in id_to_name.items():
        ids_by_canonical[normalize_class_name(raw_name, cn_map)].append(old_id)

    kept_canonical = sorted(c for c in ids_by_canonical if c in used_canonical)
    new_class_map: dict[str, int] = {}
    old_to_new: dict[int, int] = {}
    removed: list[str] = []

    for canonical in sorted(ids_by_canonical.keys()):
        if canonical not in used_canonical:
            for old_id in ids_by_canonical[canonical]:
                removed.append(id_to_name.get(old_id, canonical))

    for new_id, canonical in enumerate(kept_canonical):
        out_name = _canonical_to_output_name(canonical, cn_map)
        new_class_map[out_name] = new_id
        for old_id in ids_by_canonical[canonical]:
            old_to_new[old_id] = new_id

    return new_class_map, old_to_new, sorted(set(removed))


def _yolo_label_buckets(dataset_root: str, structure: str) -> list[tuple[str, str]]:
    if structure == "cvat11":
        buckets = yolo_flat_image_label_buckets(dataset_root)
        if buckets:
            return buckets
        img = os.path.join(dataset_root, "images")
        lbl = os.path.join(dataset_root, "labels")
        if os.path.isdir(img) and os.path.isdir(lbl):
            return [(img, lbl)]
        return []
    return find_dataset_paths(dataset_root, structure, arg=False)


def _iter_label_files(dataset_root: str, structure: str) -> list[str]:
    paths: list[str] = []
    for _img_dir, lbl_dir in _yolo_label_buckets(dataset_root, structure):
        root = Path(lbl_dir)
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.txt")):
            if p.is_file():
                paths.append(str(p))
    return paths


def collect_used_canonical_names_yolo(
    dataset_root: str,
    structure: str,
    id_to_name: dict[int, str],
    class_names_map: dict[str, str] | None,
) -> tuple[set[str], int]:
    cn_map = class_names_map or {}
    used: set[str] = set()
    unknown = 0
    for label_path in _iter_label_files(dataset_root, structure):
        for lb in read_yolo_labels(label_path):
            cls_id = int(lb.cls_id)
            raw_name = id_to_name.get(cls_id)
            if raw_name is None:
                unknown += 1
                continue
            used.add(normalize_class_name(raw_name, cn_map))
    return used, unknown


def collect_used_canonical_names_cvsdcldet(
    dataset_root: str,
    class_names_map: dict[str, str] | None,
) -> set[str]:
    cn_map = class_names_map or {}
    used: set[str] = set()
    for _img, jp in collect_cvsdcldet_pairs(Path(dataset_root)):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for det in data.get("detections") or []:
            if not isinstance(det, dict):
                continue
            label = _detection_label(det)
            if label:
                used.add(normalize_class_name(label, cn_map))
    return used


def _remap_yolo_label(label: YoloLabel, old_to_new: dict[int, int]) -> YoloLabel | None:
    new_id = old_to_new.get(int(label.cls_id))
    if new_id is None:
        return None
    if isinstance(label, YoloBBox):
        return YoloBBox(cls_id=new_id, cx=label.cx, cy=label.cy, w=label.w, h=label.h)
    return YoloSegment(cls_id=new_id, points=label.points)


def remap_yolo_labels(
    dataset_root: str,
    structure: str,
    old_to_new: dict[int, int],
    *,
    id_to_name: dict[int, str],
    class_names_map: dict[str, str] | None,
) -> tuple[int, int]:
    cn_map = class_names_map or {}
    remapped_files = 0
    unknown = 0
    for label_path in _iter_label_files(dataset_root, structure):
        labels = read_yolo_labels(label_path)
        if not labels:
            continue
        out: list[YoloLabel] = []
        changed = False
        for lb in labels:
            mapped = _remap_yolo_label(lb, old_to_new)
            if mapped is None:
                raw_name = id_to_name.get(int(lb.cls_id))
                if raw_name is None:
                    unknown += 1
                out.append(lb)
            else:
                if mapped.cls_id != lb.cls_id:
                    changed = True
                out.append(mapped)
        if changed:
            write_yolo_labels(label_path, out)
            remapped_files += 1
    return remapped_files, unknown


def remap_cvsdcldet_json(
    dataset_root: str,
    old_to_new: dict[int, int],
    *,
    id_to_name: dict[int, str],
    class_names_map: dict[str, str] | None,
    new_class_map: dict[str, int],
) -> int:
    cn_map = class_names_map or {}
    canonical_to_out = {normalize_class_name(n, cn_map): n for n in new_class_map}
    updated = 0
    name_to_new_id = {name: idx for name, idx in new_class_map.items()}

    for _img, jp in collect_cvsdcldet_pairs(Path(dataset_root)):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        detections = data.get("detections")
        if not isinstance(detections, list):
            continue
        file_changed = False
        for det in detections:
            if not isinstance(det, dict):
                continue
            label = _detection_label(det)
            if not label:
                continue
            canonical = normalize_class_name(label, cn_map)
            out_name = canonical_to_out.get(canonical, canonical)
            if out_name in name_to_new_id:
                new_id = name_to_new_id[out_name]
                if det.get("class_name") is not None:
                    if det.get("class_name") != out_name:
                        det["class_name"] = out_name
                        file_changed = True
                if "classId" in det:
                    try:
                        old_cid = int(det["classId"])
                    except (TypeError, ValueError):
                        old_cid = None
                    mapped = old_to_new.get(old_cid) if old_cid is not None else new_id
                    if mapped is not None and det.get("classId") != mapped:
                        det["classId"] = mapped
                        file_changed = True
        if file_changed:
            jp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            updated += 1
    return updated


def _split_images_rel(out_dir: str, split: str) -> str | None:
    for candidate in (split, "valid" if split == "val" else None):
        if candidate is None:
            continue
        p = Path(out_dir) / candidate / "images"
        if p.is_dir():
            return f"{candidate}/images"
    return None


def write_class_metadata(
    dataset_dir: str,
    names: list[str],
    *,
    structure: str,
) -> None:
    if structure == CVSDCLDET_STRUCTURE:
        yaml_path = os.path.join(dataset_dir, "data.yaml")
        train_rel = "."
        if os.path.isdir(os.path.join(dataset_dir, "images")):
            train_rel = "images"
        content = f"train: {train_rel}\nval: {train_rel}\ntest: {train_rel}\n\nnc: {len(names)}\nnames: {names}\n"
        Path(yaml_path).write_text(content, encoding="utf-8")
        return

    if structure == "darknet":
        obj_names_path = find_obj_names_file(dataset_dir)
        if obj_names_path:
            Path(obj_names_path).write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
        obj_data_path = find_obj_data_file(dataset_dir)
        if obj_data_path and os.path.isfile(obj_data_path):
            try:
                lines = Path(obj_data_path).read_text(encoding="utf-8").splitlines()
            except OSError:
                lines = []
            out_lines: list[str] = []
            replaced = False
            for line in lines:
                if line.strip().startswith("classes"):
                    out_lines.append(f"classes={len(names)}")
                    replaced = True
                else:
                    out_lines.append(line)
            if not replaced:
                out_lines.append(f"classes={len(names)}")
            Path(obj_data_path).write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    yaml_path = find_yaml_file(dataset_dir) or os.path.join(dataset_dir, "data.yaml")
    existing: dict[str, Any] = {}
    if os.path.isfile(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}

    if structure in ("flat", "subset_flat") or structure == "cvat11":
        train_rel = val_rel = test_rel = "images"
    elif structure == "nested_split":
        train_rel = _split_images_rel(dataset_dir, "train") or "images/train"
        val_rel = _split_images_rel(dataset_dir, "val") or _split_images_rel(dataset_dir, "valid") or train_rel
        test_rel = _split_images_rel(dataset_dir, "test") or val_rel
    else:
        train_rel = _split_images_rel(dataset_dir, "train") or "train/images"
        val_rel = _split_images_rel(dataset_dir, "val") or _split_images_rel(dataset_dir, "valid") or train_rel
        test_rel = _split_images_rel(dataset_dir, "test") or val_rel

    payload = dict(existing)
    payload.pop("path", None)
    payload["train"] = existing.get("train", train_rel)
    payload["val"] = existing.get("val", val_rel)
    payload["test"] = existing.get("test", test_rel)
    payload["nc"] = len(names)
    payload["names"] = names
    Path(yaml_path).write_text(yaml.dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def prepare_dataset_for_class_strip(
    dataset_root: str,
    structure: str,
    *,
    ensure_cb: Callable[[str], bool] | None = None,
) -> str:
    if structure == "cvat11" and ensure_cb is not None:
        ensure_cb(dataset_root)
    return structure


def strip_unused_classes(
    dataset_root: str,
    structure: str,
    info: dict[str, Any] | None = None,
    *,
    class_names_map: dict[str, str] | None = None,
    dry_run: bool = False,
    ensure_cb: Callable[[str], bool] | None = None,
) -> StripUnusedClassesStats:
    info = info or {}
    stats = StripUnusedClassesStats()

    if structure == "unknown":
        stats.skipped = True
        stats.skip_reason = "unknown structure"
        return stats

    structure = prepare_dataset_for_class_strip(dataset_root, structure, ensure_cb=ensure_cb)
    id_to_name = load_id_to_name(dataset_root, info=info)
    if not id_to_name:
        stats.skipped = True
        stats.skip_reason = "no class metadata found"
        return stats

    stats.classes_before = len({normalize_class_name(n, class_names_map) for n in id_to_name.values()})

    if structure == CVSDCLDET_STRUCTURE:
        used_canonical = collect_used_canonical_names_cvsdcldet(dataset_root, class_names_map)
        unknown = 0
    elif structure in YOLO_STRUCTURES:
        used_canonical, unknown = collect_used_canonical_names_yolo(
            dataset_root, structure, id_to_name, class_names_map
        )
    else:
        stats.skipped = True
        stats.skip_reason = f"unsupported structure {structure!r}"
        return stats

    stats.unknown_class_ids = unknown
    new_class_map, old_to_new, removed = build_compact_class_map(
        id_to_name, used_canonical, class_names_map
    )
    removed = sorted(set(removed))
    stats.removed_class_names = removed
    stats.new_class_map = new_class_map
    stats.classes_after = len(new_class_map)

    if not removed:
        return stats

    if dry_run:
        return stats

    if structure == CVSDCLDET_STRUCTURE:
        stats.labels_remapped = remap_cvsdcldet_json(
            dataset_root,
            old_to_new,
            id_to_name=id_to_name,
            class_names_map=class_names_map,
            new_class_map=new_class_map,
        )
    else:
        remapped, extra_unknown = remap_yolo_labels(
            dataset_root,
            structure,
            old_to_new,
            id_to_name=id_to_name,
            class_names_map=class_names_map,
        )
        stats.labels_remapped = remapped
        stats.unknown_class_ids += extra_unknown

    names_ordered = [k for k, _ in sorted(new_class_map.items(), key=lambda kv: kv[1])]
    write_class_metadata(dataset_root, names_ordered, structure=structure)
    return stats


def effective_structure_for_cleanup(dataset_root: str) -> str:
    return detect_structure(dataset_root)
