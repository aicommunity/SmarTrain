from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import yaml
from rich.console import Console
from rich.table import Table

from smartrain.cli_argparse import CliArgumentParser
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
console = Console()


@dataclass
class DatasetStats:
    name: str
    classes_by_id: dict[int, str]
    split_images: dict[str, int]
    split_instances: dict[str, int]
    per_class_split_instances: dict[str, dict[str, int]]
    per_class_images: dict[str, int]
    broken_label_lines: int
    unknown_class_ids: int
    orphan_images: int
    orphan_labels: int
    empty_images: int
    duplicate_groups: int = 0
    duplicate_files: int = 0
    duplicate_cross_split_groups: int = 0
    near_duplicate_groups: int = 0
    near_duplicate_cross_split_groups: int = 0

    @property
    def images_total(self) -> int:
        return sum(self.split_images.values())

    @property
    def instances_total(self) -> int:
        return sum(self.split_instances.values())

    @property
    def labeled_images(self) -> int:
        return self.images_total - self.empty_images


def _safe_read_yaml(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _classes_from_data_yaml(dataset_dir: str) -> dict[int, str]:
    raw = _safe_read_yaml(os.path.join(dataset_dir, "data.yaml"))
    names = raw.get("names")
    out: dict[int, str] = {}
    if isinstance(names, dict):
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
    elif isinstance(names, list):
        for i, v in enumerate(names):
            out[i] = str(v)
    return out


def _list_stems(folder: str, *, image_mode: bool) -> set[str]:
    out: set[str] = set()
    if not os.path.isdir(folder):
        return out
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if not os.path.isfile(p):
            continue
        stem, ext = os.path.splitext(name)
        if image_mode:
            if ext.lower() in IMAGE_EXTS:
                out.add(stem)
        else:
            if ext.lower() == ".txt":
                out.add(stem)
    return out


def _resolve_label_path(labels_dir: str, stem: str) -> str:
    return os.path.join(labels_dir, f"{stem}.txt")


def _parse_label_file(path: str) -> tuple[int, int, list[int]]:
    if not os.path.isfile(path):
        return 0, 0, []
    valid = 0
    broken = 0
    ids: list[int] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return 0, 1, []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            broken += 1
            continue
        try:
            class_id = int(float(parts[0]))
            _ = [float(x) for x in parts[1:5]]
        except ValueError:
            broken += 1
            continue
        valid += 1
        ids.append(class_id)
    return valid, broken, ids


def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _average_hash(path: str) -> int:
    from PIL import Image

    img = Image.open(path).convert("L").resize((8, 8))
    pixels = list(img.getdata())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for px in pixels:
        bits = (bits << 1) | (1 if px >= mean else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _scan_one_dataset(dataset_dir: str, name: str) -> DatasetStats:
    classes_by_id = _classes_from_data_yaml(dataset_dir)
    split_images = {s: 0 for s in SPLITS}
    split_instances = {s: 0 for s in SPLITS}
    per_class_split_instances: dict[str, dict[str, int]] = defaultdict(
        lambda: {s: 0 for s in SPLITS}
    )
    per_class_images: dict[str, int] = defaultdict(int)
    broken_label_lines = 0
    unknown_class_ids = 0
    orphan_images = 0
    orphan_labels = 0
    empty_images = 0

    def _scan_pair(images_dir: str, labels_dir: str, split: str) -> None:
        nonlocal broken_label_lines, unknown_class_ids, orphan_images, orphan_labels, empty_images
        image_stems = _list_stems(images_dir, image_mode=True)
        label_stems = _list_stems(labels_dir, image_mode=False)

        split_images[split] += len(image_stems)
        orphan_images += len(image_stems - label_stems)
        orphan_labels += len(label_stems - image_stems)

        for stem in image_stems:
            label_path = _resolve_label_path(labels_dir, stem)
            valid, broken, ids = _parse_label_file(label_path)
            broken_label_lines += broken
            split_instances[split] += valid
            if valid == 0:
                empty_images += 1
                continue
            used_classes_this_image: set[str] = set()
            for class_id in ids:
                cls_name = classes_by_id.get(class_id, f"id_{class_id}")
                if class_id not in classes_by_id:
                    unknown_class_ids += 1
                per_class_split_instances[cls_name][split] += 1
                used_classes_this_image.add(cls_name)
            for cls_name in used_classes_this_image:
                per_class_images[cls_name] += 1

    has_split_dirs = any(os.path.isdir(os.path.join(dataset_dir, s)) for s in SPLITS)
    if has_split_dirs:
        for split in SPLITS:
            split_dir = os.path.join(dataset_dir, split)
            _scan_pair(
                os.path.join(split_dir, "images"),
                os.path.join(split_dir, "labels"),
                split,
            )
    elif os.path.isdir(os.path.join(dataset_dir, "images")) and os.path.isdir(
        os.path.join(dataset_dir, "labels")
    ):
        # flat dataset -> считаем как train bucket.
        _scan_pair(
            os.path.join(dataset_dir, "images"),
            os.path.join(dataset_dir, "labels"),
            "train",
        )
    elif os.path.isfile(os.path.join(dataset_dir, "annotations.xml")) and os.path.isdir(
        os.path.join(dataset_dir, "images")
    ):
        # cvat11 dataset -> считаем объекты из annotations.xml, split=train.
        images_dir = os.path.join(dataset_dir, "images")
        image_files = [
            n
            for n in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, n))
            and os.path.splitext(n)[1].lower() in IMAGE_EXTS
        ]
        split_images["train"] += len(image_files)
        image_to_boxes: dict[str, int] = {}
        try:
            root = ET.parse(os.path.join(dataset_dir, "annotations.xml")).getroot()
            known_labels = set(classes_by_id.values())
            for image_node in root.findall(".//image"):
                img_name = str(image_node.attrib.get("name", "")).strip()
                boxes = image_node.findall("box")
                image_to_boxes[img_name] = len(boxes)
                used_classes_this_image: set[str] = set()
                for box in boxes:
                    label = str(box.attrib.get("label", "")).strip()
                    if not label:
                        broken_label_lines += 1
                        continue
                    if known_labels and label not in known_labels:
                        unknown_class_ids += 1
                    per_class_split_instances[label]["train"] += 1
                    split_instances["train"] += 1
                    used_classes_this_image.add(label)
                for cls_name in used_classes_this_image:
                    per_class_images[cls_name] += 1
        except Exception:
            broken_label_lines += 1
        for fname in image_files:
            if image_to_boxes.get(fname, 0) == 0:
                empty_images += 1
    else:
        # Unknown structure in datasets/: mark as warning signal.
        broken_label_lines += 1

    return DatasetStats(
        name=name,
        classes_by_id=classes_by_id,
        split_images=split_images,
        split_instances=split_instances,
        per_class_split_instances=dict(per_class_split_instances),
        per_class_images=dict(per_class_images),
        broken_label_lines=broken_label_lines,
        unknown_class_ids=unknown_class_ids,
        orphan_images=orphan_images,
        orphan_labels=orphan_labels,
        empty_images=empty_images,
    )


def _scan_duplicates(
    dataset_dir: str, *, check_near: bool = False
) -> tuple[int, int, int, int, int]:
    img_paths: list[tuple[str, str]] = []
    split_dirs_found = False
    for split in SPLITS:
        images_dir = os.path.join(dataset_dir, split, "images")
        if not os.path.isdir(images_dir):
            continue
        split_dirs_found = True
        for name in os.listdir(images_dir):
            p = os.path.join(images_dir, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                img_paths.append((split, p))
    if not split_dirs_found:
        flat_images = os.path.join(dataset_dir, "images")
        if os.path.isdir(flat_images):
            for name in os.listdir(flat_images):
                p = os.path.join(flat_images, name)
                if os.path.isfile(p) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                    img_paths.append(("train", p))

    by_md5: dict[tuple[int, str], list[tuple[str, str]]] = defaultdict(list)
    for split, path in img_paths:
        try:
            key = (os.path.getsize(path), _file_md5(path))
            by_md5[key].append((split, path))
        except Exception:
            continue

    dup_groups = 0
    dup_files = 0
    dup_cross_split = 0
    for members in by_md5.values():
        if len(members) < 2:
            continue
        dup_groups += 1
        dup_files += len(members)
        if len({s for s, _ in members}) > 1:
            dup_cross_split += 1

    near_groups = 0
    near_cross_split = 0
    if check_near:
        hashes: list[tuple[str, int]] = []
        for split, path in img_paths:
            try:
                hashes.append((split, _average_hash(path)))
            except Exception:
                continue
        used = set()
        for i in range(len(hashes)):
            if i in used:
                continue
            group = [i]
            for j in range(i + 1, len(hashes)):
                if _hamming(hashes[i][1], hashes[j][1]) <= 5:
                    group.append(j)
            if len(group) > 1:
                near_groups += 1
                if len({hashes[k][0] for k in group}) > 1:
                    near_cross_split += 1
                used.update(group)
    return dup_groups, dup_files, dup_cross_split, near_groups, near_cross_split


def _gini(values: list[int]) -> float:
    vals = [v for v in values if v >= 0]
    if not vals:
        return 0.0
    total = sum(vals)
    if total == 0:
        return 0.0
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    cum = 0
    for i, val in enumerate(vals_sorted, start=1):
        cum += i * val
    return (2 * cum) / (n * total) - (n + 1) / n


def _imbalance_summary(class_totals: dict[str, int]) -> dict[str, float]:
    vals = [v for v in class_totals.values() if v > 0]
    if not vals:
        return {"ratio": 0.0, "cv": 0.0, "gini": 0.0, "mean_median": 0.0}
    ratio = max(vals) / min(vals) if min(vals) > 0 else 0.0
    mean = statistics.mean(vals)
    cv = (statistics.pstdev(vals) / mean) if mean > 0 else 0.0
    median = statistics.median(vals)
    mean_median = (mean / median) if median > 0 else 0.0
    return {"ratio": ratio, "cv": cv, "gini": _gini(vals), "mean_median": mean_median}


def _available_dataset_dirs(layout: WorkspaceLayout) -> dict[str, str]:
    out: dict[str, str] = {}
    if not os.path.isdir(layout.datasets):
        return out
    for name in sorted(os.listdir(layout.datasets)):
        path = os.path.join(layout.datasets, name)
        if not os.path.isdir(path):
            continue
        out[name] = path
    return out


def _filter_dataset_names(available: dict[str, str], selected: list[str] | None) -> list[str]:
    if not selected:
        return sorted(available.keys())
    unknown = [x for x in selected if x not in available]
    if unknown:
        raise ValueError(
            f"Неизвестные датасеты: {', '.join(unknown)}. Доступные: {', '.join(sorted(available.keys()))}"
        )
    return sorted(dict.fromkeys(selected).keys())


def _collect_available_classes(available: dict[str, str]) -> list[str]:
    classes: set[str] = set()
    for ds_path in available.values():
        id_to_name = _classes_from_data_yaml(ds_path)
        classes.update(id_to_name.values())
    return sorted(classes)


def _print_interactive_catalog(available: dict[str, str]) -> None:
    names = sorted(available.keys())
    class_names = _collect_available_classes(available)
    console.print("[INFO] Доступные датасеты:")
    for name in names:
        console.print(f"  - {name}")
    console.print("[INFO] Доступные классы:")
    if class_names:
        for cls in class_names:
            console.print(f"  - {cls}")
    else:
        console.print("  - (не найдены в data.yaml)")


def _collect_class_ids_by_name(
    selected_names: list[str], scanned: dict[str, DatasetStats]
) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for ds_name in selected_names:
        ds = scanned[ds_name]
        for cid, cname in ds.classes_by_id.items():
            out[str(cname)].add(int(cid))
    return out


def _format_class_id_set(ids: set[int]) -> str:
    if not ids:
        return "-"
    s = sorted(ids)
    if len(s) == 1:
        return str(s[0])
    return f"mixed({','.join(str(x) for x in s)})"


def _render_datasets_class_catalog(
    selected_names: list[str], scanned: dict[str, DatasetStats], *, show_legend: bool
) -> None:
    class_ids = _collect_class_ids_by_name(selected_names, scanned)
    rows: list[tuple[str, str, str, str, str]] = []
    for cls in sorted(class_ids.keys()):
        present_in = 0
        total_instances = 0
        for ds_name in selected_names:
            ds = scanned[ds_name]
            cnt = int(sum(ds.per_class_split_instances.get(cls, {}).values()))
            total_instances += cnt
            if cnt > 0:
                present_in += 1
        coverage = f"{present_in}/{len(selected_names)}"
        for ds_name in selected_names:
            ds = scanned[ds_name]
            ids_here = sorted([cid for cid, cname in ds.classes_by_id.items() if cname == cls])
            if not ids_here:
                continue
            class_id = "/".join(str(x) for x in ids_here)
            rows.append((class_id, cls, coverage, str(total_instances), ds_name))
    table = Table(title="Каталог классов по датасетам")
    table.add_column("ClassID", justify="right")
    table.add_column("ClassName")
    table.add_column("Coverage", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Dataset")
    last_class_name = None
    for class_id, class_name, coverage, total, dataset in rows:
        display_class_name = class_name if class_name != last_class_name else ""
        table.add_row(class_id, display_class_name, coverage, total, dataset)
        last_class_name = class_name
    console.print(table)
    if show_legend:
        console.print("[dim]Колонки каталога классов по датасетам:[/dim]")
        console.print("[dim]- ClassID: индекс класса в конкретном датасете[/dim]")
        console.print("[dim]- ClassName: имя класса (показывается в первой строке группы)[/dim]")
        console.print("[dim]- Coverage: в скольких выбранных датасетах класс имеет объекты (present/total)[/dim]")
        console.print("[dim]- Total: общее число объектов класса по выбранным датасетам[/dim]")
        console.print("[dim]- Dataset: имя датасета для строки[/dim]")


def _render_classes_table(
    selected_names: list[str],
    scanned: dict[str, DatasetStats],
    classes_filter: set[str] | None,
    sort_key: str,
    desc: bool,
    limit: int | None,
    show_legend: bool,
) -> None:
    class_rows: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in SPLITS})
    class_images: dict[str, int] = defaultdict(int)
    class_ids_by_name = _collect_class_ids_by_name(selected_names, scanned)
    for name in selected_names:
        ds = scanned[name]
        for cls, split_map in ds.per_class_split_instances.items():
            for split in SPLITS:
                class_rows[cls][split] += split_map.get(split, 0)
        for cls, val in ds.per_class_images.items():
            class_images[cls] += val

    if classes_filter:
        class_rows = defaultdict(
            lambda: {s: 0 for s in SPLITS},
            {k: v for k, v in class_rows.items() if k in classes_filter},
        )
        class_images = defaultdict(int, {k: v for k, v in class_images.items() if k in classes_filter})

    rows = []
    split_total = {s: 0 for s in SPLITS}
    for cls, split_map in class_rows.items():
        total = sum(split_map.values())
        for s in SPLITS:
            split_total[s] += split_map[s]
        avg_per_image = total / class_images[cls] if class_images.get(cls, 0) else 0.0
        row = {
            "class_id": _format_class_id_set(class_ids_by_name.get(cls, set())),
            "class": cls,
            "train": split_map["train"],
            "val": split_map["val"],
            "test": split_map["test"],
            "total": total,
            "images": class_images.get(cls, 0),
            "avg": avg_per_image,
        }
        rows.append(row)
    rows.sort(key=lambda r: r.get(sort_key, r["total"]), reverse=desc)
    if limit is not None and limit > 0:
        rows = rows[:limit]

    table = Table(title="Статистика по классам")
    table.add_column("ClassID", justify="right")
    table.add_column("Class")
    table.add_column("Train", justify="right")
    table.add_column("Val", justify="right")
    table.add_column("Test", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Images", justify="right")
    table.add_column("Avg/Img", justify="right")
    for r in rows:
        table.add_row(
            r["class_id"],
            r["class"],
            str(r["train"]),
            str(r["val"]),
            str(r["test"]),
            str(r["total"]),
            str(r["images"]),
            f"{r['avg']:.2f}",
        )
    console.print(table)
    if show_legend:
        console.print("[dim]Колонки classes:[/dim]")
        console.print("[dim]- ClassID: индекс класса (или mixed(...), если индексы отличаются между датасетами)[/dim]")
        console.print("[dim]- Class: имя класса[/dim]")
        console.print("[dim]- Train: число объектов класса в train[/dim]")
        console.print("[dim]- Val: число объектов класса в val[/dim]")
        console.print("[dim]- Test: число объектов класса в test[/dim]")
        console.print("[dim]- Total: суммарно по всем split[/dim]")
        console.print("[dim]- Images: число изображений, где класс встречается[/dim]")
        console.print(
            "[dim]- Avg/Img: среднее число объектов класса на изображение с этим классом[/dim]"
        )

    summary_total = _imbalance_summary({k: sum(v.values()) for k, v in class_rows.items()})
    summary_train = _imbalance_summary({k: v["train"] for k, v in class_rows.items()})
    summary_val = _imbalance_summary({k: v["val"] for k, v in class_rows.items()})
    summary_test = _imbalance_summary({k: v["test"] for k, v in class_rows.items()})
    console.print("[bold]Итог по дисбалансу[/bold]")
    for label, m in (
        ("train", summary_train),
        ("val", summary_val),
        ("test", summary_test),
        ("total", summary_total),
    ):
        console.print(
            f"- {label}: ratio={m['ratio']:.3f}, cv={m['cv']:.3f}, gini={m['gini']:.3f}, mean/median={m['mean_median']:.3f}"
        )


def _class_totals(ds: DatasetStats) -> dict[str, int]:
    return {k: int(sum(v.values())) for k, v in ds.per_class_split_instances.items()}


def compare_dataset_stats(left: DatasetStats, right: DatasetStats) -> dict:
    left_classes = _class_totals(left)
    right_classes = _class_totals(right)
    left_set = set(left_classes.keys())
    right_set = set(right_classes.keys())
    common = sorted(left_set & right_set)
    rows = []
    for cls in common:
        l_split = left.per_class_split_instances.get(cls, {})
        r_split = right.per_class_split_instances.get(cls, {})
        l_total = left_classes.get(cls, 0)
        r_total = right_classes.get(cls, 0)
        rows.append(
            {
                "class": cls,
                "left_total": l_total,
                "right_total": r_total,
                "delta_total": r_total - l_total,
                "left_train": int(l_split.get("train", 0)),
                "right_train": int(r_split.get("train", 0)),
                "delta_train": int(r_split.get("train", 0)) - int(l_split.get("train", 0)),
                "left_val": int(l_split.get("val", 0)),
                "right_val": int(r_split.get("val", 0)),
                "delta_val": int(r_split.get("val", 0)) - int(l_split.get("val", 0)),
                "left_test": int(l_split.get("test", 0)),
                "right_test": int(r_split.get("test", 0)),
                "delta_test": int(r_split.get("test", 0)) - int(l_split.get("test", 0)),
            }
        )
    rows.sort(key=lambda r: abs(int(r["delta_total"])), reverse=True)
    left_m = _imbalance_summary(left_classes)
    right_m = _imbalance_summary(right_classes)
    left_empty_pct = (100.0 * left.empty_images / left.images_total) if left.images_total else 0.0
    right_empty_pct = (100.0 * right.empty_images / right.images_total) if right.images_total else 0.0
    left_q_ok = (
        left.broken_label_lines == 0
        and left.unknown_class_ids == 0
        and left.orphan_images == 0
        and left.orphan_labels == 0
    )
    right_q_ok = (
        right.broken_label_lines == 0
        and right.unknown_class_ids == 0
        and right.orphan_images == 0
        and right.orphan_labels == 0
    )
    return {
        "summary": {
            "left": {
                "name": left.name,
                "images": left.images_total,
                "instances": left.instances_total,
                "empty_pct": left_empty_pct,
                "imbalance_ratio": left_m["ratio"],
                "gini": left_m["gini"],
                "quality_ok": left_q_ok,
            },
            "right": {
                "name": right.name,
                "images": right.images_total,
                "instances": right.instances_total,
                "empty_pct": right_empty_pct,
                "imbalance_ratio": right_m["ratio"],
                "gini": right_m["gini"],
                "quality_ok": right_q_ok,
            },
            "delta": {
                "images": right.images_total - left.images_total,
                "instances": right.instances_total - left.instances_total,
                "empty_pct": right_empty_pct - left_empty_pct,
                "imbalance_ratio": right_m["ratio"] - left_m["ratio"],
                "gini": right_m["gini"] - left_m["gini"],
            },
            "issues": {
                "broken_label_lines": right.broken_label_lines - left.broken_label_lines,
                "unknown_class_ids": right.unknown_class_ids - left.unknown_class_ids,
                "orphan_images": right.orphan_images - left.orphan_images,
                "orphan_labels": right.orphan_labels - left.orphan_labels,
                "duplicates": right.duplicate_groups - left.duplicate_groups,
                "near_duplicates": right.near_duplicate_groups - left.near_duplicate_groups,
            },
        },
        "classes": {
            "left_only": sorted(left_set - right_set),
            "right_only": sorted(right_set - left_set),
            "common": rows,
        },
    }


def _render_compare_summary(report: dict, *, abs_values: bool = False) -> None:
    s = report["summary"]
    table = Table(title=f"Сравнение датасетов: {s['left']['name']} vs {s['right']['name']}")
    table.add_column("Metric")
    table.add_column("Left", justify="right")
    table.add_column("Right", justify="right")
    table.add_column("Delta(R-L)", justify="right")
    for key in ("images", "instances", "empty_pct", "imbalance_ratio", "gini"):
        lv = s["left"][key]
        rv = s["right"][key]
        dv = s["delta"][key]
        if abs_values:
            dv = abs(dv)
        if isinstance(lv, float):
            table.add_row(key, f"{lv:.3f}", f"{rv:.3f}", f"{dv:.3f}")
        else:
            table.add_row(key, str(lv), str(rv), str(dv))
    table.add_row("quality_ok", str(s["left"]["quality_ok"]), str(s["right"]["quality_ok"]), "-")
    console.print(table)
    issues = s["issues"]
    console.print(
        "[bold]Issues delta (R-L):[/bold] "
        f"broken={issues['broken_label_lines']}, unknown={issues['unknown_class_ids']}, "
        f"orphan_i={issues['orphan_images']}, orphan_l={issues['orphan_labels']}, "
        f"dup={issues['duplicates']}, near_dup={issues['near_duplicates']}"
    )


def _render_compare_classes(report: dict, *, top_n: int | None = None, abs_values: bool = False) -> None:
    left_only = report["classes"]["left_only"]
    right_only = report["classes"]["right_only"]
    if left_only:
        console.print(f"[bold]Только в left:[/bold] {', '.join(left_only)}")
    if right_only:
        console.print(f"[bold]Только в right:[/bold] {', '.join(right_only)}")
    rows = list(report["classes"]["common"])
    if top_n is not None and top_n > 0:
        rows = rows[:top_n]
    table = Table(title="Diff по общим классам")
    table.add_column("Class")
    table.add_column("L_total", justify="right")
    table.add_column("R_total", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("L_train", justify="right")
    table.add_column("R_train", justify="right")
    table.add_column("L_val", justify="right")
    table.add_column("R_val", justify="right")
    table.add_column("L_test", justify="right")
    table.add_column("R_test", justify="right")
    for r in rows:
        d = int(r["delta_total"])
        if abs_values:
            d = abs(d)
        table.add_row(
            r["class"],
            str(r["left_total"]),
            str(r["right_total"]),
            str(d),
            str(r["left_train"]),
            str(r["right_train"]),
            str(r["left_val"]),
            str(r["right_val"]),
            str(r["left_test"]),
            str(r["right_test"]),
        )
    console.print(table)


def _export_compare_report(layout: WorkspaceLayout, report: dict, *, export_json: bool, export_csv: bool) -> list[str]:
    out_paths: list[str] = []
    if not (export_json or export_csv):
        return out_paths
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(layout.analytics, "stats", stamp)
    os.makedirs(out_dir, exist_ok=True)
    if export_json:
        jp = os.path.join(out_dir, "compare_report.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        out_paths.append(jp)
    if export_csv:
        cp = os.path.join(out_dir, "compare_classes.csv")
        with open(cp, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(
                f,
                fieldnames=[
                    "class",
                    "left_total",
                    "right_total",
                    "delta_total",
                    "left_train",
                    "right_train",
                    "delta_train",
                    "left_val",
                    "right_val",
                    "delta_val",
                    "left_test",
                    "right_test",
                    "delta_test",
                ],
            )
            wr.writeheader()
            for row in report["classes"]["common"]:
                wr.writerow(row)
        out_paths.append(cp)
    return out_paths
def _export_issues(layout: WorkspaceLayout, rows: list[dict[str, str]]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(layout.analytics, "stats", stamp)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "issues_datasets.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "broken_label_lines",
                "unknown_class_ids",
                "orphan_images",
                "orphan_labels",
                "empty_images",
                "duplicate_groups",
                "duplicate_cross_split_groups",
                "near_duplicate_groups",
                "near_duplicate_cross_split_groups",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def _render_datasets_table(
    selected_names: list[str], scanned: dict[str, DatasetStats], args, layout: WorkspaceLayout
) -> None:
    rows = []
    issues_rows: list[dict[str, str]] = []
    for name in selected_names:
        ds = scanned[name]
        class_totals = {
            k: sum(v.values()) for k, v in ds.per_class_split_instances.items() if sum(v.values()) > 0
        }
        m = _imbalance_summary(class_totals)
        empty_pct = (100.0 * ds.empty_images / ds.images_total) if ds.images_total else 0.0
        quality_ok = (
            ds.broken_label_lines == 0
            and ds.unknown_class_ids == 0
            and ds.orphan_images == 0
            and ds.orphan_labels == 0
        )
        row = {
            "dataset": name,
            "classes": len(class_totals),
            "images": ds.images_total,
            "labeled": ds.labeled_images,
            "empty": ds.empty_images,
            "empty_pct": empty_pct,
            "instances": ds.instances_total,
            "imbalance": m["ratio"],
            "gini": m["gini"],
            "quality": "OK" if quality_ok else "WARN",
            "broken": ds.broken_label_lines,
            "unknown": ds.unknown_class_ids,
            "orphan_i": ds.orphan_images,
            "orphan_l": ds.orphan_labels,
            "dup": ds.duplicate_groups,
            "dup_cross": ds.duplicate_cross_split_groups,
            "near_dup": ds.near_duplicate_groups,
            "near_dup_cross": ds.near_duplicate_cross_split_groups,
        }
        rows.append(row)
        issues_rows.append(
            {
                "dataset": name,
                "broken_label_lines": str(ds.broken_label_lines),
                "unknown_class_ids": str(ds.unknown_class_ids),
                "orphan_images": str(ds.orphan_images),
                "orphan_labels": str(ds.orphan_labels),
                "empty_images": str(ds.empty_images),
                "duplicate_groups": str(ds.duplicate_groups),
                "duplicate_cross_split_groups": str(ds.duplicate_cross_split_groups),
                "near_duplicate_groups": str(ds.near_duplicate_groups),
                "near_duplicate_cross_split_groups": str(ds.near_duplicate_cross_split_groups),
            }
        )

    key = args.sort
    rows.sort(key=lambda r: r.get(key, 0), reverse=bool(args.desc))

    table = Table(title="Статистика по датасетам")
    for col in ("Dataset", "Classes", "Images", "Labeled", "Empty", "Empty%", "Instances", "Imbalance", "Gini", "Quality"):
        table.add_column(col, justify="right" if col != "Dataset" else "left")
    for r in rows:
        table.add_row(
            r["dataset"],
            str(r["classes"]),
            str(r["images"]),
            str(r["labeled"]),
            str(r["empty"]),
            f"{r['empty_pct']:.2f}",
            str(r["instances"]),
            f"{r['imbalance']:.3f}",
            f"{r['gini']:.3f}",
            r["quality"],
        )
    console.print(table)
    if not getattr(args, "no_legend", False):
        console.print("[dim]Колонки datasets:[/dim]")
        console.print("[dim]- Dataset: имя датасета[/dim]")
        console.print("[dim]- Classes: число классов с объектами[/dim]")
        console.print("[dim]- Images: число изображений[/dim]")
        console.print("[dim]- Labeled: изображения с >=1 объектом[/dim]")
        console.print("[dim]- Empty: изображения без объектов[/dim]")
        console.print("[dim]- Empty%: доля Empty от Images[/dim]")
        console.print("[dim]- Instances: число всех объектов[/dim]")
        console.print("[dim]- Imbalance: max/min по числу объектов на класс[/dim]")
        console.print("[dim]- Gini: коэффициент Джини по распределению классов[/dim]")
        console.print("[dim]- Quality: OK/WARN по базовым проверкам разметки[/dim]")
    _render_datasets_class_catalog(
        selected_names,
        scanned,
        show_legend=not getattr(args, "no_legend", False),
    )

    if any((args.check_duplicates, args.check_near_duplicates)):
        console.print("[bold]Leakage/Duplicate risk[/bold]")
        for r in rows:
            console.print(
                f"- {r['dataset']}: dup_groups={r['dup']}, cross_split={r['dup_cross']}, "
                f"near_dup_groups={r['near_dup']}, near_cross_split={r['near_dup_cross']}"
            )

    if args.export_issues:
        out_path = _export_issues(layout, issues_rows)
        console.print(f"[OK] Экспорт проблемных файлов: {out_path}")


def _prompt_yes_no(label: str, default: bool = False) -> bool:
    from prompt_toolkit import prompt

    suffix = "Y/n" if default else "y/N"
    default_text = "y" if default else "n"
    val = prompt(f"{label} [{suffix}]: ", default=default_text).strip().lower()
    if not val:
        return default
    return val in ("y", "yes", "1", "true", "да", "д")


def _prompt_interactive_classes(
    args, available_names: list[str], available_classes: list[str]
) -> None:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import WordCompleter

    names_comp = WordCompleter(available_names, ignore_case=True)
    classes_comp = WordCompleter(available_classes, ignore_case=True)
    sort_comp = WordCompleter(["total", "train", "val", "test"], ignore_case=True)
    raw_ds = prompt(
        "Датасеты через запятую (пусто=все): ",
        default="",
        completer=names_comp,
        complete_while_typing=True,
    ).strip()
    args.dataset = [x.strip() for x in raw_ds.split(",") if x.strip()] or None
    args.classes = (
        prompt(
            "Классы через запятую (пусто=все): ",
            default="",
            completer=classes_comp,
            complete_while_typing=True,
        ).strip()
        or None
    )
    args.sort = (
        prompt(
            "Сортировка (total/train/val/test): ",
            default=args.sort,
            completer=sort_comp,
            complete_while_typing=True,
        ).strip()
        or args.sort
    )
    args.desc = _prompt_yes_no("Сортировка по убыванию", default=bool(args.desc))
    lim_raw = prompt("Лимит строк (пусто=без лимита): ", default="").strip()
    args.limit = int(lim_raw) if lim_raw else None


def _prompt_interactive_datasets(args, available_names: list[str]) -> None:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import WordCompleter

    names_comp = WordCompleter(available_names, ignore_case=True)
    sort_comp = WordCompleter(
        ["images", "instances", "empty_pct", "gini", "imbalance"], ignore_case=True
    )
    raw_ds = prompt(
        "Датасеты через запятую (пусто=все): ",
        default="",
        completer=names_comp,
        complete_while_typing=True,
    ).strip()
    args.dataset = [x.strip() for x in raw_ds.split(",") if x.strip()] or None
    args.sort = (
        prompt(
            "Сортировка (images/instances/empty_pct/gini/imbalance): ",
            default=args.sort,
            completer=sort_comp,
            complete_while_typing=True,
        ).strip()
        or args.sort
    )
    args.desc = _prompt_yes_no("Сортировка по убыванию", default=bool(args.desc))
    args.check_duplicates = _prompt_yes_no(
        "Проверять exact duplicates (--check-duplicates)",
        default=bool(args.check_duplicates),
    )
    args.check_near_duplicates = _prompt_yes_no(
        "Проверять near-duplicates (--check-near-duplicates)",
        default=bool(args.check_near_duplicates),
    )
    args.export_issues = _prompt_yes_no(
        "Экспортировать issues в analytics (--export-issues)",
        default=bool(args.export_issues),
    )


def prompt_interactive_compare_args(args, available_names: list[str]) -> None:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import WordCompleter

    names_comp = WordCompleter(available_names, ignore_case=True)
    details_comp = WordCompleter(["summary", "classes", "all"], ignore_case=True)
    while True:
        left = prompt("Left dataset: ", default=str(getattr(args, "left", "") or ""), completer=names_comp, complete_while_typing=True).strip()
        right = prompt("Right dataset: ", default=str(getattr(args, "right", "") or ""), completer=names_comp, complete_while_typing=True).strip()
        if left and right and left != right:
            args.left = left
            args.right = right
            break
        console.print("[ERROR] Нужно задать разные left/right датасеты.")
    args.details = (
        prompt(
            "Details (summary/classes/all): ",
            default=str(getattr(args, "details", "summary")),
            completer=details_comp,
            complete_while_typing=True,
        ).strip()
        or "summary"
    )
    raw_top = prompt("Top-N классов (пусто=без лимита): ", default="").strip()
    args.top_n = int(raw_top) if raw_top else None
    args.abs = _prompt_yes_no("Показывать абсолютные дельты (--abs)", default=bool(getattr(args, "abs", False)))
    args.export_json = _prompt_yes_no("Экспорт JSON (--export-json)", default=bool(getattr(args, "export_json", False)))
    args.export_csv = _prompt_yes_no("Экспорт CSV (--export-csv)", default=bool(getattr(args, "export_csv", False)))


def build_stats_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(description="Единая статистика датасетов и классов (только datasets/)")
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Корень workspace (иначе {WORKSPACE_ENV_VAR}); анализируется только datasets/",
    )
    parser.add_argument("--dataset", action="append", default=None, help="Имя датасета в datasets/ (повторяемый)")
    parser.add_argument("--classes", type=str, default=None, help="Фильтр классов через запятую (для таблиц классов)")
    parser.add_argument(
        "--sort",
        choices=("images", "instances", "empty_pct", "gini", "imbalance"),
        default="images",
        help="Сортировка основной таблицы датасетов",
    )
    parser.add_argument("--desc", action="store_true", help="Сортировка основной таблицы по убыванию")
    parser.add_argument("--check-duplicates", action="store_true")
    parser.add_argument("--check-near-duplicates", action="store_true")
    parser.add_argument("--export-issues", action="store_true")
    parser.add_argument("--no-legend", action="store_true", help="Не выводить расшифровку колонок")
    parser.add_argument(
        "--class-sort",
        choices=("total", "train", "val", "test"),
        default="total",
        help="Сортировка таблицы классов",
    )
    parser.add_argument("--class-desc", action="store_true", help="Сортировка таблицы классов по убыванию")
    parser.add_argument("--class-limit", type=int, default=None, help="Лимит строк в таблицах классов")
    return parser


def build_stats_compare_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(description="Сравнение двух датасетов из datasets/")
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Корень workspace (иначе {WORKSPACE_ENV_VAR}); анализируется только datasets/",
    )
    parser.add_argument("--left", type=str, default=None, help="Левый датасет (baseline)")
    parser.add_argument("--right", type=str, default=None, help="Правый датасет (candidate)")
    parser.add_argument("--details", choices=("summary", "classes", "all"), default="summary")
    parser.add_argument("--top-n", type=int, default=None, help="Лимит строк для class diff")
    parser.add_argument("--abs", action="store_true", help="Показывать абсолютные дельты")
    parser.add_argument("--export-json", action="store_true", help="Экспорт отчёта compare в JSON")
    parser.add_argument("--export-csv", action="store_true", help="Экспорт class diff в CSV")
    parser.add_argument("--no-legend", action="store_true", help="Не выводить расшифровку колонок")
    return parser


def _run_stats(args, layout: WorkspaceLayout) -> int:
    available = _available_dataset_dirs(layout)
    if not available:
        console.print("[ERROR] В datasets/ не найдено ни одного датасета.")
        return 2
    interactive = (
        not args.dataset
        and args.sort == "images"
        and not args.desc
        and not args.check_duplicates
        and not args.check_near_duplicates
        and not args.export_issues
    )
    if interactive and sys.stdin.isatty():
        console.print("[INFO] Интерактивный режим stats")
        _print_interactive_catalog(available)
        _prompt_interactive_datasets(args, sorted(available.keys()))
    selected = _filter_dataset_names(available, args.dataset)
    scanned: dict[str, DatasetStats] = {}
    for name in selected:
        ds = _scan_one_dataset(available[name], name)
        if args.check_duplicates or args.check_near_duplicates:
            (
                ds.duplicate_groups,
                ds.duplicate_files,
                ds.duplicate_cross_split_groups,
                ds.near_duplicate_groups,
                ds.near_duplicate_cross_split_groups,
            ) = _scan_duplicates(available[name], check_near=bool(args.check_near_duplicates))
        scanned[name] = ds
    _render_datasets_table(selected, scanned, args, layout)
    classes_filter = None
    if args.classes:
        classes_filter = {x.strip() for x in args.classes.split(",") if x.strip()}
    _render_classes_table(
        selected,
        scanned,
        classes_filter,
        args.class_sort,
        bool(args.class_desc),
        args.class_limit,
        show_legend=not bool(getattr(args, "no_legend", False)),
    )
    return 0


def _run_stats_compare(args, layout: WorkspaceLayout) -> int:
    available = _available_dataset_dirs(layout)
    if not available:
        console.print("[ERROR] В datasets/ не найдено ни одного датасета.")
        return 2
    if (not args.left or not args.right) and sys.stdin.isatty():
        console.print("[INFO] Интерактивный режим stats compare")
        _print_interactive_catalog(available)
        prompt_interactive_compare_args(args, sorted(available.keys()))
    if not args.left or not args.right:
        console.print("[ERROR] Для compare нужны --left и --right.")
        return 2
    if args.left == args.right:
        console.print("[ERROR] --left и --right должны быть разными.")
        return 2
    _ = _filter_dataset_names(available, [args.left, args.right])
    left = _scan_one_dataset(available[args.left], args.left)
    right = _scan_one_dataset(available[args.right], args.right)
    report = compare_dataset_stats(left, right)
    _render_compare_summary(report, abs_values=bool(args.abs))
    if args.details in ("classes", "all"):
        _render_compare_classes(report, top_n=args.top_n, abs_values=bool(args.abs))
    out_paths = _export_compare_report(
        layout,
        report,
        export_json=bool(args.export_json),
        export_csv=bool(args.export_csv),
    )
    for p in out_paths:
        console.print(f"[OK] Экспорт: {p}")
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    subcmd = None
    if argv and argv[0] in ("datasets", "classes", "compare"):
        subcmd = argv[0]
        argv = argv[1:]
    parser = build_stats_compare_arg_parser() if subcmd == "compare" else build_stats_arg_parser()
    args = parser.parse_args(argv)
    try:
        root = resolve_workspace_root(args.workspace)
        layout = WorkspaceLayout(root)
        if subcmd == "compare":
            code = _run_stats_compare(args, layout)
        else:
            code = _run_stats(args, layout)
    except ValueError as e:
        console.print(f"[ERROR] {e}")
        code = 2
    except Exception as e:
        console.print(f"[ERROR] Неожиданная ошибка: {e}")
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()

