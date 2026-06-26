from __future__ import annotations

import os
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

from smartrain.services.datasets.cvsdcldet_converter import (
    collect_cvsdcldet_class_names,
    collect_cvsdcldet_pairs,
    is_cvsdcldet_dir,
)
from smartrain.services.datasets.dataset_scan import (
    find_obj_data_file,
    find_obj_names_file,
    find_yaml_file,
    load_obj_names,
)


IMAGE_EXTS_FLAT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _find_cvat_annotations_xml(folder_path: str) -> Optional[str]:
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower() == "annotations.xml":
                return os.path.join(root, f)
    return None


def _cvat_has_images_dir_near_xml(xml_path: str) -> bool:
    try:
        p = os.path.dirname(xml_path)
        return os.path.isdir(os.path.join(p, "images"))
    except Exception:
        return False


def _is_cvat11_images_xml(xml_path: str) -> bool:
    """
    Minimal validation that annotations.xml is similar to CVAT for images 1.1.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return False

    if (root.tag or "").strip().lower() != "annotations":
        return False

    ver_el = root.find("./version")
    if ver_el is not None and ver_el.text and ver_el.text.strip():
        if ver_el.text.strip() != "1.1":
            return False

    images = root.findall("./image")
    if not images:
        return False

    # If there is no box at all, it may be empty markup; we consider it valid.
    boxes = root.findall("./image/box")
    if boxes:
        for b in boxes[:5]:
            if not b.attrib.get("label"):
                return False
            for k in ("xtl", "ytl", "xbr", "ybr"):
                if k not in b.attrib:
                    return False
    return True


def _load_cvat11_label_names(xml_path: str) -> list[str]:
    """
    CVAT 1.1 (Images task) labels.
    Prefer meta/task/labels/label/name; fallback to unique box/@label values.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return []

    meta_names: list[str] = []
    try:
        for lb in root.findall("./meta/task/labels/label/name"):
            if lb is not None and lb.text and lb.text.strip():
                meta_names.append(lb.text.strip())
    except Exception:
        meta_names = []

    if meta_names:
        out: list[str] = []
        seen = set()
        for n in meta_names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    seen = set()
    out: list[str] = []
    for box in root.findall("./image/box"):
        label = box.attrib.get("label", "")
        if label and label not in seen:
            seen.add(label)
            out.append(label)
    return sorted(out)


def _is_split_name(dir_name: str) -> bool:
    return dir_name.lower() in ("train", "val", "test")


def yolo_flat_image_label_buckets(folder_path: str):
    """
    Pair of directories (images..., labels...) for flat YOLO datasets:
    - classic flat: one (images, labels) bucket
    - CVAT export layout: images/<sub>/ and labels/<sub>/ for arbitrary subfolders
    """
    images_path = os.path.join(folder_path, "images")
    labels_path = os.path.join(folder_path, "labels")
    if not os.path.isdir(images_path) or not os.path.isdir(labels_path):
        return []

    subdirs_img = [
        d
        for d in os.listdir(images_path)
        if os.path.isdir(os.path.join(images_path, d))
    ]
    if any(_is_split_name(d) for d in subdirs_img):
        return []

    lbl_dirnames = {
        d
        for d in os.listdir(labels_path)
        if os.path.isdir(os.path.join(labels_path, d))
    }
    paired = sorted(set(subdirs_img) & lbl_dirnames)

    has_root_imgs = any(
        os.path.isfile(os.path.join(images_path, f))
        and f.lower().endswith(ext)
        for f in os.listdir(images_path)
        for ext in IMAGE_EXTS_FLAT
    )
    has_root_lbls = any(
        os.path.isfile(os.path.join(labels_path, f))
        and f.lower().endswith(".txt")
        for f in os.listdir(labels_path)
    )

    buckets = []
    if has_root_imgs or has_root_lbls:
        buckets.append((images_path, labels_path))
    for d in paired:
        buckets.append(
            (
                os.path.join(images_path, d),
                os.path.join(labels_path, d),
            )
        )
    return buckets


def detect_structure(folder_path: str) -> str:
    subfolders = [
        d.lower()
        for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
    ]

    obj_train_data_path = os.path.join(folder_path, "obj_train_data")
    obj_names_path = find_obj_names_file(folder_path)
    obj_data_path = find_obj_data_file(folder_path)

    if os.path.exists(obj_train_data_path) and (obj_names_path or obj_data_path):
        return "darknet"

    if is_cvsdcldet_dir(folder_path):
        return "cvsdcldet"

    cvat_xml = _find_cvat_annotations_xml(folder_path)
    if cvat_xml and _cvat_has_images_dir_near_xml(cvat_xml) and _is_cvat11_images_xml(cvat_xml):
        return "cvat11"

    if any(x in subfolders for x in ["train", "val", "test"]):
        return "split"

    if all(os.path.exists(os.path.join(folder_path, subdir)) for subdir in ["images", "labels"]):
        images_path = os.path.join(folder_path, "images")
        images_entries = os.listdir(images_path)
        if any(
            os.path.isdir(os.path.join(images_path, d)) and _is_split_name(d)
            for d in images_entries
        ):
            return "nested_split"

        buckets = yolo_flat_image_label_buckets(folder_path)
        if not buckets:
            return "unknown"

        images_root = os.path.join(folder_path, "images")
        has_subset = any(img != images_root for img, _ in buckets)
        if has_subset:
            return "subset_flat"
        return "flat"

    return "unknown"


def load_yaml(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {e}")
        return None


def count_elements(folder_path: str, structure: str):
    labels_count = 0
    images_count = 0
    IMAGE_EXTS = list(IMAGE_EXTS_FLAT)

    if structure == "cvat11":
        xml_path = _find_cvat_annotations_xml(folder_path)
        if not xml_path:
            return None
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            xml_images = root.findall("./image")
            labels_count = len(xml_images)
        except Exception:
            return None
        img_dir = os.path.join(os.path.dirname(xml_path), "images")
        if os.path.isdir(img_dir):
            images_count = len(
                [
                    f
                    for f in os.listdir(img_dir)
                    if any(f.lower().endswith(ext) for ext in IMAGE_EXTS)
                ]
            )
        else:
            images_count = 0

    if structure == "split":
        for dir_name in os.listdir(folder_path):
            dir_path = os.path.join(folder_path, dir_name)
            if not os.path.isdir(dir_path):
                continue

            img_dir = os.path.join(folder_path, dir_name, "images")
            lbl_dir = os.path.join(folder_path, dir_name, "labels")

            if os.path.exists(img_dir):
                images_count += len(
                    [
                        f
                        for f in os.listdir(img_dir)
                        if any(f.lower().endswith(ext) for ext in IMAGE_EXTS)
                    ]
                )
            if os.path.exists(lbl_dir):
                labels_count += len(
                    [
                        f
                        for f in os.listdir(lbl_dir)
                        if f.lower().endswith(".txt")
                    ]
                )

    elif structure in ("flat", "subset_flat"):
        buckets = yolo_flat_image_label_buckets(folder_path)
        if not buckets:
            img_dir = os.path.join(folder_path, "images")
            lbl_dir = os.path.join(folder_path, "labels")
            buckets = [(img_dir, lbl_dir)]
        for img_dir, lbl_dir in buckets:
            if os.path.exists(img_dir):
                images_count += len(
                    [
                        f
                        for f in os.listdir(img_dir)
                        if any(f.lower().endswith(ext) for ext in IMAGE_EXTS)
                    ]
                )
            if os.path.exists(lbl_dir):
                labels_count += len(
                    [
                        f
                        for f in os.listdir(lbl_dir)
                        if f.lower().endswith(".txt")
                    ]
                )

    elif structure == "nested_split":
        for split in ["train", "val", "test"]:
            img_dir = os.path.join(folder_path, "images", split)
            lbl_dir = os.path.join(folder_path, "labels", split)
            if os.path.exists(img_dir):
                images_count += len(
                    [
                        f
                        for f in os.listdir(img_dir)
                        if any(f.lower().endswith(ext) for ext in IMAGE_EXTS)
                    ]
                )
            if os.path.exists(lbl_dir):
                labels_count += len(
                    [
                        f
                        for f in os.listdir(lbl_dir)
                        if f.lower().endswith(".txt")
                    ]
                )

    elif structure == "darknet":
        obj_train_data_path = os.path.join(folder_path, "obj_train_data")
        if os.path.exists(obj_train_data_path):
            files = os.listdir(obj_train_data_path)
            images_count = len(
                [f for f in files if any(f.lower().endswith(ext) for ext in IMAGE_EXTS)]
            )
            labels_count = len([f for f in files if f.lower().endswith(".txt")])
        else:
            return None

    elif structure == "cvsdcldet":
        pairs = collect_cvsdcldet_pairs(Path(folder_path))
        images_count = len(pairs)
        labels_count = images_count

    else:
        return None

    if images_count == labels_count:
        return images_count
    print(
        f"[WARNING] The folder {folder_path} contains the number of images ({images_count})"
        f"does not match the number of annotations ({labels_count})"
    )
    return images_count, labels_count


def process_dataset(folder_path: str, folder_name: str):
    yaml_path = find_yaml_file(folder_path)

    names = None
    structure = detect_structure(folder_path)

    if yaml_path:
        data = load_yaml(yaml_path)
        if data and "names" in data:
            names = data["names"]
            if isinstance(names, list):
                pass
            elif isinstance(names, dict):
                names = [v for k, v in sorted(names.items())]
            else:
                print(f"[ERROR] The 'names' field in {yaml_path} is not in the correct format")
                return None

    if not names and structure == "darknet":
        obj_names_path = find_obj_names_file(folder_path)
        if obj_names_path:
            names = load_obj_names(obj_names_path)
            if not names:
                print(f"[WARNING] Failed to load classes from {obj_names_path} - skipping")
                return None
        else:
            print(f"[WARNING] obj.names not found in folder {folder_name} - skip")
            return None

    if not names and structure == "cvat11":
        xml_path = _find_cvat_annotations_xml(folder_path)
        if xml_path:
            names = _load_cvat11_label_names(xml_path)
            if not names:
                print(f"[WARNING] CVAT 1.1: failed to extract labels from {xml_path} - skipping")
                return None
        else:
            print(f"[WARNING] CVAT 1.1: annotations.xml not found - skipping")
            return None

    if not names and structure == "cvsdcldet":
        names = collect_cvsdcldet_class_names(Path(folder_path))
        if not names:
            print(f"[WARNING] CvsDclDet: no class names found in {folder_name} - skip")
            return None

    if not names:
        print(f"[WARNING] Data.yaml or obj.names not found in folder {folder_name} - skip")
        return None

    elements_count = count_elements(folder_path, structure)
    return {
        "classes": {name: idx for idx, name in enumerate(names)},
        "structure": structure,
        "elements_count": elements_count,
    }

