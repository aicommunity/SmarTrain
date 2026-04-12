import os
import copy
import yaml
import json
import sys
import argparse
import hashlib
import zipfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

from smartrain.cli_argparse import CliArgumentParser
from smartrain.cvat11_converter import generate_temp_yolo_labels_from_cvat11_extracted
from smartrain.dataset_hash import calculate_dataset_hash
from smartrain.dataset_passport import write_dataset_passport
from smartrain.workspace_paths import (
    WORKSPACE_ENV_VAR,
    WorkspaceLayout,
    resolve_workspace_root,
    resolve_or_extract_dataset_root,
    DATASETS_INFO_FILE,
    CLASS_NAMES_FILE,
    DATASETS_SCAN_SUMMARY_FILE,
)


sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_FILE = DATASETS_INFO_FILE
OUTPUT_CLASS_NAMES_FILE = CLASS_NAMES_FILE
DEFAULT_DATASETS_LIST_FILE = "datasets_list.txt"
SOURCE_SIGNATURE_KEY = "source_signature"
DATASET_HASH_KEY = "dataset_hash"
SOURCE_HASH_KEY = "source_hash"
SOURCE_REF_KEY = "source_ref"
MODIFIED_KEY = "modified"


def find_yaml_file(folder_path):
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower() in ("data.yaml", "data.yml"):
                return os.path.join(root, f)
    return None


def find_obj_names_file(folder_path):
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower() == "obj.names":
                return os.path.join(root, f)
    return None


def find_obj_data_file(folder_path):
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower() == "obj.data":
                return os.path.join(root, f)
    return None


def load_obj_names(file_path):
    """Reading obj.names file (one class per line)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        return names
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {e}")
        return None


def load_obj_data(file_path):
    """Parsing obj.data file to get the number of classes"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Looking for the line classes = X
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('classes'):
                parts = line.split('=')
                if len(parts) == 2:
                    return int(parts[1].strip())
        return None
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {e}")
        return None


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
    Minimal validation that annotations.xml is similar to CVAT for images 1.1:
    - root == <annotations>
    - there is at least one <image>
    - for images-task there is usually a <box> inside an <image> (but empty markup is allowed)
    - if there is a <version>, then it must be 1.1 (otherwise we consider it inappropriate)
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

    # If there is no box at all, it may be empty markup; We consider it valid.
    # But if there is a box, let's check for the presence of required attributes.
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
        # unique preserve order
        out: list[str] = []
        seen = set()
        for n in meta_names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    seen = set()
    out = []
    for box in root.findall("./image/box"):
        label = box.attrib.get("label", "")
        if label and label not in seen:
            seen.add(label)
            out.append(label)
    return sorted(out)


IMAGE_EXTS_FLAT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _is_split_name(dir_name):
    return dir_name.lower() in ("train", "val", "test")


def yolo_flat_image_label_buckets(folder_path):
    """
    Pairs of directories (images..., labels...) for flat YOLO datasets:
    — classic flat: one pair (images, labels), if there are files in the root;
    — CVAT export “Ultralytics YOLO Detection 1.0”: images/<subfolder>/ and labels/<same subfolder>/
      (subfolder name is arbitrary, not just train/val/test).
    If both root files and shared subfolders are found, both options are returned.
    For nested split (images/train/…) returns an empty list - use nested_split.
    """
    images_path = os.path.join(folder_path, "images")
    labels_path = os.path.join(folder_path, "labels")
    if not os.path.isdir(images_path) or not os.path.isdir(labels_path):
        return []

    subdirs_img = [
        d for d in os.listdir(images_path)
        if os.path.isdir(os.path.join(images_path, d))
    ]
    if any(_is_split_name(d) for d in subdirs_img):
        return []

    lbl_dirnames = {
        d for d in os.listdir(labels_path)
        if os.path.isdir(os.path.join(labels_path, d))
    }
    paired = sorted(set(subdirs_img) & lbl_dirnames)

    has_root_imgs = any(
        os.path.isfile(os.path.join(images_path, f))
        and f.lower().endswith(IMAGE_EXTS_FLAT)
        for f in os.listdir(images_path)
    )
    has_root_lbls = any(
        os.path.isfile(os.path.join(labels_path, f)) and f.lower().endswith(".txt")
        for f in os.listdir(labels_path)
    )

    buckets = []
    if has_root_imgs or has_root_lbls:
        buckets.append((images_path, labels_path))
    for d in paired:
        buckets.append((
            os.path.join(images_path, d),
            os.path.join(labels_path, d),
        ))
    return buckets


def detect_structure(folder_path):
    subfolders = [d.lower() for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    # Darknet YOLO format checker
    obj_train_data_path = os.path.join(folder_path, "obj_train_data")
    obj_names_path = find_obj_names_file(folder_path)
    obj_data_path = find_obj_data_file(folder_path)
    
    if os.path.exists(obj_train_data_path) and (obj_names_path or obj_data_path):
        return "darknet"

    # CVAT 1.1 extracted folder: annotations.xml + images/
    cvat_xml = _find_cvat_annotations_xml(folder_path)
    if cvat_xml and _cvat_has_images_dir_near_xml(cvat_xml) and _is_cvat11_images_xml(cvat_xml):
        return "cvat11"

    if any(x in subfolders for x in ["train", "val", "test"]):
        return "split"

    elif all(os.path.exists(os.path.join(folder_path, subdir)) for subdir in ["images", "labels"]):
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

    else:
        return "unknown"


def load_yaml(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {e}")
        return None


def count_elements(folder_path, structure):
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
                images_count += len([
                    f for f in os.listdir(img_dir)
                    if any(f.lower().endswith(ext) for ext in IMAGE_EXTS)
                ])

            if os.path.exists(lbl_dir):
                labels_count += len([
                    f for f in os.listdir(lbl_dir)
                    if f.lower().endswith(".txt")
                ])

    elif structure in ("flat", "subset_flat"):
        buckets = yolo_flat_image_label_buckets(folder_path)
        if not buckets:
            img_dir = os.path.join(folder_path, "images")
            lbl_dir = os.path.join(folder_path, "labels")
            buckets = [(img_dir, lbl_dir)]
        for img_dir, lbl_dir in buckets:
            if os.path.exists(img_dir):
                images_count += len([
                    f for f in os.listdir(img_dir)
                    if any(f.lower().endswith(ext) for ext in IMAGE_EXTS)
                ])
            if os.path.exists(lbl_dir):
                labels_count += len([
                    f for f in os.listdir(lbl_dir)
                    if f.lower().endswith(".txt")
                ])

    elif structure == "nested_split":
        for split in ["train", "val", "test"]:
            img_dir = os.path.join(folder_path, "images", split)
            lbl_dir = os.path.join(folder_path, "labels", split)

            if os.path.exists(img_dir):
                images_count += len([
                    f for f in os.listdir(img_dir)
                    if any(f.lower().endswith(ext) for ext in IMAGE_EXTS)
                ])

            if os.path.exists(lbl_dir):
                labels_count += len([
                    f for f in os.listdir(lbl_dir)
                    if f.lower().endswith(".txt")
                ])

    elif structure == "darknet":
        obj_train_data_path = os.path.join(folder_path, "obj_train_data")
        if os.path.exists(obj_train_data_path):
            files = os.listdir(obj_train_data_path)
            images_count = len([
                f for f in files
                if any(f.lower().endswith(ext) for ext in IMAGE_EXTS)
            ])
            labels_count = len([
                f for f in files
                if f.lower().endswith(".txt")
            ])
        else:
            return None

    else:
        return None

    if images_count == labels_count:
        return images_count
    else:
        print(f"[WARNING] The folder {folder_path} contains the number of images ({images_count})"
              f"does not match the number of annotations ({labels_count})")
        return images_count, labels_count



def process_dataset(folder_path, folder_name):
    yaml_path = find_yaml_file(folder_path)
    
    names = None
    structure = detect_structure(folder_path)

    # Trying to load from YAML (YOLOv8 format)
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

    # If you don't find YAML, try the Darknet format
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

    # If CVAT 1.1 extracted dataset
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

    # If nothing is found
    if not names:
        print(f"[WARNING] Data.yaml or obj.names not found in folder {folder_name} - skip")
        return None

    elements_count = count_elements(folder_path, structure)

    return {
        "classes": {name: idx for idx, name in enumerate(names)},
        "structure": structure,
        "elements_count": elements_count
    }


def build_datasets_json_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        description="Processing datasets and creating JSON files with information about classes and structure"
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Root workspace (otherwise {WORKSPACE_ENV_VAR}); scan reads raw_data/ and writes datasets/",
    )

    parser.add_argument(
        "--datasets-path",
        type=str,
        default=None,
        help="Mode without workspace: root of dataset directories for scanning",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Mode without workspace: directory for datasets_info.json and class_names.json",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=("scan", "refresh"),
        default="scan",
        help="scan: crawl raw_data subdirectories (or --datasets-path); "
        "refresh: Rescan only data_path from existing datasets_info.json",
    )
    parser.add_argument(
        "--datasets-list",
        type=str,
        default=None,
        help="TXT file with a list of paths to datasets (one per line): directories or .zip",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Explicitly specify the dataset (name from raw_data or path). The flag can be repeated.",
    )
    parser.add_argument(
        "--purge-processed-raw",
        action="store_true",
        help="Workspace/scan only: after confirmation, remove from raw_data the sources processed in the current run.",
    )

    return parser


def parse_args(argv=None):
    return build_datasets_json_arg_parser().parse_args(argv)


# Dataset record fields saved during rescanning (manually in JSON)
_PRESERVED_DATASET_INFO_KEYS = (
    "roi_auto",
    "tags",
    "data_path",
    SOURCE_SIGNATURE_KEY,
    DATASET_HASH_KEY,
    SOURCE_HASH_KEY,
    SOURCE_REF_KEY,
    MODIFIED_KEY,
)


def _sorted_diff(old: Set[str], new: Set[str]) -> Tuple[List[str], List[str]]:
    added = sorted(new - old)
    removed = sorted(old - new)
    return added, removed


def _write_scan_summary(
    *,
    output_dir: str,
    datasets_final: Dict[str, Any],
    class_names_final: Dict[str, Any],
    datasets_added: List[str],
    datasets_removed: List[str],
    class_names_added: List[str],
    class_names_removed: List[str],
) -> str:
    """Writes datasets_scan_summary.json; returns the file path."""
    path = os.path.join(output_dir, DATASETS_SCAN_SUMMARY_FILE)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "final": sorted(datasets_final.keys()),
            "count": len(datasets_final),
            "added": datasets_added,
            "removed": datasets_removed,
        },
        "class_names": {
            "final": sorted(class_names_final.keys()),
            "count": len(class_names_final),
            "added": class_names_added,
            "removed": class_names_removed,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _print_scan_report(
    *,
    summary_path: str,
    datasets_added: List[str],
    datasets_removed: List[str],
    class_names_added: List[str],
    class_names_removed: List[str],
    datasets_final_count: int,
    class_names_final_count: int,
    had_previous_datasets: bool,
    had_previous_class_names: bool,
) -> None:
    print(f"[INFO] Final datasets in {OUTPUT_FILE}: {datasets_final_count}")
    if datasets_added:
        print(f"[INFO] Added datasets ({len(datasets_added)}): {', '.join(datasets_added)}")
    if datasets_removed:
        print(f"[INFO] Removed from catalog ({len(datasets_removed)}): {', '.join(datasets_removed)}")
    if not datasets_added and not datasets_removed and had_previous_datasets:
        print("[INFO] The composition of the datasets relative to the previous file has not changed.")
    print(f"[INFO] Final class names in {OUTPUT_CLASS_NAMES_FILE}: {class_names_final_count}")
    if class_names_added:
        print(f"[INFO] New class names ({len(class_names_added)}): {', '.join(class_names_added)}")
    if class_names_removed:
        print(f"[INFO] Removed class names ({len(class_names_removed)}): {', '.join(class_names_removed)}")
    if not class_names_added and not class_names_removed and had_previous_class_names:
        print("[INFO] The composition of class_names relative to the previous file has not changed.")
    print(f"[OK] Summary saved: {summary_path}")


def _merge_preserved_dataset_fields(fresh: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Adds a fresh record from a scan with the roi_auto/tags fields from the old datasets_info.json."""
    if not previous:
        return fresh
    out = dict(fresh)
    for key in _PRESERVED_DATASET_INFO_KEYS:
        if key in previous:
            out[key] = previous[key]
    return out


def _run_scan_folder_roots(folder_roots: list[tuple[str, str, dict]]) -> tuple[dict, dict]:
    """folder_roots: list (logical_name, folder_path on disk, overrides for datasets_info)."""
    datasets_info: dict = {}
    class_names: dict = {}

    for logical_name, folder_path, overrides in folder_roots:
        if not os.path.isdir(folder_path):
            print(f"[WARNING] Skipping {logical_name!r}: no directory {folder_path}")
            continue
        info = process_dataset(folder_path, logical_name)
        if info:
            if overrides:
                info.update(overrides)
            datasets_info[logical_name] = info
            for class_name in info["classes"]:
                class_names[class_name] = class_name
    return datasets_info, class_names


def _dir_has_content(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    with os.scandir(path) as it:
        for _ in it:
            return True
    return False


def _normalize_path_for_data_path(path: str, workspace_root: Optional[str]) -> str:
    """Returns the path for data_path: relative to workspace, if possible."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    if workspace_root:
        try:
            rel = os.path.relpath(abs_path, workspace_root)
            if not rel.startswith(".."):
                return rel
        except Exception:
            pass
    return abs_path


def _load_datasets_list_file(list_path: str) -> list[str]:
    entries: list[str] = []
    list_dir = os.path.dirname(list_path)
    with open(list_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            expanded = os.path.expanduser(line)
            if not os.path.isabs(expanded):
                expanded = os.path.join(list_dir, expanded)
            entries.append(os.path.abspath(expanded))
    return entries


def _unique_dataset_key(base_name: str, used_names: set[str]) -> str:
    key = base_name or "dataset"
    if key not in used_names:
        used_names.add(key)
        return key
    idx = 2
    while f"{key}_{idx}" in used_names:
        idx += 1
    unique = f"{key}_{idx}"
    used_names.add(unique)
    return unique


def _zip_extract_path(temp_root: str, zip_path: str) -> str:
    abs_zip = os.path.abspath(zip_path)
    sig = hashlib.sha1(abs_zip.encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(abs_zip))[0]
    return os.path.join(temp_root, f"{stem}_{sig}")


def _extract_zip_for_scan(zip_path: str, temp_root: str) -> str:
    os.makedirs(temp_root, exist_ok=True)
    out_dir = _zip_extract_path(temp_root, zip_path)
    marker = os.path.join(out_dir, ".extract_done")
    if os.path.isfile(marker):
        return out_dir
    if os.path.isdir(out_dir):
        for root, dirs, files in os.walk(out_dir, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok")
    return out_dir


def _append_roots_from_datasets_list(
    *,
    list_path: str,
    folder_roots: list[tuple[str, str, dict]],
    used_names: set[str],
    use_workspace: bool,
    layout: Optional[WorkspaceLayout],
    output_dir: str,
) -> None:
    entries = _load_datasets_list_file(list_path)
    workspace_root = layout.root if layout else None
    for src_path in entries:
        if not os.path.exists(src_path):
            print(f"[WARNING] Skipping from datasets-list: path not found: {src_path}")
            continue

        base_name = (
            os.path.splitext(os.path.basename(src_path))[0]
            if src_path.lower().endswith(".zip")
            else os.path.basename(src_path)
        )
        logical_name = _unique_dataset_key(base_name, used_names)
        data_path_value = _normalize_path_for_data_path(src_path, workspace_root)

        if os.path.isdir(src_path):
            folder_roots.append((logical_name, src_path, {"data_path": data_path_value}))
            continue

        if src_path.lower().endswith(".zip"):
            try:
                if use_workspace and layout:
                    extracted = resolve_or_extract_dataset_root(
                        layout.root,
                        logical_name,
                        {"data_path": data_path_value},
                        layout.raw_data,
                    )
                else:
                    extracted = _extract_zip_for_scan(
                        src_path,
                        os.path.join(output_dir, "tmp", "datasets_list_extract"),
                    )
            except Exception as e:
                print(f"[WARNING] Skipping archives from datasets-list {src_path!r}: {e}")
                continue
            folder_roots.append((logical_name, extracted, {"data_path": data_path_value}))
            continue

        print(
            f"[WARNING] Skipping from datasets-list: only directories and .zip are supported,"
            f"received: {src_path}"
        )


def _compute_source_signature(path: str) -> str:
    ap = os.path.abspath(path)
    if os.path.isfile(ap) and ap.lower().endswith(".zip"):
        st = os.stat(ap)
        payload = f"zip|{ap}|{st.st_size}|{getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9))}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    rows: list[str] = []
    for root, dirs, files in os.walk(ap):
        dirs.sort()
        files.sort()
        rel_root = os.path.relpath(root, ap)
        rows.append(f"d:{rel_root}")
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            rel = os.path.relpath(fp, ap)
            rows.append(f"f:{rel}:{st.st_size}:{getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9))}")
    joined = "\n".join(rows).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()[:16]


def _copy_source_to_training(src_root: str, dst_root: str) -> None:
    if os.path.isdir(dst_root):
        shutil.rmtree(dst_root, ignore_errors=True)
    os.makedirs(os.path.dirname(dst_root), exist_ok=True)
    shutil.copytree(src_root, dst_root)
    _ensure_training_ready_after_copy(dst_root)


def _ensure_training_ready_after_copy(dataset_root: str) -> bool:
    """
    Normalizes the copied dataset to a form suitable for training.
    Now the critical case: cvat11 (annotations.xml + images/) -> YOLO labels + data.yaml.
    """
    structure = detect_structure(dataset_root)
    if structure != "cvat11":
        return False

    xml_path = _find_cvat_annotations_xml(dataset_root)
    if not xml_path:
        return False
    names = _load_cvat11_label_names(xml_path)
    if not names:
        print(f"[WARNING] CVAT 1.1: could not determine class list for {dataset_root}")
        return False

    labels_dir = os.path.join(dataset_root, "labels")
    # Always rebuild labels from annotations.xml so stale flat labels (older scan layout)
    # cannot coexist with nested image paths.
    if os.path.isdir(labels_dir):
        shutil.rmtree(labels_dir, ignore_errors=True)
    os.makedirs(labels_dir, exist_ok=True)
    class_name_to_id = {name: idx for idx, name in enumerate(names)}
    try:
        _images_dir, _images_found, _labels_written = generate_temp_yolo_labels_from_cvat11_extracted(
            dataset_root=Path(dataset_root),
            labels_out_dir=Path(labels_dir),
            class_name_to_id=class_name_to_id,
        )
    except Exception as e:
        print(f"[WARNING] CVAT 1.1: failed to generate YOLO labels for {dataset_root}: {e}")
        return False

    data_yaml = os.path.join(dataset_root, "data.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(
            "# smartrain (CVAT 1.1 scan): images/ may contain nested subfolders; "
            "labels/ mirrors the same relative paths (YOLO pairing).\n"
            "# No path: key — Ultralytics uses this file's directory as dataset root.\n"
        )
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("test: images\n\n")
        f.write(f"nc: {len(names)}\n")
        f.write(f"names: {names}\n")
    return True


def _dataset_content_hash(path: str) -> Optional[str]:
    try:
        return str(calculate_dataset_hash(path))
    except Exception as e:
        print(f"[WARNING] Failed to calculate dataset_hash for {path!r}: {e}")
        return None


def _ensure_prev_entry(previous_info: dict, key: str) -> dict:
    entry = previous_info.get(key)
    if not isinstance(entry, dict):
        entry = {}
        previous_info[key] = entry
    return entry


def _append_to_datasets_list(list_path: str, value: str) -> None:
    existing: set[str] = set()
    if os.path.isfile(list_path):
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    existing.add(s)
    if value in existing:
        return
    with open(list_path, "a", encoding="utf-8") as f:
        if existing:
            f.write("\n")
        f.write(value)
        f.write("\n")


def _confirm_purge_processed_raw(paths: list[str]) -> bool:
    if not paths:
        return False
    print("[WARNING] Requested to remove processed sources from raw_data.")
    print("[WARNING] Will be removed:")
    for p in paths:
        print(f"  - {p}")
    if not sys.stdin.isatty():
        print("[WARNING] No interactive TTY: deletion cancelled.")
        return False
    ans = input("Continue deletion? [Y/n]: ").strip().lower()
    return ans in ("", "y", "yes", "1", "true", "yes", "d")


def _purge_raw_sources(paths: list[str]) -> tuple[int, int]:
    removed = 0
    failed = 0
    for p in paths:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
                removed += 1
            elif os.path.isfile(p):
                os.remove(p)
                removed += 1
        except Exception as e:
            failed += 1
            print(f"[WARNING] Failed to delete {p}: {e}")
    return removed, failed


def _ensure_scan_initial_passport(
    *,
    dataset_dir: str,
    dataset_name: str,
    entry: dict,
    workspace_root: str,
) -> None:
    passport_path = os.path.join(dataset_dir, "dataset_passport.json")
    if os.path.isfile(passport_path):
        return
    src_ref = entry.get(SOURCE_REF_KEY)
    src_payload: dict[str, Any] = {
        "name": str(src_ref) if src_ref else dataset_name,
        "path": str(src_ref) if src_ref else None,
        "dataset_hash": entry.get(SOURCE_HASH_KEY),
        "source_signature": entry.get(SOURCE_SIGNATURE_KEY),
    }
    # Clean None to keep passport compact and deterministic.
    src_payload = {k: v for k, v in src_payload.items() if v is not None}
    write_dataset_passport(
        output_dataset_dir=dataset_dir,
        command="scan",
        source_datasets=[src_payload],
        parameters={
            "workspace": workspace_root,
            "kind": "initial",
        },
        transformations=[
            {
                "type": "initial_sync_from_raw_data",
                "source_ref": src_ref,
                "structure": entry.get("structure"),
            }
        ],
        random_seed=None,
        stats_before={},
        stats_after={
            "classes": len(entry.get("names", []) or []),
            "dataset_hash": entry.get(DATASET_HASH_KEY),
        },
    )


def _append_explicit_dataset(
    *,
    raw: str,
    layout: WorkspaceLayout,
    output_file: str,
    folder_roots: list[tuple[str, str, dict]],
    used_names: set[str],
) -> None:
    token = (raw or "").strip()
    if not token:
        return
    candidate = os.path.abspath(os.path.expanduser(token))
    if os.path.exists(candidate):
        src_path = candidate
        base_name = os.path.splitext(os.path.basename(src_path))[0] if src_path.lower().endswith(".zip") else os.path.basename(src_path)
        list_value = src_path
    else:
        name = token[:-4] if token.lower().endswith(".zip") else token
        dir_candidate = os.path.join(layout.raw_data, name)
        zip_candidate = os.path.join(layout.raw_data, f"{name}.zip")
        if os.path.isdir(dir_candidate):
            src_path = dir_candidate
            base_name = name
            list_value = name
        elif os.path.isfile(zip_candidate):
            src_path = zip_candidate
            base_name = name
            list_value = f"{name}.zip"
        else:
            print(f"[WARNING] --dataset {raw!r}: path/name not found")
            return
    logical_name = _unique_dataset_key(base_name, used_names)
    sig = _compute_source_signature(src_path)
    prev_sig = None
    if os.path.isfile(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict) and isinstance(prev.get(logical_name), dict):
                prev_sig = prev[logical_name].get(SOURCE_SIGNATURE_KEY)
        except Exception:
            prev_sig = None
    dst = os.path.join(layout.datasets, logical_name)
    if src_path.lower().endswith(".zip"):
        try:
            extracted = resolve_or_extract_dataset_root(
                layout.root,
                logical_name,
                {"data_path": os.path.relpath(src_path, layout.root)},
                layout.raw_data,
            )
        except Exception as e:
            print(f"[WARNING] --dataset {raw!r}: failed to unpack zip ({e})")
            return
        source_for_copy = extracted
    else:
        source_for_copy = src_path
    if not (prev_sig == sig and _dir_has_content(dst)):
        _copy_source_to_training(source_for_copy, dst)
    else:
        print(f"[INFO] Skipping {logical_name!r}: source has not changed.")
    folder_roots.append(
        (
            logical_name,
            dst,
            {"data_path": os.path.relpath(dst, layout.root), SOURCE_SIGNATURE_KEY: sig},
        )
    )
    _append_to_datasets_list(os.path.join(layout.raw_data, DEFAULT_DATASETS_LIST_FILE), list_value)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    if args.datasets_path:
        use_workspace = False
        layout = None
    else:
        try:
            root = resolve_workspace_root(args.workspace)
        except ValueError as e:
            print(f"[ERROR] {e}")
            return
        use_workspace = True
        layout = WorkspaceLayout(root)
        os.makedirs(layout.raw_data, exist_ok=True)
        os.makedirs(layout.datasets, exist_ok=True)

    if use_workspace:
        # The source of truth for the index is the datasets directory (ready-made datasets).
        # raw_data is used only as a source of new/updated data.
        output_dir = layout.datasets
        output_file = os.path.join(output_dir, OUTPUT_FILE)
        output_class_names_file = os.path.join(output_dir, OUTPUT_CLASS_NAMES_FILE)
        datasets_dir = layout.datasets
        raw_source_dir = layout.raw_data
    else:
        datasets_dir = os.path.abspath(os.path.expanduser(args.datasets_path))
        if args.output_path:
            output_dir = os.path.abspath(os.path.expanduser(args.output_path))
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = datasets_dir
        output_file = os.path.join(output_dir, OUTPUT_FILE)
        output_class_names_file = os.path.join(output_dir, OUTPUT_CLASS_NAMES_FILE)

    datasets_info: dict = {}
    class_names: dict = {}
    processed_raw_sources: set[str] = set()

    if args.mode == "refresh" and not use_workspace:
        print("[ERROR] --mode refresh is only supported without --datasets-path (via workspace).")
        return
    if args.purge_processed_raw and (not use_workspace or args.mode != "scan"):
        print("[ERROR] --purge-processed-raw is only available in workspace with --mode scan.")
        return

    if use_workspace and args.mode == "refresh":
        if not os.path.isfile(output_file):
            print(f"[ERROR] Refresh mode: {output_file} not found")
            return
        with open(output_file, "r", encoding="utf-8") as f:
            previous_full = json.load(f)
        if not isinstance(previous_full, dict):
            print(f"[ERROR] {output_file}: JSON object with datasets expected.")
            return
        folder_roots = []
        for name, prev_entry in previous_full.items():
            if not isinstance(prev_entry, dict):
                continue
            if "data_path" not in prev_entry:
                folder_roots.append((name, os.path.join(datasets_dir, name), {}))
            else:
                dp = prev_entry["data_path"]
                if not isinstance(dp, str):
                    print(f"[WARNING] Skipping {name!r}: data_path is not a string")
                    continue
                root_path = resolve_or_extract_dataset_root(layout.root, name, prev_entry, datasets_dir)
                folder_roots.append((name, root_path, {"data_path": dp}))
        datasets_info, class_names = _run_scan_folder_roots(folder_roots)
        previous_info = previous_full
    else:
        if not os.path.exists(datasets_dir):
            print(f"[ERROR] Folder '{datasets_dir}' not found.")
            return
        previous_info = {}
        if os.path.isfile(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    loaded_prev = json.load(f)
                if isinstance(loaded_prev, dict):
                    previous_info = loaded_prev
            except Exception as e:
                print(f"[WARNING] Failed to read existing {output_file}: {e}")
        previous_info_for_diff = copy.deepcopy(previous_info)
        folder_roots = []
        used_names: set[str] = set()

        def _sync_one_source(*, logical_name: str, source_for_copy: str, source_ref: str, source_signature: str) -> bool:
            dst_dir = os.path.join(layout.datasets, logical_name)
            prev_entry = previous_info.get(logical_name)
            normalized_on_skip = False
            if isinstance(prev_entry, dict) and bool(prev_entry.get(MODIFIED_KEY)):
                print(f"[WARNING] Skipping synchronization {logical_name!r}: modified=true.")
                return False

            source_hash = _dataset_content_hash(source_for_copy)
            if source_hash:
                for ds_name in os.listdir(layout.datasets):
                    ds_path = os.path.join(layout.datasets, ds_name)
                    if not os.path.isdir(ds_path):
                        continue
                    if ds_name == logical_name:
                        continue
                    ds_hash = None
                    if isinstance(previous_info.get(ds_name), dict):
                        ds_hash = previous_info[ds_name].get(DATASET_HASH_KEY)
                    if not ds_hash:
                        ds_hash = _dataset_content_hash(ds_path)
                    if ds_hash and ds_hash == source_hash:
                        # Recreate datasets/<logical_name> after manual deletion: the slot may still
                        # be listed in datasets_info.json while the directory is gone; do not treat
                        # another folder with the same content as a reason to skip materializing it.
                        if not _dir_has_content(dst_dir) and logical_name in previous_info:
                            break
                        print(
                            f"[WARNING] Skipping source {logical_name!r}: data matches datasets/{ds_name!r}."
                        )
                        return False

            prev_sig = prev_entry.get(SOURCE_SIGNATURE_KEY) if isinstance(prev_entry, dict) else None
            if prev_sig == source_signature and _dir_has_content(dst_dir):
                print(f"[INFO] Skipping {logical_name!r}: source has not changed.")
                # Even with skip we maintain compatibility: we configure the training-ready layout.
                normalized_on_skip = _ensure_training_ready_after_copy(dst_dir)
            else:
                _copy_source_to_training(source_for_copy, dst_dir)

            entry = _ensure_prev_entry(previous_info, logical_name)
            entry[SOURCE_SIGNATURE_KEY] = source_signature
            entry[SOURCE_REF_KEY] = source_ref
            entry[MODIFIED_KEY] = bool(entry.get(MODIFIED_KEY, False))
            if source_hash:
                entry[SOURCE_HASH_KEY] = source_hash
            # dataset_hash should reflect the actual contents of datasets/<name>
            # after possible normalization in training-ready layout.
            # Do not overwrite dataset_hash with normal skip (otherwise we will lose detection of manual edits).
            # We update it only after actual synchronization or normalization of cvat11.
            if not (prev_sig == source_signature and _dir_has_content(dst_dir)) or normalized_on_skip:
                current_dst_hash = _dataset_content_hash(dst_dir)
                if current_dst_hash:
                    entry[DATASET_HASH_KEY] = current_dst_hash
            return True

        if use_workspace:
            if os.path.isdir(raw_source_dir):
                for src_name in os.listdir(raw_source_dir):
                    src_path = os.path.join(raw_source_dir, src_name)
                    if not (os.path.isdir(src_path) or src_name.lower().endswith(".zip")):
                        continue
                    logical_name = os.path.splitext(src_name)[0] if src_name.lower().endswith(".zip") else src_name
                    if src_name.lower().endswith(".zip"):
                        sig = _compute_source_signature(src_path)
                        try:
                            extracted = resolve_or_extract_dataset_root(
                                layout.root,
                                logical_name,
                                {"data_path": os.path.relpath(src_path, layout.root)},
                                raw_source_dir,
                            )
                        except Exception as e:
                            print(f"[WARNING] Skipping archive {src_name!r} from raw_data: {e}")
                            continue
                        synced = _sync_one_source(
                            logical_name=logical_name,
                            source_for_copy=extracted,
                            source_ref=os.path.relpath(src_path, layout.root),
                            source_signature=sig,
                        )
                        if synced:
                            processed_raw_sources.add(os.path.abspath(src_path))
                    else:
                        synced = _sync_one_source(
                            logical_name=logical_name,
                            source_for_copy=src_path,
                            source_ref=os.path.relpath(src_path, layout.root),
                            source_signature=_compute_source_signature(src_path),
                        )
                        if synced:
                            processed_raw_sources.add(os.path.abspath(src_path))

        if use_workspace and args.dataset:
            used_names_for_explicit = {
                d for d in os.listdir(layout.datasets) if os.path.isdir(os.path.join(layout.datasets, d))
            }
            for raw_item in args.dataset:
                token = (raw_item or "").strip()
                if not token:
                    continue
                candidate = os.path.abspath(os.path.expanduser(token))
                if os.path.exists(candidate):
                    src_path = candidate
                    base_name = (
                        os.path.splitext(os.path.basename(src_path))[0]
                        if src_path.lower().endswith(".zip")
                        else os.path.basename(src_path)
                    )
                    list_value = src_path
                else:
                    name = token[:-4] if token.lower().endswith(".zip") else token
                    dir_candidate = os.path.join(layout.raw_data, name)
                    zip_candidate = os.path.join(layout.raw_data, f"{name}.zip")
                    if os.path.isdir(dir_candidate):
                        src_path = dir_candidate
                        base_name = name
                        list_value = name
                    elif os.path.isfile(zip_candidate):
                        src_path = zip_candidate
                        base_name = name
                        list_value = f"{name}.zip"
                    else:
                        print(f"[WARNING] --dataset {raw_item!r}: path/name not found")
                        continue
                logical_name = _unique_dataset_key(base_name, used_names_for_explicit)
                if src_path.lower().endswith(".zip"):
                    sig = _compute_source_signature(src_path)
                    try:
                        extracted = resolve_or_extract_dataset_root(
                            layout.root,
                            logical_name,
                            {"data_path": os.path.relpath(src_path, layout.root)},
                            layout.raw_data,
                        )
                    except Exception as e:
                        print(f"[WARNING] --dataset {raw_item!r}: failed to unpack zip ({e})")
                        continue
                    _sync_one_source(
                        logical_name=logical_name,
                        source_for_copy=extracted,
                        source_ref=src_path,
                        source_signature=sig,
                    )
                else:
                    _sync_one_source(
                        logical_name=logical_name,
                        source_for_copy=src_path,
                        source_ref=src_path,
                        source_signature=_compute_source_signature(src_path),
                    )
                _append_to_datasets_list(
                    os.path.join(layout.raw_data, DEFAULT_DATASETS_LIST_FILE), list_value
                )
        if args.datasets_list:
            list_path = os.path.abspath(os.path.expanduser(args.datasets_list))
            if not os.path.isfile(list_path):
                print(f"[ERROR] File not found --datasets-list: {list_path}")
                return
        elif use_workspace:
            auto_list = os.path.join(raw_source_dir, DEFAULT_DATASETS_LIST_FILE)
            list_path = auto_list if os.path.isfile(auto_list) else None
        else:
            list_path = None
        if list_path:
            # In workspace mode, datasets_list.txt describes external sources,
            # of which you need to prepare copies in datasets and include them in the index.
            if use_workspace:
                entries = _load_datasets_list_file(list_path)
                for src_path in entries:
                    if not os.path.exists(src_path):
                        print(f"[WARNING] Skipping from datasets-list: path not found: {src_path}")
                        continue
                    base_name = (
                        os.path.splitext(os.path.basename(src_path))[0]
                        if src_path.lower().endswith(".zip")
                        else os.path.basename(src_path)
                    )
                    logical_name = _unique_dataset_key(base_name, used_names)
                    dst_dir = os.path.join(layout.datasets, logical_name)
                    sig = _compute_source_signature(src_path)
                    prev_sig = None
                    if os.path.isfile(output_file):
                        try:
                            with open(output_file, "r", encoding="utf-8") as pf:
                                prev_j = json.load(pf)
                            if isinstance(prev_j, dict) and isinstance(prev_j.get(logical_name), dict):
                                prev_sig = prev_j[logical_name].get(SOURCE_SIGNATURE_KEY)
                        except Exception:
                            prev_sig = None
                    if src_path.lower().endswith(".zip"):
                        try:
                            extracted = _extract_zip_for_scan(
                                src_path,
                                os.path.join(output_dir, "tmp", "datasets_list_extract"),
                            )
                        except Exception as e:
                            print(f"[WARNING] Skipping archives from datasets-list {src_path!r}: {e}")
                            continue
                        source_for_copy = extracted
                    else:
                        source_for_copy = src_path
                    _sync_one_source(
                        logical_name=logical_name,
                        source_for_copy=source_for_copy,
                        source_ref=src_path,
                        source_signature=sig,
                    )
            else:
                _append_roots_from_datasets_list(
                    list_path=list_path,
                    folder_roots=folder_roots,
                    used_names=used_names,
                    use_workspace=use_workspace,
                    layout=layout,
                    output_dir=output_dir,
                )

        for folder_name in os.listdir(datasets_dir):
            folder_path = os.path.join(datasets_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            overrides: dict[str, Any] = {}
            if use_workspace:
                rel = os.path.relpath(folder_path, layout.root)
                overrides = {"data_path": rel}
            folder_roots.append((folder_name, folder_path, overrides))
            used_names.add(folder_name)

        datasets_info, class_names = _run_scan_folder_roots(folder_roots)

    if use_workspace and args.mode == "scan" and args.purge_processed_raw:
        root_raw = os.path.abspath(layout.raw_data)
        list_path = os.path.abspath(os.path.join(root_raw, DEFAULT_DATASETS_LIST_FILE))
        purge_candidates = sorted(
            p
            for p in processed_raw_sources
            if p.startswith(root_raw + os.sep) and os.path.abspath(p) != list_path
        )
        if _confirm_purge_processed_raw(purge_candidates):
            removed, failed = _purge_raw_sources(purge_candidates)
            print(f"[INFO] Removed from raw_data: {removed}, errors: {failed}")
        else:
            print("[INFO] Removal of processed sources from raw_data cancelled.")

    for name in list(datasets_info.keys()):
        if name in previous_info and isinstance(previous_info[name], dict):
            datasets_info[name] = _merge_preserved_dataset_fields(
                datasets_info[name], previous_info[name]
            )
        if use_workspace:
            ds_path = os.path.join(layout.datasets, name)
            current_hash = _dataset_content_hash(ds_path)
            if current_hash:
                prev_hash = datasets_info[name].get(DATASET_HASH_KEY)
                was_modified = bool(datasets_info[name].get(MODIFIED_KEY, False))
                if prev_hash and prev_hash != current_hash and not was_modified:
                    print(
                        f"[WARNING] Dataset {name!r} was manually changed in datasets; "
                        "set modified=true, synchronization from raw_data is disabled."
                    )
                    datasets_info[name][MODIFIED_KEY] = True
                elif not was_modified:
                    datasets_info[name][MODIFIED_KEY] = False
                else:
                    datasets_info[name][MODIFIED_KEY] = True
                datasets_info[name][DATASET_HASH_KEY] = current_hash
            elif MODIFIED_KEY not in datasets_info[name]:
                datasets_info[name][MODIFIED_KEY] = bool(datasets_info[name].get(MODIFIED_KEY, False))

    if use_workspace:
        for name, entry in datasets_info.items():
            if not isinstance(entry, dict):
                continue
            try:
                dataset_root = resolve_or_extract_dataset_root(layout.root, name, entry, datasets_dir)
            except Exception:
                dataset_root = os.path.join(datasets_dir, name)
            if not os.path.isdir(dataset_root):
                continue
            try:
                _ensure_scan_initial_passport(
                    dataset_dir=dataset_root,
                    dataset_name=name,
                    entry=entry,
                    workspace_root=layout.root,
                )
            except Exception as e:
                print(f"[WARNING] Failed to write initial passport for {name!r}: {e}")

    old_ds_keys: Set[str] = set()
    if 'previous_info_for_diff' in locals() and isinstance(previous_info_for_diff, dict):
        old_ds_keys = {str(k) for k in previous_info_for_diff.keys()}
    elif isinstance(previous_info, dict):
        old_ds_keys = {str(k) for k in previous_info.keys()}
    new_ds_keys = set(datasets_info.keys())
    ds_added, ds_removed = _sorted_diff(old_ds_keys, new_ds_keys)

    previous_cn_keys: Set[str] = set()
    if os.path.isfile(output_class_names_file):
        try:
            with open(output_class_names_file, "r", encoding="utf-8") as f:
                prev_cn = json.load(f)
            if isinstance(prev_cn, dict):
                previous_cn_keys = set(prev_cn.keys())
        except Exception as e:
            print(f"[WARNING] Failed to read previous {OUTPUT_CLASS_NAMES_FILE} for summary: {e}")

    new_cn_keys = set(class_names.keys())
    cn_added, cn_removed = _sorted_diff(previous_cn_keys, new_cn_keys)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(datasets_info, f, ensure_ascii=False, indent=4)
        print(f"[OK] Information saved successfully in {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to write JSON: {e}")
        return

    try:
        with open(output_class_names_file, "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=4)
        print(f"[OK] Information saved successfully in {output_class_names_file}")
    except Exception as e:
        print(f"[ERROR] Failed to write JSON: {e}")
        return

    summary_path = _write_scan_summary(
        output_dir=output_dir,
        datasets_final=datasets_info,
        class_names_final=class_names,
        datasets_added=ds_added,
        datasets_removed=ds_removed,
        class_names_added=cn_added,
        class_names_removed=cn_removed,
    )
    _print_scan_report(
        summary_path=summary_path,
        datasets_added=ds_added,
        datasets_removed=ds_removed,
        class_names_added=cn_added,
        class_names_removed=cn_removed,
        datasets_final_count=len(datasets_info),
        class_names_final_count=len(class_names),
        had_previous_datasets=bool(old_ds_keys),
        had_previous_class_names=bool(previous_cn_keys),
    )


if __name__ == "__main__":
    main()
