import os
import yaml
import json
import sys
import argparse
import hashlib
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

from smartrain.cli_argparse import CliArgumentParser
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
    """Чтение obj.names файла (по одному классу на строку)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        return names
    except Exception as e:
        print(f"[ERROR] Не удалось прочитать {file_path}: {e}")
        return None


def load_obj_data(file_path):
    """Парсинг obj.data файла для получения количества классов"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Ищем строку classes = X
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('classes'):
                parts = line.split('=')
                if len(parts) == 2:
                    return int(parts[1].strip())
        return None
    except Exception as e:
        print(f"[ERROR] Не удалось прочитать {file_path}: {e}")
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
    Минимальная валидация, что annotations.xml похож на CVAT for images 1.1:
    - root == <annotations>
    - есть хотя бы один <image>
    - для images-task обычно есть <box> внутри <image> (но допускаем пустую разметку)
    - если есть <version>, то она должна быть 1.1 (иначе считаем неподходящим)
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

    # Если совсем нет box, это может быть пустая разметка; считаем валидной.
    # Но если есть box, проверим наличие обязательных атрибутов.
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
    Пары каталогов (images…, labels…) для плоских датасетов YOLO:
    — классический flat: одна пара (images, labels), если есть файлы в корне;
    — экспорт CVAT «Ultralytics YOLO Detection 1.0»: images/<подпапка>/ и labels/<та же подпапка>/
      (имя подпапки произвольное, не только train/val/test).
    Если встречаются и корневые файлы, и общие подпапки — возвращаются оба варианта.
    Для вложенного split (images/train/…) возвращает пустой список — используйте nested_split.
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

    # Проверка на формат Darknet YOLO
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
        print(f"[ERROR] Не удалось прочитать {file_path}: {e}")
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
        print(f"[WARNING] В папке {folder_path} число изображений ({images_count}) "
              f"не совпадает с числом аннотаций ({labels_count})")
        return images_count, labels_count



def process_dataset(folder_path, folder_name):
    yaml_path = find_yaml_file(folder_path)
    
    names = None
    structure = detect_structure(folder_path)

    # Попытка загрузить из YAML (формат YOLOv8)
    if yaml_path:
        data = load_yaml(yaml_path)
        if data and "names" in data:
            names = data["names"]
            if isinstance(names, list):
                pass
            elif isinstance(names, dict):
                names = [v for k, v in sorted(names.items())]
            else:
                print(f"[ERROR] Поле 'names' в {yaml_path} имеет неверный формат")
                return None

    # Если не нашли YAML, пробуем формат Darknet
    if not names and structure == "darknet":
        obj_names_path = find_obj_names_file(folder_path)
        if obj_names_path:
            names = load_obj_names(obj_names_path)
            if not names:
                print(f"[WARNING] Не удалось загрузить классы из {obj_names_path} — пропуск")
                return None
        else:
            print(f"[WARNING] В папке {folder_name} не найден obj.names — пропуск")
            return None

    # Если CVAT 1.1 extracted dataset
    if not names and structure == "cvat11":
        xml_path = _find_cvat_annotations_xml(folder_path)
        if xml_path:
            names = _load_cvat11_label_names(xml_path)
            if not names:
                print(f"[WARNING] CVAT 1.1: не удалось извлечь labels из {xml_path} — пропуск")
                return None
        else:
            print(f"[WARNING] CVAT 1.1: не найден annotations.xml — пропуск")
            return None

    # Если ничего не нашли
    if not names:
        print(f"[WARNING] В папке {folder_name} не найден data.yaml или obj.names — пропуск")
        return None

    elements_count = count_elements(folder_path, structure)

    return {
        "classes": {name: idx for idx, name in enumerate(names)},
        "structure": structure,
        "elements_count": elements_count
    }


def build_datasets_json_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        description="Обработка датасетов и создание JSON файлов с информацией о классах и структуре"
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Корень workspace (иначе {WORKSPACE_ENV_VAR}); JSON пишутся в source_datasets/",
    )

    parser.add_argument(
        "--datasets-path",
        type=str,
        default=None,
        help="Режим без workspace: корень каталогов датасетов для сканирования",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Режим без workspace: каталог для datasets_info.json и class_names.json",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=("scan", "refresh"),
        default="scan",
        help="scan: обход подкаталогов source_datasets (или --datasets-path); "
        "refresh: пересканировать только data_path из существующего datasets_info.json",
    )
    parser.add_argument(
        "--datasets-list",
        type=str,
        default=None,
        help="TXT-файл со списком путей к датасетам (по одному на строку): директории или .zip",
    )

    return parser


def parse_args(argv=None):
    return build_datasets_json_arg_parser().parse_args(argv)


# Поля записи датасета, сохраняемые при пересканировании (вручную в JSON)
_PRESERVED_DATASET_INFO_KEYS = ("roi_auto", "tags", "data_path")


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
    """Пишет datasets_scan_summary.json; возвращает путь к файлу."""
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
    print(f"[INFO] Итоговых датасетов в {OUTPUT_FILE}: {datasets_final_count}")
    if datasets_added:
        print(f"[INFO] Добавлены датасеты ({len(datasets_added)}): {', '.join(datasets_added)}")
    if datasets_removed:
        print(f"[INFO] Исключены из каталога ({len(datasets_removed)}): {', '.join(datasets_removed)}")
    if not datasets_added and not datasets_removed and had_previous_datasets:
        print("[INFO] Состав датасетов относительно прошлого файла не изменился.")
    print(f"[INFO] Итоговых имён классов в {OUTPUT_CLASS_NAMES_FILE}: {class_names_final_count}")
    if class_names_added:
        print(f"[INFO] Новые имена классов ({len(class_names_added)}): {', '.join(class_names_added)}")
    if class_names_removed:
        print(f"[INFO] Удалённые имена классов ({len(class_names_removed)}): {', '.join(class_names_removed)}")
    if not class_names_added and not class_names_removed and had_previous_class_names:
        print("[INFO] Состав class_names относительно прошлого файла не изменился.")
    print(f"[OK] Сводка сохранена: {summary_path}")


def _merge_preserved_dataset_fields(fresh: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Дополняет свежую запись из скана полями roi_auto/tags из старого datasets_info.json."""
    if not previous:
        return fresh
    out = dict(fresh)
    for key in _PRESERVED_DATASET_INFO_KEYS:
        if key in previous:
            out[key] = previous[key]
    return out


def _run_scan_folder_roots(folder_roots: list[tuple[str, str, dict]]) -> tuple[dict, dict]:
    """folder_roots: список (logical_name, folder_path на диске, overrides для datasets_info)."""
    datasets_info: dict = {}
    class_names: dict = {}

    for logical_name, folder_path, overrides in folder_roots:
        if not os.path.isdir(folder_path):
            print(f"[WARNING] Пропуск {logical_name!r}: нет каталога {folder_path}")
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
    """Возвращает path для data_path: относительный к workspace, если возможно."""
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
            print(f"[WARNING] Пропуск из datasets-list: путь не найден: {src_path}")
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
                        layout.source_datasets,
                    )
                else:
                    extracted = _extract_zip_for_scan(
                        src_path,
                        os.path.join(output_dir, "tmp", "datasets_list_extract"),
                    )
            except Exception as e:
                print(f"[WARNING] Пропуск архива из datasets-list {src_path!r}: {e}")
                continue
            folder_roots.append((logical_name, extracted, {"data_path": data_path_value}))
            continue

        print(
            f"[WARNING] Пропуск из datasets-list: поддерживаются только директории и .zip, "
            f"получено: {src_path}"
        )


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
        os.makedirs(layout.source_datasets, exist_ok=True)

    if use_workspace:
        output_dir = layout.source_datasets
        output_file = os.path.join(output_dir, OUTPUT_FILE)
        output_class_names_file = os.path.join(output_dir, OUTPUT_CLASS_NAMES_FILE)
        datasets_dir = layout.source_datasets
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

    if args.mode == "refresh" and not use_workspace:
        print("[ERROR] Режим --mode refresh поддерживается только без --datasets-path (через workspace).")
        return

    if use_workspace and args.mode == "refresh":
        if not os.path.isfile(output_file):
            print(f"[ERROR] Режим refresh: не найден {output_file}")
            return
        with open(output_file, "r", encoding="utf-8") as f:
            previous_full = json.load(f)
        if not isinstance(previous_full, dict):
            print(f"[ERROR] {output_file}: ожидается объект JSON с датасетами.")
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
                    print(f"[WARNING] Пропуск {name!r}: data_path не строка")
                    continue
                root_path = resolve_or_extract_dataset_root(layout.root, name, prev_entry, datasets_dir)
                folder_roots.append((name, root_path, {"data_path": dp}))
        datasets_info, class_names = _run_scan_folder_roots(folder_roots)
        previous_info = previous_full
    else:
        if not os.path.exists(datasets_dir):
            print(f"[ERROR] Папка '{datasets_dir}' не найдена.")
            return
        folder_roots = []
        used_names: set[str] = set()
        for folder_name in os.listdir(datasets_dir):
            folder_path = os.path.join(datasets_dir, folder_name)
            if os.path.isdir(folder_path):
                folder_roots.append((folder_name, folder_path, {}))
                used_names.add(folder_name)
            elif use_workspace and folder_name.lower().endswith(".zip"):
                zip_path = folder_path
                logical_name = os.path.splitext(folder_name)[0]
                existing_dir = os.path.join(datasets_dir, logical_name)
                if _dir_has_content(existing_dir):
                    print(
                        f"[INFO] Пропуск архива {folder_name!r}: найден непустой каталог "
                        f"{existing_dir!r}"
                    )
                    continue
                try:
                    extracted = resolve_or_extract_dataset_root(
                        layout.root,
                        logical_name,
                        {"data_path": os.path.relpath(zip_path, layout.root)},
                        datasets_dir,
                    )
                except Exception as e:
                    print(f"[WARNING] Пропуск {folder_name!r}: не удалось распаковать zip ({e})")
                    continue
                folder_roots.append(
                    (
                        logical_name,
                        extracted,
                        {"data_path": os.path.relpath(zip_path, layout.root)},
                    )
                )
                used_names.add(logical_name)
        if args.datasets_list:
            list_path = os.path.abspath(os.path.expanduser(args.datasets_list))
            if not os.path.isfile(list_path):
                print(f"[ERROR] Не найден файл --datasets-list: {list_path}")
                return
        elif use_workspace:
            auto_list = os.path.join(datasets_dir, DEFAULT_DATASETS_LIST_FILE)
            list_path = auto_list if os.path.isfile(auto_list) else None
        else:
            list_path = None
        if list_path:
            _append_roots_from_datasets_list(
                list_path=list_path,
                folder_roots=folder_roots,
                used_names=used_names,
                use_workspace=use_workspace,
                layout=layout,
                output_dir=output_dir,
            )
        datasets_info, class_names = _run_scan_folder_roots(folder_roots)
        previous_info = {}
        if os.path.isfile(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    previous_info = json.load(f)
            except Exception as e:
                print(f"[WARNING] Не удалось прочитать существующий {output_file}: {e}")

    for name in list(datasets_info.keys()):
        if name in previous_info and isinstance(previous_info[name], dict):
            datasets_info[name] = _merge_preserved_dataset_fields(
                datasets_info[name], previous_info[name]
            )

    old_ds_keys: Set[str] = set()
    if isinstance(previous_info, dict):
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
            print(f"[WARNING] Не удалось прочитать прежний {OUTPUT_CLASS_NAMES_FILE} для сводки: {e}")

    new_cn_keys = set(class_names.keys())
    cn_added, cn_removed = _sorted_diff(previous_cn_keys, new_cn_keys)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(datasets_info, f, ensure_ascii=False, indent=4)
        print(f"[OK] Информация успешно сохранена в {output_file}")
    except Exception as e:
        print(f"[ERROR] Не удалось записать JSON: {e}")
        return

    try:
        with open(output_class_names_file, "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=4)
        print(f"[OK] Информация успешно сохранена в {output_class_names_file}")
    except Exception as e:
        print(f"[ERROR] Не удалось записать JSON: {e}")
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
