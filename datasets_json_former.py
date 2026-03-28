import os
import yaml
import json
import sys
import argparse
from typing import Any, Dict, Optional

from workspace_paths import (
    WORKSPACE_ENV_VAR,
    WorkspaceLayout,
    resolve_workspace_root,
    resolve_dataset_root,
    DATASETS_INFO_FILE,
    CLASS_NAMES_FILE,
)


sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_FILE = DATASETS_INFO_FILE
OUTPUT_CLASS_NAMES_FILE = CLASS_NAMES_FILE


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


def parse_args():
    parser = argparse.ArgumentParser(
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

    return parser.parse_args()


# Поля записи датасета, сохраняемые при пересканировании (вручную в JSON)
_PRESERVED_DATASET_INFO_KEYS = ("roi_auto", "tags", "data_path")


def _merge_preserved_dataset_fields(fresh: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Дополняет свежую запись из скана полями roi_auto/tags из старого datasets_info.json."""
    if not previous:
        return fresh
    out = dict(fresh)
    for key in _PRESERVED_DATASET_INFO_KEYS:
        if key in previous:
            out[key] = previous[key]
    return out


def _run_scan_folder_roots(folder_roots: list[tuple[str, str]]) -> tuple[dict, dict]:
    """folder_roots: список (logical_name, folder_path на диске)."""
    datasets_info: dict = {}
    class_names: dict = {}

    for logical_name, folder_path in folder_roots:
        if not os.path.isdir(folder_path):
            print(f"[WARNING] Пропуск {logical_name!r}: нет каталога {folder_path}")
            continue
        info = process_dataset(folder_path, logical_name)
        if info:
            datasets_info[logical_name] = info
            for class_name in info["classes"]:
                class_names[class_name] = class_name
    return datasets_info, class_names


def main():
    args = parse_args()

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
                folder_roots.append((name, os.path.join(datasets_dir, name)))
            else:
                dp = prev_entry["data_path"]
                if not isinstance(dp, str):
                    print(f"[WARNING] Пропуск {name!r}: data_path не строка")
                    continue
                root_path = resolve_dataset_root(layout.root, name, prev_entry, datasets_dir)
                folder_roots.append((name, root_path))
        datasets_info, class_names = _run_scan_folder_roots(folder_roots)
        previous_info = previous_full
    else:
        if not os.path.exists(datasets_dir):
            print(f"[ERROR] Папка '{datasets_dir}' не найдена.")
            return
        folder_roots = []
        for folder_name in os.listdir(datasets_dir):
            folder_path = os.path.join(datasets_dir, folder_name)
            if os.path.isdir(folder_path):
                folder_roots.append((folder_name, folder_path))
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

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(datasets_info, f, ensure_ascii=False, indent=4)
        print(f"[OK] Информация успешно сохранена в {output_file}")
    except Exception as e:
        print(f"[ERROR] Не удалось записать JSON: {e}")

    try:
        with open(output_class_names_file, "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=4)
        print(f"[OK] Информация успешно сохранена в {output_class_names_file}")
    except Exception as e:
        print(f"[ERROR] Не удалось записать JSON: {e}")


if __name__ == "__main__":
    main()
