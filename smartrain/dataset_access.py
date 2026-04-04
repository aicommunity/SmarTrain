"""
Единый резолв корня датасета (включая zip) и пары каталогов images/labels для всех structure.
Используется dataset_former, dataset_roi_yolo и др.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from smartrain.cvat11_converter import generate_temp_yolo_labels_from_cvat11_extracted
from smartrain.datasets_json_former import yolo_flat_image_label_buckets
from smartrain.workspace_paths import resolve_or_extract_dataset_root


def resolve_dataset_root_for_entry(
    dataset_name: str,
    info: dict[str, Any],
    *,
    workspace_root: str | None,
    source_catalog_dir: str,
    legacy_source_parent: str,
) -> str:
    """
    Абсолютный корень данных датасета: zip распаковывается в кэш workspace при workspace_root;
    в legacy-режиме zip в data_path не распаковывается (как раньше в dataset_former).
    """
    if workspace_root is not None:
        return resolve_or_extract_dataset_root(workspace_root, dataset_name, info, source_catalog_dir)
    if "data_path" in info:
        raw = info["data_path"]
        if not isinstance(raw, str):
            raise TypeError(f"data_path для {dataset_name!r} должен быть строкой.")
        if os.path.isabs(raw):
            return os.path.abspath(raw)
        return os.path.abspath(os.path.join(legacy_source_parent, os.path.normpath(raw)))
    return os.path.join(legacy_source_parent, dataset_name)


def find_dataset_paths(dataset_path: str, structure: str, arg: bool = False) -> list[tuple[str, str]]:
    """Пары (images_dir, labels_dir) для YOLO-раскладок; без cvat11."""
    paths: list[tuple[str, str]] = []
    dataset_splitting = ["train", "val"] if arg else ["train", "val", "test"]
    if structure == "split":
        for subset in dataset_splitting:
            subdir = os.path.join(dataset_path, subset)
            if os.path.exists(os.path.join(subdir, "images")) and os.path.exists(
                os.path.join(subdir, "labels")
            ):
                paths.append((os.path.join(subdir, "images"), os.path.join(subdir, "labels")))
    elif structure in ("flat", "subset_flat"):
        buckets = yolo_flat_image_label_buckets(dataset_path)
        if buckets:
            paths.extend(buckets)
        elif os.path.exists(os.path.join(dataset_path, "images")) and os.path.exists(
            os.path.join(dataset_path, "labels")
        ):
            paths.append((os.path.join(dataset_path, "images"), os.path.join(dataset_path, "labels")))
    elif structure == "nested_split":
        for subset in dataset_splitting:
            img_dir = os.path.join(dataset_path, "images", subset)
            lbl_dir = os.path.join(dataset_path, "labels", subset)
            if os.path.exists(img_dir) and os.path.exists(lbl_dir):
                paths.append((img_dir, lbl_dir))
    elif structure == "darknet":
        obj_train_data_path = os.path.join(dataset_path, "obj_train_data")
        if os.path.exists(obj_train_data_path):
            paths.append((obj_train_data_path, obj_train_data_path))
    return paths


def iter_image_label_buckets(
    dataset_root: str,
    structure: str,
    info: dict[str, Any],
    *,
    dataset_name: str,
    temp_root: str,
    exclude_test: bool = False,
) -> list[tuple[str, str]]:
    """
    Список пар (images_dir, labels_dir). Для cvat11 генерирует временные YOLO .txt в temp_root.
    """
    if structure == "cvat11":
        if "classes" not in info or not isinstance(info["classes"], dict):
            raise ValueError(f"{dataset_name!r}: нет classes для cvat11 в datasets_info.json")
        class_map: dict = info["classes"]
        labels_out = Path(temp_root) / "cvat11_labels" / dataset_name
        if labels_out.exists():
            shutil.rmtree(labels_out)
        images_dir, _images_found, _labels_written = generate_temp_yolo_labels_from_cvat11_extracted(
            dataset_root=Path(dataset_root),
            labels_out_dir=labels_out,
            class_name_to_id={str(k): int(v) for k, v in class_map.items()},
        )
        return [(str(images_dir), str(labels_out))]
    return find_dataset_paths(dataset_root, structure, arg=exclude_test)
