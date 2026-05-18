from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Sequence


def _ensure_training_ready_after_copy(
    dataset_root: str,
    *,
    detect_structure_cb: Callable[[str], str],
    find_cvat_annotations_xml_cb: Callable[[str], str | None],
    load_cvat11_label_names_cb: Callable[[str], Sequence[str] | None],
    generate_temp_yolo_labels_cb: Callable[..., object],
) -> bool:
    """
    Normalizes the copied dataset to a form suitable for training.
    Critical case: cvat11 (annotations.xml + images/) -> YOLO labels + data.yaml.
    """

    structure = detect_structure_cb(dataset_root)
    if structure != "cvat11":
        return False

    xml_path = find_cvat_annotations_xml_cb(dataset_root)
    if not xml_path:
        return False

    names = load_cvat11_label_names_cb(xml_path)
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
        generate_temp_yolo_labels_cb(
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
        )
        f.write(
            "# No path: key — Ultralytics uses this file's directory as dataset root.\n"
        )
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("test: images\n\n")
        f.write(f"nc: {len(names)}\n")
        f.write(f"names: {list(names)}\n")

    return True

