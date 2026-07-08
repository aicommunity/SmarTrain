from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Sequence
import yaml


def convert_cvat11_folder_to_yolo_flat(
    source_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    detect_structure_cb: Callable[[str], str],
    find_cvat_annotations_xml_cb: Callable[[str], str | None],
    load_cvat11_label_names_cb: Callable[[str], Sequence[str] | None],
    generate_temp_yolo_labels_cb: Callable[..., object],
    force: bool = False,
) -> dict[str, object]:
    """
    Convert CVAT for images 1.1 folder (annotations.xml + images/) to YOLO flat layout.

    When output_dir is None or equals source_dir, labels and data.yaml are written in-place
    (images/ must already exist). Otherwise images/ is copied to output_dir first.
    """
    source_root = Path(source_dir).expanduser().resolve()
    in_place = output_dir is None or Path(output_dir).expanduser().resolve() == source_root
    target_root = source_root if in_place else Path(output_dir).expanduser().resolve()

    structure = detect_structure_cb(str(source_root))
    if structure != "cvat11":
        raise ValueError(f"Expected cvat11 structure, got {structure!r}: {source_root}")

    xml_path = find_cvat_annotations_xml_cb(str(source_root))
    if not xml_path:
        raise FileNotFoundError(f"annotations.xml not found under {source_root}")

    names = load_cvat11_label_names_cb(xml_path)
    if not names:
        raise ValueError(f"Could not determine class list from {xml_path}")

    if not in_place:
        if target_root.exists():
            if not force:
                raise FileExistsError(f"Output already exists: {target_root}. Use --force to overwrite.")
            if target_root.is_dir():
                shutil.rmtree(target_root)
            else:
                target_root.unlink()
        target_root.mkdir(parents=True, exist_ok=True)
        src_images = source_root / "images"
        dst_images = target_root / "images"
        if src_images.is_dir():
            shutil.copytree(src_images, dst_images)
        xml_src = Path(xml_path)
        if xml_src.is_file():
            xml_dst = target_root / xml_src.name
            if xml_src.parent != source_root:
                xml_dst = target_root / xml_src.relative_to(source_root)
            xml_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(xml_src, xml_dst)

    labels_dir = target_root / "labels"
    if labels_dir.is_dir():
        shutil.rmtree(labels_dir, ignore_errors=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_name_to_id = {name: idx for idx, name in enumerate(names)}
    generate_temp_yolo_labels_cb(
        dataset_root=target_root,
        labels_out_dir=labels_dir,
        class_name_to_id=class_name_to_id,
    )

    data_yaml = target_root / "data.yaml"
    header = (
        "# smartrain (CVAT 1.1): images/ may contain nested subfolders; "
        "labels/ mirrors the same relative paths (YOLO pairing).\n"
        "# No path: key — Ultralytics uses this file's directory as dataset root.\n"
    )
    if in_place:
        header = (
            "# smartrain (CVAT 1.1 scan): images/ may contain nested subfolders; "
            "labels/ mirrors the same relative paths (YOLO pairing).\n"
            "# No path: key — Ultralytics uses this file's directory as dataset root.\n"
        )
    data_yaml.write_text(
        header
        + "train: images\n"
        + "val: images\n"
        + "test: images\n\n"
        + f"nc: {len(names)}\n"
        + f"names: {list(names)}\n",
        encoding="utf-8",
    )

    try:
        payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise ValueError(f"Failed to parse generated data.yaml: {data_yaml} ({e})") from e
    if not isinstance(payload, dict):
        raise ValueError(f"Generated data.yaml has unexpected structure: {data_yaml}")
    train_rel = str(payload.get("train") or "").strip().replace("\\", "/").lstrip("./")
    if not train_rel:
        raise ValueError(f"Generated data.yaml has empty train path: {data_yaml}")
    train_abs = os.path.normpath(os.path.join(str(target_root), train_rel))
    if not os.path.isdir(train_abs):
        raise ValueError(
            "Generated CVAT->YOLO data.yaml points to a missing train directory: "
            f"{train_rel} (resolved: {train_abs})"
        )

    images_count = 0
    images_dir = target_root / "images"
    if images_dir.is_dir():
        from smartrain.services.datasets.cvat11_converter import YOLO_IMAGE_EXTS

        images_count = sum(
            1
            for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in YOLO_IMAGE_EXTS
        )
    labels_count = sum(1 for _ in labels_dir.rglob("*.txt"))

    return {
        "output_dir": str(target_root),
        "nc": len(names),
        "names": list(names),
        "images_count": images_count,
        "labels_count": labels_count,
    }


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

    try:
        convert_cvat11_folder_to_yolo_flat(
            dataset_root,
            None,
            detect_structure_cb=detect_structure_cb,
            find_cvat_annotations_xml_cb=find_cvat_annotations_xml_cb,
            load_cvat11_label_names_cb=load_cvat11_label_names_cb,
            generate_temp_yolo_labels_cb=generate_temp_yolo_labels_cb,
        )
    except Exception as e:
        print(f"[WARNING] CVAT 1.1: failed to generate YOLO labels for {dataset_root}: {e}")
        return False

    return True
