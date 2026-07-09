from __future__ import annotations

from pathlib import Path

from smartrain.services.datasets.yolo_labels import YoloLabel, read_yolo_labels


def label_path_from_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        mapped = parts[:]
        mapped[idx] = "labels"
        return Path(*mapped).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def load_gt_labels(image_path: Path) -> tuple[Path | None, list[YoloLabel]]:
    label_path = label_path_from_image(image_path)
    labels = read_yolo_labels(str(label_path))
    if labels:
        return label_path, labels
    if label_path.is_file():
        return label_path, labels
    return None, []

