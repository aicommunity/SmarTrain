"""Pair YOLO label files with corresponding images under parallel directory trees."""

from __future__ import annotations

from pathlib import Path

from smartrain.services.datasets.cvat11_converter import YOLO_IMAGE_EXTS


def collect_label_image_pairs(images_path: str, labels_path: str) -> list[tuple[str, str]]:
    """
    All YOLO *.txt under labels_path (recursive), paired with images under images_path
    using the same relative path (labels/sub/a.txt -> images/sub/a.jpg).
    """
    pairs: list[tuple[str, str]] = []
    labels_root = Path(labels_path)
    images_root = Path(images_path)
    if not labels_root.is_dir() or not images_root.is_dir():
        return pairs
    for label_path in sorted(labels_root.rglob("*.txt")):
        rel = label_path.relative_to(labels_root)
        parent = rel.parent
        stem = rel.stem
        for ext in YOLO_IMAGE_EXTS:
            cand = images_root / parent / f"{stem}{ext}"
            if cand.is_file():
                pairs.append((str(cand), str(label_path)))
                break
    return pairs
