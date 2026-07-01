from __future__ import annotations

import os
from typing import Literal

import cv2
import numpy as np
from PIL import Image

from smartrain.services.datasets.yolo_augment_geom import (
    labels_to_legacy_tuples,
    read_augment_label_file,
    write_augment_label_file,
)
from smartrain.services.datasets.yolo_labels import rotate_yolo_labels_90cw_k

ORTHOGONAL_K = {"cw": 1, "ccw": 3}


def rotate_image_k(img: np.ndarray, k: int) -> np.ndarray:
    kk = int(k) % 4
    if kk == 0:
        return img
    if kk == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if kk == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def apply_orthogonal_rotate(
    image_path: str,
    label_path: str,
    out_img: str,
    out_lbl: str,
    *,
    direction: Literal["cw", "ccw"],
) -> list[tuple[int, float, float, float, float]]:
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    k = ORTHOGONAL_K[direction]
    dst = rotate_image_k(img, k)
    labels = read_augment_label_file(label_path)
    rotated_labels, _new_w, _new_h = rotate_yolo_labels_90cw_k(labels, w=w, h=h, k=k)
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    os.makedirs(os.path.dirname(out_lbl), exist_ok=True)
    Image.fromarray(dst).save(out_img)
    write_augment_label_file(out_lbl, rotated_labels)
    return labels_to_legacy_tuples(rotated_labels)
