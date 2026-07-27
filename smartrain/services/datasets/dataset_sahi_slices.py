"""Prepare a YOLO dataset of SAHI-style sliding-window slices for fine-tune.

Recipe (arXiv:2202.06934): ``prepare-slices → train → sahi infer``.
Does not require the ``sahi`` package — pure crop + label transform.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from smartrain.core.runtime.workspace_paths import WorkspaceLayout
from smartrain.services.datasets.dataset_access import (
    iter_image_label_buckets,
    resolve_dataset_root_for_entry,
)
from smartrain.services.datasets.dataset_cli_catalog import load_datasets_catalog
from smartrain.services.datasets.dataset_cli_common import update_datasets_sidecar
from smartrain.services.datasets.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.services.datasets.dataset_roi_yolo import _transform_label_line
from smartrain.services.datasets.data_yaml_writer import write_data_yaml_from_names

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def iter_slice_windows(
    iw: int,
    ih: int,
    *,
    slice_w: int = 640,
    slice_h: int = 640,
    overlap_w: float = 0.2,
    overlap_h: float = 0.2,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(x1, y1, x2, y2)`` crop windows covering the image."""
    sw = max(1, int(slice_w))
    sh = max(1, int(slice_h))
    step_x = max(1, int(round(sw * (1.0 - float(overlap_w)))))
    step_y = max(1, int(round(sh * (1.0 - float(overlap_h)))))
    xs = list(range(0, max(1, iw - sw + 1), step_x))
    ys = list(range(0, max(1, ih - sh + 1), step_y))
    if not xs or xs[-1] + sw < iw:
        xs.append(max(0, iw - sw))
    if not ys or ys[-1] + sh < ih:
        ys.append(max(0, ih - sh))
    seen: set[tuple[int, int]] = set()
    for y0 in ys:
        for x0 in xs:
            key = (x0, y0)
            if key in seen:
                continue
            seen.add(key)
            x1, y1 = int(x0), int(y0)
            x2, y2 = min(iw, x0 + sw), min(ih, y0 + sh)
            if x2 > x1 and y2 > y1:
                yield (x1, y1, x2, y2)


def _detect_split(images_path: str) -> str:
    low = images_path.replace("\\", "/").lower()
    for name in ("train", "val", "valid", "test"):
        if f"/{name}/" in f"/{low}/" or low.endswith(f"/{name}") or f"/{name}/images" in low:
            return "val" if name == "valid" else name
    return "train"


def prepare_sahi_slices_dataset(
    *,
    workspace: str,
    dataset: str,
    output_name: str | None = None,
    slice_h: int = 640,
    slice_w: int = 640,
    overlap_h: float = 0.2,
    overlap_w: float = 0.2,
) -> dict[str, Any]:
    """Create ``datasets/<name>_sahi_slices`` with sliced images/labels."""
    layout = WorkspaceLayout(workspace)
    catalog = load_datasets_catalog(layout)
    if dataset not in catalog:
        raise KeyError(f"Unknown dataset: {dataset}")
    entry = catalog[dataset]
    if not isinstance(entry, dict):
        raise TypeError(f"Invalid catalog entry for {dataset!r}")

    src_root = resolve_dataset_root_for_entry(
        dataset,
        entry,
        workspace_root=layout.root,
        source_catalog_dir=layout.datasets,
        legacy_source_parent=layout.datasets,
    )
    buckets = iter_image_label_buckets(
        src_root,
        str(entry.get("structure", "split")),
        entry,
        dataset_name=dataset,
        temp_root=os.path.join(layout.root, "tmp"),
        exclude_test=False,
    )

    class_map = entry.get("classes", {}) if isinstance(entry.get("classes"), dict) else {}
    names = [k for k, _ in sorted(((str(k), int(v)) for k, v in class_map.items()), key=lambda kv: kv[1])]
    if not names:
        names = ["class_0"]

    preferred = (output_name or f"{dataset}_sahi_slices").strip() or f"{dataset}_sahi_slices"
    out_key = next_dataset_name(layout.datasets, preferred)
    out_dir = os.path.join(layout.datasets, out_key)
    os.makedirs(out_dir, exist_ok=True)

    n_written = 0
    for images_path, labels_path in buckets:
        split = _detect_split(images_path)
        for name in sorted(os.listdir(images_path)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            img_path = os.path.join(images_path, name)
            lbl_path = os.path.join(labels_path, f"{stem}.txt")
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    iw, ih = im.size
                    label_lines: list[str] = []
                    if os.path.isfile(lbl_path):
                        with open(lbl_path, "r", encoding="utf-8") as f:
                            label_lines = f.readlines()
                    for si, (x1, y1, x2, y2) in enumerate(
                        iter_slice_windows(
                            iw,
                            ih,
                            slice_w=slice_w,
                            slice_h=slice_h,
                            overlap_w=overlap_w,
                            overlap_h=overlap_h,
                        )
                    ):
                        crop = (x1, y1, x2, y2)
                        out_lines: list[str] = []
                        for line in label_lines:
                            transformed = _transform_label_line(line, crop, iw, ih)
                            if transformed:
                                out_lines.append(transformed)
                        img_out_dir = os.path.join(out_dir, split, "images")
                        lbl_out_dir = os.path.join(out_dir, split, "labels")
                        os.makedirs(img_out_dir, exist_ok=True)
                        os.makedirs(lbl_out_dir, exist_ok=True)
                        out_stem = f"{stem}_s{si:04d}"
                        out_img = os.path.join(img_out_dir, f"{out_stem}.jpg")
                        out_lbl = os.path.join(lbl_out_dir, f"{out_stem}.txt")
                        im.crop((x1, y1, x2, y2)).save(out_img, quality=95)
                        with open(out_lbl, "w", encoding="utf-8") as f:
                            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
                        n_written += 1
            except Exception as exc:
                print(f"[WARN] skip {img_path}: {exc}")
                continue

    write_data_yaml_from_names(
        out_dir,
        names,
        train_rel="train/images",
        val_rel="val/images",
        test_rel="test/images",
    )
    write_dataset_passport(
        output_dataset_dir=out_dir,
        command="sahi prepare-slices",
        source_datasets=[{"name": dataset, "path": src_root}],
        parameters={
            "slice_h": slice_h,
            "slice_w": slice_w,
            "overlap_h": overlap_h,
            "overlap_w": overlap_w,
        },
        transformations=[{"type": "sahi_slices", "citation": "arXiv:2202.06934"}],
        workspace_root=layout.root,
    )
    out_class_map = {n: i for i, n in enumerate(names)}
    update_datasets_sidecar(
        layout=layout,
        output_key=out_key,
        class_map=out_class_map,
        target_dir=out_dir,
        output_hash="",
    )
    return {"dataset": out_key, "path": out_dir, "slices_written": n_written}
