from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from smartrain.services.datasets.cvat11_converter import (
    YOLO_IMAGE_EXTS,
    build_cvat11_annotations_xml,
)

_CVS_IMAGE_EXTS = YOLO_IMAGE_EXTS


def _is_detection_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    for key in ("x", "y", "width", "height"):
        if key not in obj:
            return False
    if "class_name" not in obj and "classId" not in obj:
        return False
    return True


def _json_has_cvsdcldet_detections(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    detections = data.get("detections")
    if not isinstance(detections, list):
        return False
    if not detections:
        return True
    return any(_is_detection_dict(d) for d in detections)


def _find_image_for_stem(directory: Path, stem: str) -> Optional[Path]:
    for ext in _CVS_IMAGE_EXTS:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def is_cvsdcldet_dir(folder_path: str | Path) -> bool:
    """Return True when folder looks like CvsDclDet (paired image + json with detections)."""
    root = Path(folder_path)
    if not root.is_dir():
        return False
    json_files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
    if not json_files:
        return False
    matched = 0
    for jp in json_files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _json_has_cvsdcldet_detections(data):
            continue
        if _find_image_for_stem(root, jp.stem) is not None:
            matched += 1
    return matched > 0


def collect_cvsdcldet_pairs(source_dir: Path) -> List[Tuple[Path, Path]]:
    """Return sorted (image_path, json_path) pairs with matching stems."""
    root = Path(source_dir)
    pairs: List[Tuple[Path, Path]] = []
    for jp in sorted(root.glob("*.json")):
        if not jp.is_file():
            continue
        img = _find_image_for_stem(root, jp.stem)
        if img is None:
            print(f"[WARNING] JSON without image, skipping: {jp.name}")
            continue
        pairs.append((img, jp))
    orphan_images = 0
    json_stems = {jp.stem for _, jp in pairs}
    for ext in _CVS_IMAGE_EXTS:
        for img in root.glob(f"*{ext}"):
            if img.stem not in json_stems:
                orphan_images += 1
    if orphan_images:
        print(f"[WARNING] {orphan_images} image(s) without matching JSON in {root}")
    return pairs


def collect_cvsdcldet_class_names(source_dir: Path) -> List[str]:
    names: set[str] = set()
    for _img, jp in collect_cvsdcldet_pairs(source_dir):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for det in data.get("detections") or []:
            if not isinstance(det, dict):
                continue
            label = _detection_label(det)
            if label:
                names.add(label)
    return sorted(names)


def parse_rename_classes_args(pairs: Sequence[Sequence[str]] | None) -> Dict[str, str]:
    """
    Parse --rename-classes old:new pairs (repeatable flag).
    Raises ValueError on duplicate source keys.
    """
    out: Dict[str, str] = {}
    if not pairs:
        return out
    for item in pairs:
        if len(item) != 2:
            raise ValueError("Each --rename-classes entry must be old:new")
        old, new = str(item[0]).strip(), str(item[1]).strip()
        if not old or not new:
            raise ValueError("Empty class name in --rename-classes")
        if old in out and out[old] != new:
            raise ValueError(f"Duplicate rename source class: {old!r}")
        out[old] = new
    return out


def _detection_label(det: dict) -> str:
    name = det.get("class_name")
    if name is not None and str(name).strip():
        return str(name).strip()
    cid = det.get("classId")
    if cid is not None:
        return str(cid)
    return ""


def _apply_rename(label: str, class_rename: Dict[str, str] | None) -> str:
    if not class_rename:
        return label
    return class_rename.get(label, label)


def _detection_to_bbox(
    det: dict,
    *,
    img_w: int,
    img_h: int,
    class_rename: Dict[str, str] | None,
) -> Optional[Tuple[str, Tuple[float, float, float, float]]]:
    try:
        x = float(det["x"])
        y = float(det["y"])
        w = float(det["width"])
        h = float(det["height"])
    except (TypeError, ValueError, KeyError):
        return None
    if w <= 0 or h <= 0:
        return None
    label = _apply_rename(_detection_label(det), class_rename)
    if not label:
        return None
    xtl = max(0.0, min(x, float(img_w)))
    ytl = max(0.0, min(y, float(img_h)))
    xbr = max(0.0, min(x + w, float(img_w)))
    ybr = max(0.0, min(y + h, float(img_h)))
    if xbr <= xtl or ybr <= ytl:
        return None
    return label, (xtl, ytl, xbr, ybr)


def _pack_cvat11_zip(
    *,
    output_dir: Path,
    task_name: str,
    zip_path: Path,
    force: bool,
) -> None:
    if zip_path.exists():
        if not force:
            raise FileExistsError(f"Zip already exists: {zip_path}. Use --force to overwrite.")
        zip_path.unlink()
    images_dir = output_dir / "images"
    ann = output_dir / "annotations.xml"
    if not ann.is_file():
        raise FileNotFoundError(f"annotations.xml not found in {output_dir}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(ann, arcname=str(Path(task_name) / "annotations.xml"))
        if images_dir.is_dir():
            for p in sorted(images_dir.iterdir()):
                if p.is_file():
                    zf.write(p, arcname=str(Path(task_name) / "images" / p.name))


def convert_cvsdcldet_to_cvat11(
    *,
    source_dir: Path,
    output_dir: Path,
    task_name: str | None = None,
    class_rename: Dict[str, str] | None = None,
    force: bool = False,
    create_zip: bool = False,
    zip_path: Path | None = None,
) -> Dict[str, Any]:
    """
    Convert CvsDclDet folder (paired image + json) to CVAT for images 1.1 layout.
    Writes annotations.xml + images/ under output_dir; optionally packs a CVAT zip.
    """
    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not is_cvsdcldet_dir(source_dir):
        raise ValueError(f"Not a CvsDclDet directory: {source_dir}")

    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --force to overwrite.")
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()

    effective_task_name = (task_name or source_dir.name).strip() or "cvsdcldet_task"
    pairs = collect_cvsdcldet_pairs(source_dir)
    if not pairs:
        raise ValueError(f"No image+json pairs found in {source_dir}")

    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    cvat_images: List[Tuple[str, int, int, List[Tuple[str, Tuple[float, float, float, float]]]]] = []
    all_labels: List[str] = []
    boxes_count = 0

    for img_path, json_path in pairs:
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        img_w = int(img_w)
        img_h = int(img_h)

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARNING] Failed to read {json_path.name}: {e}")
            data = {}

        boxes: List[Tuple[str, Tuple[float, float, float, float]]] = []
        for det in data.get("detections") or []:
            if not isinstance(det, dict):
                continue
            parsed = _detection_to_bbox(det, img_w=img_w, img_h=img_h, class_rename=class_rename)
            if parsed is None:
                continue
            boxes.append(parsed)
            all_labels.append(parsed[0])
            boxes_count += 1

        dest_img = images_out / img_path.name
        shutil.copy2(img_path, dest_img)
        cvat_images.append((img_path.name, img_w, img_h, boxes))

    unique_labels = sorted(set(all_labels))
    annotations_xml = build_cvat11_annotations_xml(
        task_name=effective_task_name,
        images=cvat_images,
        labels=unique_labels,
    )
    (output_dir / "annotations.xml").write_text(annotations_xml, encoding="utf-8")

    effective_zip: Path | None = None
    if create_zip:
        effective_zip = zip_path or Path(str(output_dir) + ".cvat11.zip")
        effective_zip = Path(effective_zip).expanduser().resolve()
        _pack_cvat11_zip(
            output_dir=output_dir,
            task_name=effective_task_name,
            zip_path=effective_zip,
            force=force,
        )

    return {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "task_name": effective_task_name,
        "classes": unique_labels,
        "nc": len(unique_labels),
        "images_count": len(cvat_images),
        "boxes_count": boxes_count,
        "zip_path": str(effective_zip) if effective_zip else None,
    }
