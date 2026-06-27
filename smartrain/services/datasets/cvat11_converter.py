from __future__ import annotations

import html
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

from PIL import Image

from smartrain.services.datasets.yolo_labels import YoloBBox, YoloSegment, read_yolo_labels, serialize_yolo_labels

YOLO_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class CvatBox:
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float


@dataclass(frozen=True)
class CvatPolygon:
    label: str
    points: Tuple[Tuple[float, float], ...]


def collect_cvat11_meta_label_names(root: ET.Element) -> List[str]:
    """Read label names from CVAT meta (task export or job export)."""
    out: List[str] = []
    seen: set[str] = set()
    for xpath in ("./meta/task/labels/label/name", "./meta/job/labels/label/name"):
        for lb in root.findall(xpath):
            if lb is not None and lb.text:
                name = lb.text.strip()
                if name and name not in seen:
                    seen.add(name)
                    out.append(name)
    return out


def collect_cvat11_shape_label_names(root: ET.Element) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for tag in ("box", "polygon", "polyline"):
        for el in root.findall(f"./image/{tag}"):
            label = str(el.attrib.get("label", "")).strip()
            if label and label not in seen:
                seen.add(label)
                out.append(label)
    return sorted(out)


def load_cvat11_label_names_from_xml(xml_path: str | Path) -> List[str]:
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception:
        return []
    meta = collect_cvat11_meta_label_names(root)
    if meta:
        return meta
    return collect_cvat11_shape_label_names(root)


def _parse_cvat_points_shape(el: ET.Element) -> Optional[CvatPolygon]:
    label = str(el.attrib.get("label", "")).strip()
    if not label:
        return None
    pts: list[tuple[float, float]] = []
    for token in str(el.attrib.get("points", "")).split(";"):
        token = token.strip()
        if not token or "," not in token:
            continue
        xs, ys = token.split(",", 1)
        try:
            pts.append((float(xs), float(ys)))
        except ValueError:
            continue
    if len(pts) < 3:
        return None
    return CvatPolygon(label=label, points=tuple(pts))


def find_cvat_annotations_and_images_dir(extracted_root: Path) -> Tuple[Path, Path]:
    annotations = list(extracted_root.rglob("annotations.xml"))
    if not annotations:
        raise FileNotFoundError("annotations.xml not found inside extracted CVAT zip.")

    for ann in annotations:
        parent = ann.parent
        img_dir = parent / "images"
        if img_dir.exists() and img_dir.is_dir():
            return ann, img_dir

    images_dirs = [p for p in extracted_root.rglob("images") if p.is_dir()]
    if not images_dirs:
        raise FileNotFoundError("images/ folder not found inside extracted CVAT zip.")
    return annotations[0], images_dirs[0]


def is_cvat11_images_xml(xml_path: Path) -> bool:
    """
    Lightweight validation for CVAT for images 1.1 annotations.xml.
    We intentionally keep this permissive to support real-world exports.
    """
    try:
        tree = ET.parse(str(xml_path))
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
    return True


def _cvat_xml_name_to_label_rel(cvat_image_name: str, *, use_nested: bool) -> Path:
    """
    Relative path for a YOLO .txt under labels/, mirroring images/ when use_nested.

    When the image file exists only at images/<basename>, use_nested=False so labels stay flat.
    """
    name = (cvat_image_name or "").strip().replace("\\", "/")
    if not name:
        return Path("_invalid.txt")
    if not use_nested:
        return Path(Path(name).name)

    parts: List[str] = []
    for seg in name.split("/"):
        seg = seg.strip()
        if not seg or seg == ".":
            continue
        if seg == "..":
            if parts:
                parts.pop()
            else:
                return Path(Path(name).name)
        else:
            parts.append(seg)
    if not parts:
        return Path(Path(name).name)
    return Path(*parts)


def _label_path_contained_in_dir(label_file: Path, base_dir: Path) -> bool:
    try:
        label_file.resolve().relative_to(base_dir.resolve())
        return True
    except ValueError:
        return False


def generate_temp_yolo_labels_from_cvat11_extracted(
    *,
    dataset_root: Path,
    labels_out_dir: Path,
    class_name_to_id: Dict[str, int],
) -> Tuple[Path, int, int]:
    """
    Native bridge: read extracted CVAT 1.1 dataset (annotations.xml + images/)
    and generate YOLO txt labels into labels_out_dir WITHOUT copying images.

    Returns:
      images_dir, images_found_count, labels_written_count
    """
    dataset_root = Path(dataset_root)
    labels_out_dir = Path(labels_out_dir)
    _ensure_dir(labels_out_dir)

    xml_path, images_dir = find_cvat_annotations_and_images_dir(dataset_root)
    if not is_cvat11_images_xml(xml_path):
        raise ValueError(f"Not a CVAT 1.1 images annotations.xml: {xml_path}")

    _task_name, _labels_in_meta, images = load_cvat11_images_and_labels(xml_path)

    images_found = 0
    labels_written = 0

    for img in images:
        cvat_image_name = img.get("name", "")
        if not isinstance(cvat_image_name, str) or not cvat_image_name:
            continue

        primary = images_dir / cvat_image_name
        if primary.exists():
            src = primary
            use_nested = True
        else:
            fb = images_dir / Path(cvat_image_name).name
            if not fb.exists():
                continue
            src = fb
            use_nested = False

        images_found += 1

        img_w = int(img.get("width", -1))
        img_h = int(img.get("height", -1))
        if img_w <= 0 or img_h <= 0:
            with Image.open(src) as im:
                img_w, img_h = im.size
                img_w = int(img_w)
                img_h = int(img_h)

        lines: List[str] = []
        for b in img.get("boxes", []):
            if not isinstance(b, CvatBox):
                continue
            class_id = class_name_to_id.get(b.label)
            if class_id is None:
                continue
            yolo = _cvat_box_to_yolo_line(b, class_id=class_id, img_w=img_w, img_h=img_h)
            if yolo:
                lines.append(yolo)

        for poly in img.get("polygons", []):
            if not isinstance(poly, CvatPolygon):
                continue
            class_id = class_name_to_id.get(poly.label)
            if class_id is None:
                continue
            yolo = _cvat_polygon_to_yolo_line(poly, class_id=class_id, img_w=img_w, img_h=img_h)
            if yolo:
                lines.append(yolo)

        label_rel = _cvat_xml_name_to_label_rel(cvat_image_name, use_nested=use_nested)
        label_path = labels_out_dir / label_rel.with_suffix(".txt")
        if not _label_path_contained_in_dir(label_path, labels_out_dir):
            label_path = labels_out_dir / Path(Path(cvat_image_name).name).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        labels_written += 1

    return images_dir, images_found, labels_written


def _safe_int_from_xml_attr(v: str) -> int:
    try:
        return int(float(v))
    except Exception:
        return -1


def load_cvat11_images_and_labels(xml_path: Path) -> Tuple[str, List[str], List[Dict]]:
    """
    CVAT 1.1 images task (Images + bbox).

    Returns:
      task_name: str
      labels_in_meta: list[str] (may be empty)
      images: list of dict {name,width,height,boxes:[CvatBox,...]}
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    task_name = ""
    meta_task_name = root.find("./meta/task/name")
    if meta_task_name is not None and meta_task_name.text:
        task_name = meta_task_name.text.strip()
    if not task_name:
        job_id = root.find("./meta/job/id")
        if job_id is not None and job_id.text and job_id.text.strip():
            task_name = f"job_{job_id.text.strip()}"

    labels_in_meta = collect_cvat11_meta_label_names(root)

    images: List[Dict] = []
    for img_el in root.findall("./image"):
        name = img_el.attrib.get("name", "")
        w = _safe_int_from_xml_attr(img_el.attrib.get("width", ""))
        h = _safe_int_from_xml_attr(img_el.attrib.get("height", ""))

        boxes: List[CvatBox] = []
        polygons: List[CvatPolygon] = []
        for box_el in img_el.findall("./box"):
            label = box_el.attrib.get("label", "")
            if not label:
                continue
            try:
                xtl = float(box_el.attrib.get("xtl", "0"))
                ytl = float(box_el.attrib.get("ytl", "0"))
                xbr = float(box_el.attrib.get("xbr", "0"))
                ybr = float(box_el.attrib.get("ybr", "0"))
            except Exception:
                continue
            boxes.append(CvatBox(label=label, xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr))

        for shape_tag in ("polygon", "polyline"):
            for poly_el in img_el.findall(f"./{shape_tag}"):
                parsed = _parse_cvat_points_shape(poly_el)
                if parsed is not None:
                    polygons.append(parsed)

        images.append({"name": name, "width": w, "height": h, "boxes": boxes, "polygons": polygons})

    return task_name, labels_in_meta, images


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _default_tmp_dir(base_dir: Optional[Path] = None) -> Path:
    """
    A single place for temporary files: <base>/tmp (or ./tmp), not the system /tmp.
    """
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    p = (root / "tmp").resolve()
    _ensure_dir(p)
    return p


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _cvat_box_to_yolo_line(box: CvatBox, *, class_id: int, img_w: int, img_h: int) -> Optional[str]:
    if img_w <= 0 or img_h <= 0:
        return None
    xtl = _clamp(box.xtl, 0.0, float(img_w))
    ytl = _clamp(box.ytl, 0.0, float(img_h))
    xbr = _clamp(box.xbr, 0.0, float(img_w))
    ybr = _clamp(box.ybr, 0.0, float(img_h))
    w = xbr - xtl
    h = ybr - ytl
    if w <= 0 or h <= 0:
        return None
    cx = (xtl + xbr) / 2.0 / float(img_w)
    cy = (ytl + ybr) / 2.0 / float(img_h)
    nw = w / float(img_w)
    nh = h / float(img_h)
    cx = _clamp(cx, 0.0, 1.0)
    cy = _clamp(cy, 0.0, 1.0)
    nw = _clamp(nw, 0.0, 1.0)
    nh = _clamp(nh, 0.0, 1.0)
    return f"{class_id} {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}"


def _cvat_polygon_to_yolo_line(poly: CvatPolygon, *, class_id: int, img_w: int, img_h: int) -> Optional[str]:
    if img_w <= 0 or img_h <= 0 or len(poly.points) < 3:
        return None
    coords: list[str] = []
    for x, y in poly.points:
        nx = _clamp(float(x) / float(img_w), 0.0, 1.0)
        ny = _clamp(float(y) / float(img_h), 0.0, 1.0)
        coords.append(f"{nx:.6f}")
        coords.append(f"{ny:.6f}")
    return f"{class_id} " + " ".join(coords)


def import_cvat11_zip_to_yolo(
    *,
    cvat_zip_path: Path,
    output_dir: Path,
    task_name: Optional[str] = None,
    force: bool = False,
    tmp_base_dir: Optional[Path] = None,
) -> Dict:
    """
    Convert CVAT 1.1 ZIP (Images + bbox) to YOLO dataset folder:

      output_dir/
        data.yaml
        images/<image files>
        labels/<stem>.txt

    Returns dict metadata: {output_dir, task_name, nc, names, images_count, labels_count}.
    """
    cvat_zip_path = Path(cvat_zip_path)
    output_dir = Path(output_dir)

    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {output_dir}. Use --force to overwrite.")
        for child in output_dir.iterdir():
            if child.is_dir():
                for sub in child.rglob("*"):
                    if sub.is_file():
                        sub.unlink()
                for sub in sorted(child.rglob("*"), reverse=True):
                    if sub.is_dir():
                        sub.rmdir()
                child.rmdir()
            else:
                child.unlink()
        # keep root dir
    _ensure_dir(output_dir)

    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    _ensure_dir(images_out)
    _ensure_dir(labels_out)

    tmp_dir = _default_tmp_dir(tmp_base_dir)
    with tempfile.TemporaryDirectory(prefix="cvat11_import_", dir=str(tmp_dir)) as td:
        td_path = Path(td)
        with zipfile.ZipFile(cvat_zip_path, "r") as zf:
            zf.extractall(td_path)

        xml_path, images_dir = find_cvat_annotations_and_images_dir(td_path)
        xml_task_name, labels_in_meta, images = load_cvat11_images_and_labels(xml_path)
        effective_task_name = (task_name or xml_task_name or cvat_zip_path.stem).strip() or "cvat_task"

        # class list: prefer meta labels order; otherwise collect from boxes and sort.
        if labels_in_meta:
            names = list(dict.fromkeys(labels_in_meta))  # preserve order, unique
        else:
            seen = set()
            collected: List[str] = []
            for img in images:
                for b in img.get("boxes", []):
                    if isinstance(b, CvatBox) and b.label not in seen:
                        seen.add(b.label)
                        collected.append(b.label)
            names = sorted(collected)
        name_to_id = {n: i for i, n in enumerate(names)}

        images_count = 0
        labels_count = 0

        for img in images:
            cvat_image_name = img.get("name", "")
            if not isinstance(cvat_image_name, str) or not cvat_image_name:
                continue

            src = images_dir / cvat_image_name
            if not src.exists():
                src = images_dir / Path(cvat_image_name).name
            if not src.exists():
                continue

            dst_name = Path(cvat_image_name).name
            dst = images_out / dst_name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            images_count += 1

            img_w = int(img.get("width", -1))
            img_h = int(img.get("height", -1))
            if img_w <= 0 or img_h <= 0:
                with Image.open(dst) as im:
                    img_w, img_h = im.size
                    img_w = int(img_w)
                    img_h = int(img_h)

            lines: List[str] = []
            for b in img.get("boxes", []):
                if not isinstance(b, CvatBox):
                    continue
                class_id = name_to_id.get(b.label)
                if class_id is None:
                    continue
                yolo = _cvat_box_to_yolo_line(b, class_id=class_id, img_w=img_w, img_h=img_h)
                if yolo:
                    lines.append(yolo)
            for poly in img.get("polygons", []):
                if not isinstance(poly, CvatPolygon):
                    continue
                class_id = name_to_id.get(poly.label)
                if class_id is None:
                    continue
                yolo = _cvat_polygon_to_yolo_line(poly, class_id=class_id, img_w=img_w, img_h=img_h)
                if yolo:
                    lines.append(yolo)

            label_path = labels_out / f"{Path(dst_name).stem}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            labels_count += 1

        # data.yaml (flat dataset)
        yaml_path = output_dir / "data.yaml"
        yaml_path.write_text(
            "train: images\n"
            "val: images\n"
            "test: images\n\n"
            f"nc: {len(names)}\n"
            f"names: {names}\n",
            encoding="utf-8",
        )

    return {
        "output_dir": str(output_dir),
        "task_name": effective_task_name,
        "nc": len(names),
        "names": names,
        "images_count": images_count,
        "labels_count": labels_count,
    }


def _iter_yolo_pairs_flat(images_dir: Path, labels_dir: Path) -> Iterable[Tuple[Path, Path]]:
    if not images_dir.exists() or not labels_dir.exists():
        return
    for lbl in sorted(labels_dir.glob("*.txt")):
        stem = lbl.stem
        img = None
        for ext in YOLO_IMAGE_EXTS:
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                img = cand
                break
        if img is None:
            continue
        yield img, lbl


def _parse_yolo_label_file(label_path: Path) -> list[YoloBBox | YoloSegment]:
    return read_yolo_labels(str(label_path))


def _yolo_box_to_cvat_bbox(
    *,
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    if img_w <= 0 or img_h <= 0:
        return None
    cx = _clamp(cx, 0.0, 1.0)
    cy = _clamp(cy, 0.0, 1.0)
    w = _clamp(w, 0.0, 1.0)
    h = _clamp(h, 0.0, 1.0)
    xtl = (cx - w / 2.0) * float(img_w)
    ytl = (cy - h / 2.0) * float(img_h)
    xbr = (cx + w / 2.0) * float(img_w)
    ybr = (cy + h / 2.0) * float(img_h)
    xtl = _clamp(xtl, 0.0, float(img_w))
    ytl = _clamp(ytl, 0.0, float(img_h))
    xbr = _clamp(xbr, 0.0, float(img_w))
    ybr = _clamp(ybr, 0.0, float(img_h))
    if xbr <= xtl or ybr <= ytl:
        return None
    return xtl, ytl, xbr, ybr


def _fmt_float(v: float) -> str:
    s = f"{v:.3f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def build_cvat11_annotations_xml(
    *,
    task_name: str,
    images: Sequence[Tuple[str, int, int, List[Tuple[str, Tuple[float, float, float, float]]]]],
    labels: Sequence[str],
) -> str:
    now = datetime.now(timezone.utc).isoformat()

    label_xml_parts: List[str] = []
    for lb in sorted(set(labels)):
        safe = html.escape(lb, quote=False)
        label_xml_parts.append(
            "        <label>\n"
            f"          <name>{safe}</name>\n"
            "          <type>bbox</type>\n"
            "          <attributes>\n"
            "          </attributes>\n"
            "        </label>"
        )

    meta_task = f"""  <meta>
    <task>
      <id>0</id>
      <name>{html.escape(task_name, quote=False)}</name>
      <size>{len(images)}</size>
      <mode>annotation</mode>
      <overlap>0</overlap>
      <bugtracker></bugtracker>
      <flipped>False</flipped>
      <created>{now}</created>
      <updated>{now}</updated>
      <labels>
{chr(10).join(label_xml_parts)}
      </labels>
      <segments>
      </segments>
      <owner>
        <username>export</username>
        <email></email>
      </owner>
    </task>
    <dumped>{now}</dumped>
  </meta>"""

    image_xml_parts: List[str] = []
    for image_id, item in enumerate(images):
        if len(item) == 4:
            image_name, w, h, boxes = item
            polygons: List[Tuple[str, List[Tuple[float, float]]]] = []
        else:
            image_name, w, h, boxes, polygons = item
        box_parts: List[str] = []
        for z, (label, (xtl, ytl, xbr, ybr)) in enumerate(boxes):
            box_parts.append(
                "    <box "
                f'label="{html.escape(label, quote=True)}" '
                f'xtl="{_fmt_float(xtl)}" '
                f'ytl="{_fmt_float(ytl)}" '
                f'xbr="{_fmt_float(xbr)}" '
                f'ybr="{_fmt_float(ybr)}" '
                'occluded="0" '
                f'z_order="{z}">'
                "</box>"
            )
        poly_parts: List[str] = []
        for z, (label, pts) in enumerate(polygons):
            pts_str = ";".join(f"{_fmt_float(x)},{_fmt_float(y)}" for x, y in pts)
            poly_parts.append(
                "    <polygon "
                f'label="{html.escape(label, quote=True)}" '
                f'points="{pts_str}" '
                'occluded="0" '
                f'z_order="{z + len(boxes)}">'
                "</polygon>"
            )
        ann_xml = "\n".join(box_parts + poly_parts)
        image_xml_parts.append(
            f'  <image id="{image_id}" name="{html.escape(image_name, quote=True)}" width="{w}" height="{h}">\n{ann_xml}\n  </image>'
        )

    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<annotations>\n"
        "  <version>1.1</version>\n"
        f"{meta_task}\n"
        f"{chr(10).join(image_xml_parts)}\n"
        "</annotations>\n"
    )


def export_yolo_to_cvat11_zip(
    *,
    dataset_dir: Path,
    task_name: Optional[str],
    output_zip_path: Path,
    names: Sequence[str],
    images_dir: Optional[Path] = None,
    labels_dir: Optional[Path] = None,
    force: bool = False,
    tmp_base_dir: Optional[Path] = None,
) -> Dict:
    """
    Export flat YOLO dataset (images/ + labels/) to CVAT 1.1 zip.
    Supports YOLO bbox and polygon labels.
    """
    dataset_dir = Path(dataset_dir)
    output_zip_path = Path(output_zip_path)
    if output_zip_path.exists() and not force:
        raise FileExistsError(f"Zip already exists: {output_zip_path}. Use --force to overwrite.")

    images_dir = images_dir or (dataset_dir / "images")
    labels_dir = labels_dir or (dataset_dir / "labels")

    id_to_name = {i: str(n) for i, n in enumerate(list(names))}
    effective_task_name = (task_name or dataset_dir.name).strip() or "yolo_task"

    cvat_images: List[
        Tuple[str, int, int, List[Tuple[str, Tuple[float, float, float, float]]], List[Tuple[str, List[Tuple[float, float]]]]]
    ] = []
    all_labels: List[str] = []

    for img_path, lbl_path in _iter_yolo_pairs_flat(images_dir, labels_dir):
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        img_w = int(img_w)
        img_h = int(img_h)

        boxes: List[Tuple[str, Tuple[float, float, float, float]]] = []
        polygons: List[Tuple[str, List[Tuple[float, float]]]] = []
        for lb in _parse_yolo_label_file(lbl_path):
            label = id_to_name.get(int(lb.cls_id))
            if not label:
                continue
            if isinstance(lb, YoloBBox):
                bb = _yolo_box_to_cvat_bbox(cx=lb.cx, cy=lb.cy, w=lb.w, h=lb.h, img_w=img_w, img_h=img_h)
                if bb is None:
                    continue
                boxes.append((label, bb))
                all_labels.append(label)
            elif isinstance(lb, YoloSegment):
                pts = [(float(x) * img_w, float(y) * img_h) for x, y in lb.points]
                if len(pts) >= 3:
                    polygons.append((label, pts))
                    all_labels.append(label)

        cvat_images.append((img_path.name, img_w, img_h, boxes, polygons))

    annotations_xml = build_cvat11_annotations_xml(
        task_name=effective_task_name,
        images=cvat_images,
        labels=all_labels,
    )

    if output_zip_path.exists():
        output_zip_path.unlink()

    tmp_dir = _default_tmp_dir(tmp_base_dir)
    with tempfile.TemporaryDirectory(prefix="cvat11_export_", dir=str(tmp_dir)) as td:
        td_path = Path(td)
        task_root = td_path / effective_task_name
        img_out = task_root / "images"
        _ensure_dir(img_out)
        (task_root / "annotations.xml").write_text(annotations_xml, encoding="utf-8")

        for img_path, _lbl_path in _iter_yolo_pairs_flat(images_dir, labels_dir):
            (img_out / img_path.name).write_bytes(img_path.read_bytes())

        with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(task_root / "annotations.xml", arcname=str(Path(effective_task_name) / "annotations.xml"))
            for p in sorted(img_out.iterdir()):
                if p.is_file() and p.name.lower().endswith(".jpg"):
                    zf.write(p, arcname=str(Path(effective_task_name) / "images" / p.name))

    return {
        "dataset_dir": str(dataset_dir),
        "task_name": effective_task_name,
        "zip_path": str(output_zip_path),
        "images_count": len(cvat_images),
        "labels_count": len(all_labels),
    }

