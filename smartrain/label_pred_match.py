#!/usr/bin/env python3
"""
Split-level model check: run inference vs GT labels, count TP/FP/FN/class confusion,
optional viz for frames with mismatches. Outputs are grouped by evaluated dataset
(the dataset in --data-yaml), not only by the training run path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from smartrain.cli_argparse import CliArgumentParser
from smartrain.ultralytics_ephemeral import ephemeral_ultralytics_project
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root

# Default under workspace analytics/ (folder name matches CLI subcommand).
_ANALYTICS_SUBDIR = "label-pred-match"


@dataclass
class Box:
    cls_id: int
    conf: float | None
    x1: float
    y1: float
    x2: float
    y2: float
    source: str  # "gt" | "pred"


def _fs_safe_segment(name: str, max_len: int = 120) -> str:
    bad = '<>:"/\\|?*\n\r\t'
    out = "".join(c if c not in bad and ord(c) >= 32 else "_" for c in name)
    out = out.strip(". ")
    out = out[:max_len] if out else "eval"
    return out or "eval"


def _eval_dataset_key(data_yaml_path: str) -> str:
    """Folder name for the dataset being scored (from path .../datasets/<name>/data.yaml)."""
    p = Path(data_yaml_path).resolve()
    parts = p.parts
    lower = [x.lower() for x in parts]
    if "datasets" in lower:
        i = lower.index("datasets")
        if i + 1 < len(parts):
            return _fs_safe_segment(parts[i + 1])
    return _fs_safe_segment(p.parent.name)


def _safe_viz_filename(img_path: str) -> str:
    """Avoid collisions when different splits share the same file basename."""
    p = Path(img_path)
    key = hashlib.sha1(os.path.normcase(os.path.normpath(img_path)).encode("utf-8")).hexdigest()[:10]
    ext = p.suffix.lower() or ".jpg"
    return f"{p.stem}__{key}{ext}"


def build_label_pred_match_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description=(
            "Evaluate trained weights on a labeled split: match predictions to GT, "
            "log FP/FN/class confusion, optional visualization for mismatched frames."
        )
    )
    p.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Workspace root (else {WORKSPACE_ENV_VAR}) for default analytics path.",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=str, default=None, help="Single training run directory.")
    group.add_argument(
        "--runs-group-dir",
        type=str,
        default=None,
        help="Directory containing multiple run folders (e.g. runs/<trained_dataset>/*).",
    )
    p.add_argument(
        "--data-yaml",
        type=str,
        required=True,
        help="data.yaml of the dataset you are scoring (evaluated split).",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Which split in data.yaml to use.",
    )
    p.add_argument("--conf-thr", type=float, default=0.25, help="Confidence threshold for predictions.")
    p.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for pred<->GT matching.")
    p.add_argument("--max-images", type=int, default=0, help="Max images (0 = all).")
    p.add_argument(
        "--out-root",
        type=str,
        default=None,
        help=f"Output root (default: <workspace>/analytics/{_ANALYTICS_SUBDIR}).",
    )
    p.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualization for frames with FP/FN/confusion under viz/.",
    )
    p.add_argument("--device", type=str, default="0", help="Inference device (e.g. 0, cpu).")
    return p


def _xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    bw = w * img_w
    bh = h * img_h
    x1 = (xc * img_w) - bw / 2.0
    y1 = (yc * img_h) - bh / 2.0
    x2 = x1 + bw
    y2 = y1 + bh
    return x1, y1, x2, y2


def _box_metrics(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> dict[str, float]:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    denom = float(max(1, img_w * img_h))
    return {
        "w_px": w,
        "h_px": h,
        "area_px": area,
        "w_norm": w / max(1.0, float(img_w)),
        "h_norm": h / max(1.0, float(img_h)),
        "area_norm": area / denom,
    }


def _size_bin(area_norm: float) -> str:
    if area_norm < 0.01:
        return "small"
    if area_norm < 0.05:
        return "medium"
    return "large"


def _iou(a: Box, b: Box) -> float:
    xx1 = max(a.x1, b.x1)
    yy1 = max(a.y1, b.y1)
    xx2 = min(a.x2, b.x2)
    yy2 = min(a.y2, b.y2)
    iw = max(0.0, xx2 - xx1)
    ih = max(0.0, yy2 - yy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a.x2 - a.x1) * (a.y2 - a.y1))
    area_b = max(0.0, (b.x2 - b.x1) * (b.y2 - b.y1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _load_names(data_yaml_path: str) -> list[str]:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        try:
            return [str(v) for _k, v in sorted(names.items())]
        except Exception:
            return [str(v) for v in names.values()]
    return []


def _split_images_from_yaml(data_yaml_path: str, split_name: str) -> list[str]:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {data_yaml_path}")
    rel = data.get(split_name)
    if not rel or not isinstance(rel, str):
        raise ValueError(f"data.yaml has no split={split_name!r}")
    root = os.path.dirname(os.path.abspath(data_yaml_path))
    img_dir = os.path.abspath(os.path.join(root, rel))
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Split image directory not found: {img_dir}")
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    out = sorted(
        p
        for p in glob(os.path.join(img_dir, "**", "*"), recursive=True)
        if os.path.isfile(p) and p.lower().endswith(exts)
    )
    return out


def _label_path_for_image(img_path: str) -> str:
    d = Path(img_path)
    parts = list(d.parts)
    try:
        i = parts.index("images")
        parts[i] = "labels"
    except ValueError:
        return ""
    return str(Path(*parts).with_suffix(".txt"))


def _read_gt_boxes(img_path: str, img_w: int, img_h: int) -> list[Box]:
    label_path = _label_path_for_image(img_path)
    if not label_path or not os.path.isfile(label_path):
        return []
    out: list[Box] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            x1, y1, x2, y2 = _xywhn_to_xyxy(xc, yc, w, h, img_w, img_h)
            out.append(Box(cls_id=cls_id, conf=None, x1=x1, y1=y1, x2=x2, y2=y2, source="gt"))
    return out


def _pred_boxes(result: Any, conf_thr: float) -> list[Box]:
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None:
        return []
    try:
        xyxy = boxes_obj.xyxy.cpu().numpy()
        cls = boxes_obj.cls.cpu().numpy()
        conf = boxes_obj.conf.cpu().numpy()
    except Exception:
        return []
    out: list[Box] = []
    for i in range(len(xyxy)):
        c = float(conf[i])
        if c < conf_thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        out.append(
            Box(
                cls_id=int(cls[i]),
                conf=c,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                source="pred",
            )
        )
    return out


def _resolve_runs(args: argparse.Namespace) -> list[str]:
    if args.run_dir:
        rd = os.path.abspath(os.path.expanduser(args.run_dir))
        return [rd]
    rg = os.path.abspath(os.path.expanduser(args.runs_group_dir))
    if not os.path.isdir(rg):
        raise FileNotFoundError(f"runs-group directory not found: {rg}")
    return sorted(d for d in glob(os.path.join(rg, "*")) if os.path.isdir(d))


def _resolve_output_root(args: argparse.Namespace) -> str:
    if args.out_root:
        out = os.path.abspath(os.path.expanduser(args.out_root))
        os.makedirs(out, exist_ok=True)
        return out
    try:
        ws = resolve_workspace_root(args.workspace)
        out = os.path.join(WorkspaceLayout(ws).analytics, _ANALYTICS_SUBDIR)
    except ValueError:
        out = os.path.join(os.getcwd(), "analytics", _ANALYTICS_SUBDIR)
    os.makedirs(out, exist_ok=True)
    return out


def _draw_viz(
    image_path: str,
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    matched_gt: set[int],
    matched_pred: set[int],
    names: list[str],
    out_path: str,
) -> None:
    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    font_size = max(14, min(32, int(round(max(img.size) * 0.02))))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    def _draw_label(x: float, y: float, text: str, color: tuple[int, int, int]) -> None:
        l, t, r, b = d.textbbox((x, y), text, font=font)
        pad = 3
        rect = (l - pad, t - pad, r + pad, b + pad)
        d.rectangle(rect, fill=(0, 0, 0), outline=color, width=2)
        d.text((x, y), text, fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))

    for gi, g in enumerate(gt_boxes):
        lbl = names[g.cls_id] if 0 <= g.cls_id < len(names) else str(g.cls_id)
        color = (0, 255, 0) if gi in matched_gt else (255, 215, 0)
        d.rectangle([g.x1, g.y1, g.x2, g.y2], outline=color, width=2)
        _draw_label(g.x1 + 2, max(0.0, g.y1 - (font_size + 8)), f"GT:{lbl}", color)

    for pi, p in enumerate(pred_boxes):
        lbl = names[p.cls_id] if 0 <= p.cls_id < len(names) else str(p.cls_id)
        color = (80, 170, 255) if pi in matched_pred else (255, 80, 80)
        conf_txt = f"{p.conf:.2f}" if p.conf is not None else "-"
        d.rectangle([p.x1, p.y1, p.x2, p.y2], outline=color, width=2)
        _draw_label(p.x1 + 2, p.y1 + 2, f"P:{lbl} {conf_txt}", color)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        img.save(out_path, quality=95, subsampling=0, optimize=True)
    elif ext == ".png":
        img.save(out_path, compress_level=1, optimize=True)
    else:
        img.save(out_path)


def _append_detection_row(
    out: list[dict[str, Any]],
    image_path: str,
    img_w: int,
    img_h: int,
    kind: str,
    box: Box,
    gt_class: str | None,
    pred_class: str | None,
    iou_match: float | None,
) -> None:
    m = _box_metrics(box.x1, box.y1, box.x2, box.y2, img_w, img_h)
    out.append(
        {
            "image_path": image_path,
            "kind": kind,
            "gt_class": gt_class,
            "pred_class": pred_class,
            "conf": box.conf,
            "iou_match": iou_match,
            "x1": box.x1,
            "y1": box.y1,
            "x2": box.x2,
            "y2": box.y2,
            **m,
            "size_bin": _size_bin(m["area_norm"]),
        }
    )


def _analyze_single_run(
    run_dir: str,
    images: list[str],
    names: list[str],
    args: argparse.Namespace,
    out_root: str,
    *,
    eval_dataset_key: str,
    data_yaml: str,
) -> dict[str, int] | None:
    from ultralytics import YOLO

    run_dir = os.path.abspath(run_dir)
    model_name = os.path.basename(run_dir.rstrip(os.sep))
    train_parent_name = os.path.basename(os.path.dirname(run_dir.rstrip(os.sep)))

    best_pt = os.path.join(run_dir, "train", "weights", "best.pt")
    if not os.path.isfile(best_pt):
        print(f"[WARN] {model_name}: missing best.pt, skip ({best_pt})")
        return None

    # analytics/label-pred-match/<run_parent>/<run_folder>/<eval_dataset>/...
    run_group_seg = _fs_safe_segment(train_parent_name)
    run_folder_seg = _fs_safe_segment(model_name)
    out_dir = os.path.join(out_root, run_group_seg, run_folder_seg, eval_dataset_key)
    viz_dir = os.path.join(out_dir, "viz")
    if args.save_viz:
        os.makedirs(viz_dir, exist_ok=True)

    model = YOLO(best_pt)
    image_limit = args.max_images if args.max_images and args.max_images > 0 else len(images)
    target_images = images[:image_limit]

    errors_rows: list[dict[str, Any]] = []
    detections_rows: list[dict[str, Any]] = []
    totals = {
        "images_total": 0,
        "images_with_errors": 0,
        "gt_instances_total": 0,
        "pred_instances_total": 0,
        "tp_total": 0,
        "tn_total": 0,
        "fp_total": 0,
        "fn_total": 0,
        "confusion_total": 0,
    }

    progress = tqdm(target_images, desc=model_name, unit="img")
    with ephemeral_ultralytics_project("smartrain_lpm_") as ultra_proj:
        for img_path in progress:
            try:
                im = Image.open(img_path)
                img_w, img_h = im.size
                im.close()
            except Exception as e:
                print(f"[WARN] {model_name}: cannot open image {img_path}: {e}")
                continue

            gt_boxes = _read_gt_boxes(img_path, img_w, img_h)
            try:
                res = model.predict(
                    source=img_path,
                    conf=float(args.conf_thr),
                    verbose=False,
                    device=args.device,
                    save=False,
                    project=ultra_proj,
                    name=uuid.uuid4().hex[:12],
                    exist_ok=False,
                )
                pred_boxes = _pred_boxes(res[0], float(args.conf_thr)) if res else []
            except Exception as e:
                print(f"[WARN] {model_name}: predict failed for {img_path}: {e}")
                continue

            totals["images_total"] += 1
            totals["gt_instances_total"] += len(gt_boxes)
            totals["pred_instances_total"] += len(pred_boxes)

            used_gt: set[int] = set()
            used_pred: set[int] = set()
            tp = 0
            fp = 0
            fn = 0
            confusion = 0

            for pi, p in enumerate(pred_boxes):
                best_iou = 0.0
                best_gi: int | None = None
                for gi, g in enumerate(gt_boxes):
                    if gi in used_gt:
                        continue
                    iou = _iou(p, g)
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi
                if best_gi is None or best_iou < float(args.iou_thr):
                    fp += 1
                    lbl = names[p.cls_id] if 0 <= p.cls_id < len(names) else str(p.cls_id)
                    _append_detection_row(detections_rows, img_path, img_w, img_h, "fp", p, None, lbl, None)
                    continue

                used_pred.add(pi)
                used_gt.add(best_gi)
                g = gt_boxes[best_gi]
                gt_lbl = names[g.cls_id] if 0 <= g.cls_id < len(names) else str(g.cls_id)
                pr_lbl = names[p.cls_id] if 0 <= p.cls_id < len(names) else str(p.cls_id)
                if p.cls_id == g.cls_id:
                    tp += 1
                    _append_detection_row(
                        detections_rows, img_path, img_w, img_h, "tp", p, gt_lbl, pr_lbl, best_iou
                    )
                else:
                    confusion += 1
                    _append_detection_row(
                        detections_rows, img_path, img_w, img_h, "confusion", p, gt_lbl, pr_lbl, best_iou
                    )

            for gi, g in enumerate(gt_boxes):
                if gi in used_gt:
                    continue
                fn += 1
                gt_lbl = names[g.cls_id] if 0 <= g.cls_id < len(names) else str(g.cls_id)
                _append_detection_row(
                    detections_rows, img_path, img_w, img_h, "fn", g, gt_lbl, None, None
                )

            totals["tp_total"] += tp
            totals["fp_total"] += fp
            totals["fn_total"] += fn
            totals["confusion_total"] += confusion

            has_error = (fp + fn + confusion) > 0
            if has_error:
                totals["images_with_errors"] += 1
                if args.save_viz:
                    out_viz = os.path.join(viz_dir, _safe_viz_filename(img_path))
                    _draw_viz(img_path, gt_boxes, pred_boxes, used_gt, used_pred, names, out_viz)

            fn_sizes = [
                r["area_norm"] for r in detections_rows if r["image_path"] == img_path and r["kind"] == "fn"
            ]
            gt_sizes = [
                _box_metrics(g.x1, g.y1, g.x2, g.y2, img_w, img_h)["area_norm"] for g in gt_boxes
            ]
            errors_rows.append(
                {
                    "image_path": img_path,
                    "img_w": img_w,
                    "img_h": img_h,
                    "gt_count": len(gt_boxes),
                    "pred_count": len(pred_boxes),
                    "tp_count": tp,
                    "fp_count": fp,
                    "fn_count": fn,
                    "confusion_count": confusion,
                    "error_count": fp + fn + confusion,
                    "mean_gt_area_norm": (sum(gt_sizes) / len(gt_sizes)) if gt_sizes else None,
                    "mean_fn_area_norm": (sum(fn_sizes) / len(fn_sizes)) if fn_sizes else None,
                }
            )
            progress.set_postfix(tp=totals["tp_total"], fp=totals["fp_total"], fn=totals["fn_total"], refresh=False)

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(errors_rows).to_csv(os.path.join(out_dir, "errors.csv"), index=False, encoding="utf-8")
    pd.DataFrame(detections_rows).to_csv(
        os.path.join(out_dir, "detections.csv"), index=False, encoding="utf-8"
    )
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": run_dir,
                "run_name": model_name,
                "eval_dataset": eval_dataset_key,
                "data_yaml": data_yaml,
                "train_runs_parent": train_parent_name,
                "settings": {
                    "split": args.split,
                    "conf_thr": args.conf_thr,
                    "iou_thr": args.iou_thr,
                    "max_images": args.max_images,
                    "device": args.device,
                    "save_viz": bool(args.save_viz),
                },
                "totals": totals,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(
        "[INFO] "
        f"{model_name}: instances(gt/pred)={totals['gt_instances_total']}/{totals['pred_instances_total']}, "
        f"TP={totals['tp_total']}, TN={totals['tn_total']}, FP={totals['fp_total']}, FN={totals['fn_total']}"
    )
    if totals["tn_total"] == 0:
        print("[INFO] TN is not defined for object detection; left as 0.")
    print(f"[OK] {model_name}: written to {out_dir}")
    return totals


def main(argv: list[str] | None = None) -> None:
    args = build_label_pred_match_arg_parser().parse_args(argv)
    data_yaml = os.path.abspath(os.path.expanduser(args.data_yaml))
    if not os.path.isfile(data_yaml):
        print(f"[ERROR] data.yaml not found: {data_yaml}", file=sys.stderr)
        sys.exit(1)

    try:
        runs = _resolve_runs(args)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    if not runs:
        print("[ERROR] No run directories to process.", file=sys.stderr)
        sys.exit(1)

    try:
        images = _split_images_from_yaml(data_yaml, args.split)
    except Exception as e:
        print(f"[ERROR] Failed to list split images: {e}", file=sys.stderr)
        sys.exit(1)
    if not images:
        print("[ERROR] No images in split.", file=sys.stderr)
        sys.exit(1)

    names = _load_names(data_yaml)
    eval_key = _eval_dataset_key(data_yaml)
    out_root = _resolve_output_root(args)
    print(f"[INFO] output root ({_ANALYTICS_SUBDIR}): {out_root}")
    print(f"[INFO] evaluated dataset (from --data-yaml): {eval_key}")

    grand = {
        "runs_analyzed": 0,
        "gt_instances_total": 0,
        "pred_instances_total": 0,
        "tp_total": 0,
        "tn_total": 0,
        "fp_total": 0,
        "fn_total": 0,
    }
    for run_dir in runs:
        totals = _analyze_single_run(
            run_dir,
            images,
            names,
            args,
            out_root,
            eval_dataset_key=eval_key,
            data_yaml=data_yaml,
        )
        if totals is None:
            continue
        grand["runs_analyzed"] += 1
        grand["gt_instances_total"] += int(totals.get("gt_instances_total", 0))
        grand["pred_instances_total"] += int(totals.get("pred_instances_total", 0))
        grand["tp_total"] += int(totals.get("tp_total", 0))
        grand["tn_total"] += int(totals.get("tn_total", 0))
        grand["fp_total"] += int(totals.get("fp_total", 0))
        grand["fn_total"] += int(totals.get("fn_total", 0))

    print(
        "[INFO] Final totals: "
        f"runs={grand['runs_analyzed']}, "
        f"instances(gt/pred)={grand['gt_instances_total']}/{grand['pred_instances_total']}, "
        f"TP={grand['tp_total']}, TN={grand['tn_total']}, FP={grand['fp_total']}, FN={grand['fn_total']}"
    )


if __name__ == "__main__":
    main()
