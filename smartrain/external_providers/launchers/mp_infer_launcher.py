from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--project", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--result-json", default=None)
    p.add_argument("--task", default="detection")
    args = p.parse_args(argv)

    from ultralytics import YOLO  # noqa

    model = YOLO(args.model)
    kwargs = {"source": args.source, "conf": float(args.conf), "imgsz": int(args.imgsz), "save": True}
    if args.device:
        kwargs["device"] = str(args.device)
    if args.project:
        kwargs["project"] = str(args.project)
    if args.name:
        kwargs["name"] = str(args.name)
    if args.project or args.name:
        kwargs["exist_ok"] = True
    preds = model.predict(**kwargs)
    if args.result_json:
        _write_structured_result(args.result_json, preds, task_type=str(args.task or "detection"))
    return 0


def _to_float(v):
    try:
        to_scalar = getattr(v, "item", None)
        if callable(to_scalar):
            return float(to_scalar())
        return float(v)
    except Exception:
        return None


def _extract_task_outputs(preds, task_type: str) -> dict[str, object]:
    if not preds:
        if task_type == "classification":
            return {"classification": {}}
        if task_type == "segmentation":
            return {"segments": []}
        return {"detections": []}
    r = preds[0]
    if task_type == "classification":
        probs = getattr(r, "probs", None)
        if probs is None:
            return {"classification": {}}
        top1_idx = getattr(probs, "top1", None)
        top1_conf = getattr(probs, "top1conf", None)
        top5 = getattr(probs, "top5", None)
        top5conf = getattr(probs, "top5conf", None)
        top_k: list[dict[str, object]] = []
        if isinstance(top5, (list, tuple)):
            for i, cls_idx_raw in enumerate(top5):
                try:
                    cls_idx = int(cls_idx_raw)
                except Exception:
                    continue
                conf = None
                if isinstance(top5conf, (list, tuple)) and i < len(top5conf):
                    conf = _to_float(top5conf[i])
                top_k.append({"class_index": cls_idx, "class_name": str(cls_idx), "confidence": conf})
        if top1_idx is None:
            return {"classification": {"top_k": top_k}}
        return {
            "classification": {
                "top1": {
                    "class_index": int(top1_idx),
                    "class_name": str(int(top1_idx)),
                    "confidence": _to_float(top1_conf),
                },
                "top_k": top_k,
            }
        }
    boxes_obj = getattr(r, "boxes", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return {"segments": []} if task_type == "segmentation" else {"detections": []}
    xyxy = boxes_obj.xyxy.cpu().numpy()
    cls = boxes_obj.cls.cpu().numpy()
    confs = boxes_obj.conf.cpu().numpy()
    if task_type == "segmentation":
        masks_obj = getattr(r, "masks", None)
        polygons = getattr(masks_obj, "xy", None) if masks_obj is not None else None
        rows: list[dict[str, object]] = []
        for i in range(len(xyxy)):
            cls_idx = int(cls[i])
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            polygon_xy: list[list[float]] = []
            if isinstance(polygons, (list, tuple)) and i < len(polygons):
                poly = polygons[i]
                to_list = getattr(poly, "tolist", None)
                if callable(to_list):
                    poly = to_list()
                if isinstance(poly, list):
                    for point in poly:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            try:
                                polygon_xy.append([float(point[0]), float(point[1])])
                            except Exception:
                                continue
            rows.append(
                {
                    "bbox_roi_xyxy": [x1, y1, x2, y2],
                    "class_index": cls_idx,
                    "class_name": str(cls_idx),
                    "confidence": float(confs[i]),
                    "polygon_roi_xy": polygon_xy,
                }
            )
        return {"segments": rows}
    rows: list[dict[str, object]] = []
    for i in range(len(xyxy)):
        cls_idx = int(cls[i])
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        rows.append(
            {
                "bbox_roi_xyxy": [x1, y1, x2, y2],
                "class_index": cls_idx,
                "class_name": str(cls_idx),
                "confidence": float(confs[i]),
            }
        )
    return {"detections": rows}


def _write_structured_result(path: str, preds, *, task_type: str) -> None:
    images: list[dict[str, object]] = []
    task_outputs = _extract_task_outputs(preds, task_type=str(task_type or "detection").strip().lower())
    detections = task_outputs.get("detections")
    images.append({"detections": detections if isinstance(detections, list) else [], "task_outputs": task_outputs})
    payload = {"return_code": 0, "images": images}
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

