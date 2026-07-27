from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from smartrain.core.inference.ultralytics_prediction_extract import extract_task_outputs_from_ultralytics_preds


def _extract_task_outputs(preds: Any, task_type: str) -> dict[str, Any]:
    return extract_task_outputs_from_ultralytics_preds(None, preds, task_type=task_type)


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

    from smartrain.external_providers.task_alias import ultralytics_task_alias

    model = YOLO(args.model, task=ultralytics_task_alias(args.task))
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


def _write_structured_result(path: str, preds, *, task_type: str) -> None:
    images: list[dict[str, object]] = []
    task_outputs = extract_task_outputs_from_ultralytics_preds(
        None, preds, task_type=str(task_type or "detection").strip().lower()
    )
    detections = task_outputs.get("detections")
    images.append({"detections": detections if isinstance(detections, list) else [], "task_outputs": task_outputs})
    payload = {"return_code": 0, "images": images}
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

