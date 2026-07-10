from __future__ import annotations

import argparse
from pathlib import Path

from smartrain.core.runtime.mpl_runtime import configure_matplotlib_before_ultralytics, ensure_matplotlib_training_runtime

configure_matplotlib_before_ultralytics()


from smartrain.external_providers.task_alias import ultralytics_task_alias


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--project", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--task", default="detection")
    args = p.parse_args(argv)

    ensure_matplotlib_training_runtime(non_interactive=True)
    from ultralytics import YOLO  # noqa

    repo = Path(args.repo).expanduser().resolve()
    model_spec = args.model
    if not model_spec or model_spec.lower().startswith("yolo"):
        local_weight = repo / "yolov8n.pt"
        if local_weight.is_file():
            model_spec = str(local_weight)
    model = YOLO(model_spec, task=ultralytics_task_alias(args.task))
    kwargs = {"data": args.data, "batch": int(args.batch), "epochs": int(args.epochs), "imgsz": int(args.imgsz), "task": ultralytics_task_alias(args.task)}
    if args.device:
        kwargs["device"] = str(args.device)
    if args.project:
        kwargs["project"] = str(args.project)
    if args.name:
        kwargs["name"] = str(args.name)
    model.train(**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

