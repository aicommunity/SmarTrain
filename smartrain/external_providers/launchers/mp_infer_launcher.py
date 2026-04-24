from __future__ import annotations

import argparse


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
    model.predict(**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

