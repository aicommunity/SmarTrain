from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from smartrain.external_providers.launchers.mfel_shim import MFELConvModuleShim, patch_mfel_missing_symbols
from smartrain.external_providers.task_alias import ultralytics_task_alias

__all__ = ["MFELConvModuleShim", "main"]


def _write_val_results_csv(result, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    to_csv = getattr(result, "to_csv", None)
    if callable(to_csv):
        try:
            csv_path.write_text(str(to_csv()), encoding="utf-8")
            return
        except Exception:
            pass
    # Fallback: write key metrics that are typically present on Ultralytics Metric objects.
    keys = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
    vals = []
    for k in keys:
        v = None
        try:
            result_dict = getattr(result, "results_dict", None)
            if isinstance(result_dict, dict):
                v = result_dict.get(k)
        except Exception:
            v = None
        vals.append(v)
    lines = [
        ",".join(keys),
        ",".join("" if v is None else str(v) for v in vals),
    ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--project", default=None)
    p.add_argument("--name", default="test")
    p.add_argument("--task", default="detection")
    args = p.parse_args(argv)

    repo = Path(args.repo).expanduser().resolve()
    os.chdir(str(repo))
    sys.path.insert(0, str(repo))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(repo / ".ultralytics"))

    try:
        from ultralytics import YOLO  # noqa
    except ModuleNotFoundError as e:
        if str(getattr(e, "name", "")) == "DCNv4":
            print(
                "[ERROR] MFEL-YOLO dependency is missing: DCNv4. "
                "Install DCNv4 in provider venv or use another provider.",
                file=sys.stderr,
            )
            return 2
        raise
    patch_mfel_missing_symbols(host_globals=globals())

    model = YOLO(args.model, task=ultralytics_task_alias(getattr(args, "task", "detection")))
    kwargs = {
        "data": args.data,
        "split": "test",
        "imgsz": int(args.imgsz),
        "exist_ok": True,
    }
    if args.conf is not None:
        kwargs["conf"] = float(args.conf)
    if args.iou is not None:
        kwargs["iou"] = float(args.iou)
    if args.batch is not None:
        kwargs["batch"] = int(args.batch)
    if args.device:
        kwargs["device"] = str(args.device)
    if args.project:
        kwargs["project"] = str(args.project)
    if args.name:
        kwargs["name"] = str(args.name)
    result = model.val(**kwargs)
    if args.project and args.name:
        out_dir = Path(str(args.project)).expanduser().resolve() / str(args.name)
        _write_val_results_csv(result, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

