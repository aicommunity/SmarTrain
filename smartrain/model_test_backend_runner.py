from __future__ import annotations

import argparse
import json
from pathlib import Path

from smartrain.model_test_backends import run_native_format_backend


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Isolated native backend runner for model tests.")
    p.add_argument("--root-dir", required=True, type=str)
    p.add_argument("--weights-path", required=True, type=str)
    p.add_argument("--dataset-yaml-path", required=True, type=str)
    p.add_argument("--format-name", required=True, type=str)
    p.add_argument("--result-json", required=True, type=str)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--perf", action="store_true")
    p.add_argument("--perf-warmup-images", type=int, default=5)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_native_format_backend(
        root_dir=args.root_dir,
        weights_path=args.weights_path,
        dataset_yaml_path=args.dataset_yaml_path,
        format_name=args.format_name,
        imgsz=args.imgsz,
        val_conf=args.conf,
        val_iou=args.iou,
        val_batch=args.batch,
        collect_performance=bool(args.perf),
        perf_warmup_images=int(max(0, args.perf_warmup_images)),
    )
    payload = {
        "success": bool(result.success),
        "error": result.error,
        "format": result.format,
        "backend": result.backend,
    }
    Path(args.result_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
