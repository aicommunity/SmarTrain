"""
Tile inference via SAHI (large images/frames).
Dependency: pip install 'smartrain[sahi]'
"""
from __future__ import annotations

import argparse
import sys

from smartrain.cli_argparse import CliArgumentParser


def build_sahi_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="SAHI: slicing + YOLO (Ultralytics)")
    p.add_argument("--model", type=str, required=True, help="Path to .pt model")
    p.add_argument("--source", type=str, required=True, help="Image or image directory")
    p.add_argument(
        "--output",
        type=str,
        default="sahi_out",
        help="Directory for visualizations (export_visuals)",
    )
    p.add_argument("--slice-h", type=int, default=640)
    p.add_argument("--slice-w", type=int, default=640)
    p.add_argument("--overlap-h", type=float, default=0.2)
    p.add_argument("--overlap-w", type=float, default=0.2)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cuda")
    return p


def main(argv: list[str] | None = None) -> None:
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        print(
            "[ERROR] SAHI not installed: pip install 'smartrain[sahi]' or pip install sahi",
            file=sys.stderr,
        )
        sys.exit(1)

    import os
    from pathlib import Path

    args = build_sahi_arg_parser().parse_args(argv)
    src = Path(args.source).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device,
    )

    def run_one(image_path: Path) -> None:
        import cv2

        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[WARN] Failed to read {image_path}", file=sys.stderr)
            return
        result = get_sliced_prediction(
            frame,
            model,
            slice_height=args.slice_h,
            slice_width=args.slice_w,
            overlap_height_ratio=args.overlap_h,
            overlap_width_ratio=args.overlap_w,
        )
        sub = out / image_path.stem
        sub.mkdir(parents=True, exist_ok=True)
        result.export_visuals(export_dir=str(sub))
        print(f"[OK] {image_path.name} -> {sub}")

    if src.is_file():
        run_one(src)
    elif src.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for f in sorted(src.iterdir()):
            if f.suffix.lower() in exts:
                run_one(f)
    else:
        print(f"[ERROR] There is no such path: {src}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
