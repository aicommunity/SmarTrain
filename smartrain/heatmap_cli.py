"""
Визуализация heatmap через ultralytics.solutions.Heatmap.
"""
from __future__ import annotations

import argparse
import sys

from smartrain.cli_argparse import CliArgumentParser


def build_heatmap_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Heatmap (Ultralytics solutions)")
    p.add_argument("--model", type=str, required=True, help="Путь к весам .pt")
    p.add_argument("--source", type=str, required=True, help="Путь к изображению")
    p.add_argument("--output", type=str, default=None, help="Куда сохранить результат (иначе показ окна)")
    p.add_argument(
        "--colormap",
        type=int,
        default=None,
        help="cv2 colormap, напр. cv2.COLORMAP_PARULA (число). По умолчанию — из Ultralytics",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    import cv2
    from ultralytics import solutions

    args = build_heatmap_arg_parser().parse_args(argv)
    im0 = cv2.imread(args.source)
    if im0 is None:
        print(f"[ERROR] Не удалось прочитать изображение: {args.source}", file=sys.stderr)
        sys.exit(1)

    kw: dict = {"model": args.model}
    if args.output is None:
        kw["show"] = True
    else:
        kw["show"] = False
    if args.colormap is not None:
        kw["colormap"] = args.colormap

    heatmap = solutions.Heatmap(**kw)
    im = heatmap.generate_heatmap(im0)

    if args.output:
        if not cv2.imwrite(args.output, im):
            print(f"[ERROR] Не удалось записать {args.output}", file=sys.stderr)
            sys.exit(1)
        print(f"[OK] {args.output}")
    # при show=True окно открывает Ultralytics


if __name__ == "__main__":
    main()
