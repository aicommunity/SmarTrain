"""
Tile inference via SAHI (large images/frames) and dataset prepare-slices.

Dependency for ``infer``: pip install 'smartrain[sahi]'
``prepare-slices`` does not require the sahi package.
"""
from __future__ import annotations

import argparse
import sys

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser


def _add_slice_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--slice-h", type=int, default=640)
    p.add_argument("--slice-w", type=int, default=640)
    p.add_argument("--overlap-h", type=float, default=0.2)
    p.add_argument("--overlap-w", type=float, default=0.2)


def build_sahi_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="SAHI: slicing-aided inference and dataset prep")
    sub = p.add_subparsers(dest="sahi_command", required=True)

    p_infer = sub.add_parser("infer", help="Sliced inference + export_visuals (requires sahi)")
    p_infer.add_argument("--model", type=str, required=True, help="Path to .pt model")
    p_infer.add_argument("--source", type=str, required=True, help="Image or image directory")
    p_infer.add_argument(
        "--output",
        type=str,
        default="sahi_out",
        help="Directory for visualizations (export_visuals)",
    )
    _add_slice_args(p_infer)
    p_infer.add_argument("--conf", type=float, default=0.25)
    p_infer.add_argument("--device", type=str, default="cuda")

    p_prep = sub.add_parser(
        "prepare-slices",
        help="Build YOLO dataset of sliding-window slices for fine-tune (no sahi pkg needed)",
    )
    p_prep.add_argument("--workspace", type=str, default=None, help="Workspace root")
    p_prep.add_argument("--dataset", type=str, required=True, help="Source dataset key")
    p_prep.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output dataset name (default: <dataset>_sahi_slices)",
    )
    _add_slice_args(p_prep)
    return p


def _run_infer(args: argparse.Namespace) -> None:
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        print(
            "[ERROR] SAHI not installed: pip install 'smartrain[sahi]' or pip install sahi",
            file=sys.stderr,
        )
        sys.exit(1)

    from pathlib import Path

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


def _run_prepare_slices(args: argparse.Namespace) -> None:
    from smartrain.core.runtime.workspace_paths import resolve_workspace_root
    from smartrain.services.datasets.dataset_sahi_slices import prepare_sahi_slices_dataset

    try:
        workspace = resolve_workspace_root(getattr(args, "workspace", None))
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        result = prepare_sahi_slices_dataset(
            workspace=workspace,
            dataset=str(args.dataset),
            output_name=getattr(args, "output_name", None),
            slice_h=int(args.slice_h),
            slice_w=int(args.slice_w),
            overlap_h=float(args.overlap_h),
            overlap_w=float(args.overlap_w),
        )
    except Exception as exc:
        print(f"[ERROR] prepare-slices failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] SAHI slices dataset: {result['dataset']} ({result['slices_written']} slices)")
    print(f"[INFO] path: {result['path']}")


def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    # Backward compatible: ``smartrain sahi --model ...`` → ``infer``.
    if argv and argv[0].startswith("-"):
        argv = ["infer", *argv]
    parser = build_sahi_arg_parser()
    args = parser.parse_args(argv)
    if args.sahi_command == "prepare-slices":
        _run_prepare_slices(args)
        return
    _run_infer(args)


if __name__ == "__main__":
    main()
