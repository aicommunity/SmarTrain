from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _write_val_results_csv(result, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    if hasattr(result, "to_csv"):
        try:
            csv_path.write_text(str(result.to_csv()), encoding="utf-8")
            return
        except Exception:
            pass
    # Fallback: write key metrics that are typically present on Ultralytics Metric objects.
    keys = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
    vals = []
    for k in keys:
        v = None
        try:
            if hasattr(result, "results_dict") and isinstance(result.results_dict, dict):
                v = result.results_dict.get(k)
        except Exception:
            v = None
        vals.append(v)
    lines = [
        ",".join(keys),
        ",".join("" if v is None else str(v) for v in vals),
    ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _patch_mfel_missing_symbols() -> None:
    try:
        import torch.nn as nn
        from ultralytics.nn.modules import block as block_mod
    except Exception:
        return
    if hasattr(block_mod, "ConvModule"):
        return

    class MFELConvModuleShim:
        def __init__(self, c1, c2, k=1, s=1, p=0, norm_cfg=None, act_cfg=None, **kwargs):
            import torch.nn as _nn

            super().__init__()
            groups = int(kwargs.get("groups", 1) or 1)
            dilation = int(kwargs.get("dilation", 1) or 1)
            bias = bool(kwargs.get("bias", False))
            self.conv = _nn.Conv2d(c1, c2, k, s, p, groups=groups, dilation=dilation, bias=bias)
            self.bn = _nn.BatchNorm2d(int(c2))
            self.act = _nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    class _PatchedConvModule(MFELConvModuleShim, nn.Module):  # type: ignore[misc]
        pass

    _PatchedConvModule.__module__ = __name__
    _PatchedConvModule.__qualname__ = "MFELPatchedConvModule"
    globals()["MFELPatchedConvModule"] = _PatchedConvModule
    setattr(block_mod, "ConvModule", _PatchedConvModule)


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
    _patch_mfel_missing_symbols()

    model = YOLO(args.model, task="detect")
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

