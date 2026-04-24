from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


class MFELConvModuleShim:  # picklable top-level shim for broken forks
    def __init__(self, c1, c2, k=1, s=1, p=0, norm_cfg=None, act_cfg=None, **kwargs):
        import torch.nn as nn

        super().__init__()
        groups = int(kwargs.get("groups", 1) or 1)
        dilation = int(kwargs.get("dilation", 1) or 1)
        bias = bool(kwargs.get("bias", False))
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=groups, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(int(c2))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


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


def _patch_mfel_missing_symbols() -> None:
    try:
        import torch.nn as nn
        from ultralytics.nn.modules import block as block_mod
    except Exception:
        return
    if hasattr(block_mod, "ConvModule"):
        return

    class _PatchedConvModule(MFELConvModuleShim, nn.Module):  # type: ignore[misc]
        pass

    _PatchedConvModule.__module__ = __name__
    _PatchedConvModule.__qualname__ = "MFELPatchedConvModule"
    globals()["MFELPatchedConvModule"] = _PatchedConvModule
    setattr(block_mod, "ConvModule", _PatchedConvModule)


if __name__ == "__main__":
    raise SystemExit(main())

