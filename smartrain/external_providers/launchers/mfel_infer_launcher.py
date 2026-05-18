from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from smartrain.core.inference.ultralytics_prediction_extract import extract_task_outputs_from_ultralytics_preds


def _extract_task_outputs(preds: Any, task_type: str) -> dict[str, Any]:
    return extract_task_outputs_from_ultralytics_preds(None, preds, task_type=task_type)


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
    p.add_argument("--result-json", default=None)
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
    _patch_mfel_missing_symbols()

    model = YOLO(args.model, task="detect")
    resolved_device = _resolve_mfel_predict_device(args.device)
    kwargs = {"source": args.source, "conf": float(args.conf), "imgsz": int(args.imgsz), "save": True}
    if resolved_device:
        kwargs["device"] = str(resolved_device)
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


def _resolve_mfel_predict_device(requested_device: str | None) -> str | None:
    requested = str(requested_device or "").strip().lower()
    if requested not in {"", "cpu"}:
        return requested_device
    try:
        import torch

        if torch.cuda.is_available():
            print("[INFO] MFEL launcher: overriding --device cpu to device=0 (CUDA available).")
            return "0"
    except Exception:
        pass
    return requested_device


def _patch_mfel_missing_symbols() -> None:
    try:
        import torch.nn as nn
        from ultralytics.nn.modules import block as block_mod
    except Exception:
        return
    if getattr(block_mod, "ConvModule", None) is not None:
        return

    class _PatchedConvModule(MFELConvModuleShim, nn.Module):  # type: ignore[misc]
        pass

    _PatchedConvModule.__module__ = __name__
    _PatchedConvModule.__qualname__ = "MFELPatchedConvModule"
    globals()["MFELPatchedConvModule"] = _PatchedConvModule
    setattr(block_mod, "ConvModule", _PatchedConvModule)


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

