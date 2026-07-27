from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from smartrain.external_providers.launchers.mfel_shim import MFELConvModuleShim, patch_mfel_missing_symbols
from smartrain.external_providers.task_alias import ultralytics_task_alias

# Re-export for pickle/compat with workers that resolve this module name.
__all__ = ["MFELConvModuleShim", "main"]


def _resolve_mfel_model_spec(repo: Path, model_arg: str) -> str:
    model_raw = str(model_arg or "").strip()
    cfg_root = repo / "ultralytics" / "cfg" / "MFEL-YOLO"
    if not cfg_root.is_dir():
        raise FileNotFoundError(f"MFEL config directory not found: {cfg_root}")
    if model_raw:
        model_path = Path(model_raw).expanduser()
        if model_path.is_absolute() and model_path.is_file():
            return str(model_path)
        if model_path.is_file():
            return str((repo / model_path).resolve())
        alias = model_raw.lower().replace(".yaml", "").replace(".pt", "")
        if alias in {"mfel-yolo", "mfel_yolo"}:
            candidate = cfg_root / "MFEL-YOLO.yaml"
            if candidate.is_file():
                return str(candidate)
        if alias in {"e_pan+", "e-pan+", "e_pan_plus", "e-pan-plus"}:
            candidate = cfg_root / "E_PAN+.yaml"
            if candidate.is_file():
                return str(candidate)
        candidate = cfg_root / f"{model_raw}.yaml"
        if candidate.is_file():
            return str(candidate)
    default_candidate = cfg_root / "MFEL-YOLO.yaml"
    if default_candidate.is_file():
        return str(default_candidate)
    raise FileNotFoundError(
        "MFEL model config not found. Supported aliases usually include "
        "'mfel-yolo' and 'e_pan+'."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--project", default=None)
    p.add_argument("--name", default=None)
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

    # MFEL provider supports only its own custom configs from ultralytics/cfg/MFEL-YOLO.
    try:
        model_spec = _resolve_mfel_model_spec(repo, args.model)
    except FileNotFoundError as exc:
        print(
            f"[ERROR] {exc}",
            file=sys.stderr,
        )
        return 2
    model = YOLO(model_spec, task=ultralytics_task_alias(args.task))
    kwargs = {"data": args.data, "batch": int(args.batch), "epochs": int(args.epochs), "imgsz": int(args.imgsz), "task": ultralytics_task_alias(args.task)}
    # MFEL custom ops are unstable under AMP on some GPUs/drivers (nan losses on epoch 1).
    kwargs["amp"] = False
    # Stabilize optimization for MFEL custom architecture.
    kwargs["pretrained"] = False
    kwargs["optimizer"] = "AdamW"
    kwargs["lr0"] = 0.001
    kwargs["lrf"] = 0.01
    kwargs["warmup_epochs"] = 1.0
    kwargs["weight_decay"] = 0.0001
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

