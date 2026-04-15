from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smartrain.cli_argparse import CliArgumentParser
from smartrain.cli_prompts import print_numbered_options, prompt_choice, prompt_int, prompt_text
from smartrain.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.interactive_contract import is_interactive_allowed
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, resolve_workspace_root


@dataclass
class ConvertStats:
    total: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0


def build_model_convert_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Convert YOLO models (.pt/.onnx) to ONNX and TensorRT (empty call starts interactive mode)"
    )
    p.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR})",
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to .pt/.onnx model or directory with model files",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target directory for exported artifacts (defaults to source model directory)",
    )
    p.add_argument(
        "--format",
        type=str,
        choices=["onnx", "tensorrt", "both"],
        default=None,
        help="Export format. Required in non-interactive mode",
    )
    p.add_argument("--batch", type=int, default=1, help="Batch size")
    p.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use dynamic input shapes (default static)",
    )
    p.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp32",
        help="Export precision profile",
    )
    p.add_argument(
        "--imgsz",
        type=str,
        default="640",
        help="Image size: single value (640) or H,W",
    )
    p.add_argument("--opset", type=int, default=None, help="ONNX opset")
    p.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simplify ONNX graph",
    )
    p.add_argument(
        "--workspace-gib",
        type=float,
        default=None,
        help="TensorRT workspace in GiB",
    )
    p.add_argument("--device", type=str, default=None, help="Export device (cpu, 0, 0,1, ...)")
    p.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset yaml for TensorRT INT8 calibration",
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Calibration dataset fraction for TensorRT INT8",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing exported files")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue batch conversion even if one model fails",
    )
    return p


def _parse_imgsz(value: str) -> int | tuple[int, int]:
    raw = str(value).strip()
    if "," not in raw:
        return int(raw)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("imgsz must be integer or H,W")
    return int(parts[0]), int(parts[1])


def _discover_models(workspace_root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    models_dir = workspace_root / "models"
    runs_dir = workspace_root / "runs"
    allowed = {".pt", ".onnx"}
    if models_dir.exists():
        for p in sorted(models_dir.rglob("*")):
            if p.is_file():
                if p.suffix.lower() in allowed:
                    found.append(("models", p))
    if runs_dir.exists():
        for p in sorted(runs_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() in allowed:
                found.append(("runs", p))
    return found


def _collect_input_models(input_path: Path) -> list[Path]:
    allowed = {".pt", ".onnx"}
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in allowed else []
    if input_path.is_dir():
        out: list[Path] = []
        for p in input_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in allowed:
                out.append(p)
        return sorted(out)
    return []


def _interactive_fill(args: argparse.Namespace, workspace_root: Path) -> None:
    discovered = _discover_models(workspace_root)
    if discovered:
        printable = []
        options = []
        for source, path in discovered:
            rel = path.relative_to(workspace_root) if path.is_relative_to(workspace_root) else path
            label = f"{source}: {rel}"
            printable.append(label)
            options.append(label)
        options.append("<manual path>")
        print_numbered_options("available models", printable)
        selected = prompt_choice("Select input model", options, default=options[0], show_options=False)
        if selected == "<manual path>":
            args.input = prompt_text("Input path (.pt/.onnx file or dir)", default="models")
        else:
            idx = options.index(selected)
            args.input = str(discovered[idx][1])
    else:
        print("[WARN] No .pt/.onnx models discovered in workspace/models or workspace/runs.")
        args.input = prompt_text("Input path (.pt/.onnx file or dir)", default="models")

    args.format = prompt_choice("Export format", ["onnx", "tensorrt", "both"], default="onnx")
    batch_mode = prompt_choice("Batch mode", ["static", "dynamic"], default="static")
    args.dynamic = batch_mode == "dynamic"
    args.batch = prompt_int("Batch size", default=1)
    args.precision = prompt_choice("Precision", ["fp32", "fp16", "int8"], default="fp32")
    args.output_dir = prompt_text("Output directory (empty = source model dir)", default="").strip() or None


def _validate_args(
    args: argparse.Namespace,
    *,
    interactive_allowed: bool,
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> None:
    interactive_mode = interactive_allowed and len(argv) == 0 and sys.stdin.isatty()
    if not interactive_mode:
        if not args.input or not args.format:
            parser.error(
                "incomplete arguments: use --input and --format (or run command without arguments for interactive mode)."
            )
    if args.batch < 1:
        parser.error("--batch must be >= 1")
    try:
        _ = _parse_imgsz(args.imgsz)
    except Exception as e:
        parser.error(f"invalid --imgsz: {e}")
    if args.precision == "int8" and args.format == "onnx":
        parser.error("INT8 precision is not supported for ONNX export in this command. Use fp32/fp16 or TensorRT.")
    if args.precision in {"fp16", "int8"} and str(args.device).strip().lower() == "cpu":
        parser.error(f"--precision {args.precision} is incompatible with --device cpu")


def _precision_kwargs(args: argparse.Namespace, target: str) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    if args.precision == "fp16":
        kw["half"] = True
    elif args.precision == "int8":
        if target == "engine":
            kw["int8"] = True
    return kw


def _check_tensorrt_ready() -> tuple[bool, str]:
    reasons: list[str] = []
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            reasons.append("CUDA GPU is not available")
    except Exception:
        reasons.append("PyTorch CUDA check failed")
    try:
        import tensorrt  # type: ignore # noqa: F401
    except Exception:
        trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
        if not Path(trtexec).exists():
            reasons.append("python package 'tensorrt' is not installed and trtexec is not found")
    if reasons:
        return False, "; ".join(reasons)
    return True, ""


def _resolve_trtexec_bin() -> str | None:
    candidate = shutil.which("trtexec")
    if candidate:
        return candidate
    fallback = "/usr/src/tensorrt/bin/trtexec"
    if Path(fallback).exists():
        return fallback
    return None


def _guess_onnx_input_name(onnx_path: Path) -> str:
    try:
        import onnx  # type: ignore

        model = onnx.load(str(onnx_path))
        if model.graph.input:
            return str(model.graph.input[0].name)
    except Exception:
        pass
    return "images"


def _trtexec_export_from_onnx(
    onnx_path: Path,
    engine_target: Path,
    args: argparse.Namespace,
    imgsz: int | tuple[int, int],
) -> tuple[bool, str]:
    trtexec_bin = _resolve_trtexec_bin()
    if not trtexec_bin:
        return False, "trtexec binary is not found"
    h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    input_name = _guess_onnx_input_name(onnx_path)
    batch = int(args.batch)
    if batch < 1:
        return False, "batch must be >= 1"

    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        "--explicitBatch",
        f"--saveEngine={engine_target}",
        "--verbose",
    ]
    if args.workspace_gib is not None:
        cmd.append(f"--workspace={max(1, int(float(args.workspace_gib) * 1024))}")
    if args.precision == "fp16":
        cmd.append("--fp16")
    elif args.precision == "int8":
        cmd.append("--int8")
    if bool(args.dynamic):
        shape = f"{input_name}:{batch}x3x{h}x{w}"
        cmd.append(f"--minShapes={input_name}:1x3x{h}x{w}")
        cmd.append(f"--optShapes={shape}")
        cmd.append(f"--maxShapes={shape}")

    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    except Exception as e:
        return False, f"failed to run trtexec: {e}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        short = err.splitlines()[-1] if err else f"exit={proc.returncode}"
        return False, f"trtexec failed: {short}"
    if not engine_target.exists():
        return False, "trtexec finished without engine artifact"
    return True, "ok"


def _maybe_move_output(exported: Path, target: Path, force: bool) -> tuple[bool, str]:
    if target.exists():
        if not force:
            return False, "exists"
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() == target.resolve():
        return True, "same-path"
    shutil.move(str(exported), str(target))
    return True, "moved"


def _convert_one(pt_path: Path, args: argparse.Namespace) -> tuple[bool, bool, bool]:
    """
    Returns tuple: (ok_any, failed_any, skipped_any).
    """
    source_path = pt_path
    source_ext = source_path.suffix.lower()
    if source_ext not in {".pt", ".onnx"}:
        print(f"[ERROR] Unsupported input extension for {source_path}. Expected .pt or .onnx")
        return False, True, False

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else source_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    imgsz = _parse_imgsz(args.imgsz)
    do_onnx = args.format in {"onnx", "both"} and source_ext == ".pt"
    do_trt = args.format in {"tensorrt", "both"}
    ok_any = False
    failed_any = False
    skipped_any = False

    if source_ext == ".onnx" and args.format == "onnx":
        print(f"[WARN] Skip ONNX export for {source_path}: input is already ONNX.")
        return False, False, True

    if source_ext == ".onnx" and args.format == "both":
        print(f"[INFO] Input is ONNX, `both` acts as TensorRT-only for: {source_path}")

    model = None
    base_common: dict[str, Any] = {
        "imgsz": imgsz,
        "batch": args.batch,
        "dynamic": bool(args.dynamic),
    }
    if args.device is not None:
        base_common["device"] = args.device

    if do_onnx:
        if model is None:
            try:
                from ultralytics import YOLO
            except Exception as e:
                print(f"[ERROR] ultralytics import failed: {e}")
                return False, True, False
            model = YOLO(str(source_path))
        onnx_target = out_dir / f"{source_path.stem}.onnx"
        if onnx_target.exists() and not args.force:
            print(f"[WARN] Skip ONNX (exists): {onnx_target}")
            skipped_any = True
        else:
            onnx_kw = {
                **base_common,
                **_precision_kwargs(args, "onnx"),
                "format": "onnx",
                "simplify": bool(args.simplify),
            }
            if args.opset is not None:
                onnx_kw["opset"] = int(args.opset)
            try:
                exported = Path(str(model.export(**onnx_kw))).expanduser().resolve()
                ok_move, reason = _maybe_move_output(exported, onnx_target, args.force)
                if ok_move:
                    print(f"[OK] ONNX: {onnx_target}")
                    ok_any = True
                else:
                    print(f"[WARN] Skip ONNX ({reason}): {onnx_target}")
                    skipped_any = True
            except Exception as e:
                print(f"[ERROR] ONNX export failed for {source_path}: {e}")
                failed_any = True

    if do_trt:
        ready, reason = _check_tensorrt_ready()
        if not ready:
            print(f"[WARN] TensorRT export unavailable for {source_path}: {reason}")
            failed_any = True
        else:
            engine_target = out_dir / f"{source_path.stem}.engine"
            if engine_target.exists() and not args.force:
                print(f"[WARN] Skip TensorRT (exists): {engine_target}")
                skipped_any = True
            elif source_ext == ".onnx":
                if args.force and engine_target.exists():
                    engine_target.unlink()
                ok_trt, trt_reason = _trtexec_export_from_onnx(source_path, engine_target, args, imgsz)
                if ok_trt:
                    print(f"[OK] TensorRT: {engine_target}")
                    ok_any = True
                else:
                    print(f"[ERROR] TensorRT export failed for {source_path}: {trt_reason}")
                    failed_any = True
            else:
                if model is None:
                    try:
                        from ultralytics import YOLO
                    except Exception as e:
                        print(f"[ERROR] ultralytics import failed: {e}")
                        return False, True, False
                    model = YOLO(str(source_path))
                engine_kw: dict[str, Any] = {
                    **base_common,
                    **_precision_kwargs(args, "engine"),
                    "format": "engine",
                    "simplify": bool(args.simplify),
                }
                if args.workspace_gib is not None:
                    engine_kw["workspace"] = float(args.workspace_gib)
                if args.precision == "int8":
                    if args.data:
                        engine_kw["data"] = args.data
                    else:
                        print("[WARN] INT8 selected without --data; Ultralytics fallback dataset will be used.")
                    engine_kw["fraction"] = float(args.fraction)
                try:
                    exported = Path(str(model.export(**engine_kw))).expanduser().resolve()
                    ok_move, move_reason = _maybe_move_output(exported, engine_target, args.force)
                    if ok_move:
                        print(f"[OK] TensorRT: {engine_target}")
                        ok_any = True
                    else:
                        print(f"[WARN] Skip TensorRT ({move_reason}): {engine_target}")
                        skipped_any = True
                except Exception as e:
                    print(f"[ERROR] TensorRT export failed for {source_path}: {e}")
                    failed_any = True

    return ok_any, failed_any, skipped_any


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    parser = build_model_convert_arg_parser()
    args = parser.parse_args(argv)
    interactive_allowed = is_interactive_allowed(argv)
    workspace_root = Path(resolve_workspace_root(args.workspace))

    interactive_used = False
    if interactive_allowed and len(argv) == 0 and sys.stdin.isatty():
        _interactive_fill(args, workspace_root)
        interactive_used = True

    _validate_args(args, interactive_allowed=interactive_allowed, parser=parser, argv=argv)

    input_path = Path(str(args.input)).expanduser().resolve()
    models = _collect_input_models(input_path)
    if not models:
        print(f"[ERROR] No .pt/.onnx models found by input path: {input_path}")
        raise SystemExit(2)

    stats = ConvertStats(total=len(models))
    print(f"[INFO] Found {len(models)} model(s) for conversion.")
    for model_path in models:
        print(f"[INFO] Convert: {model_path}")
        ok_any, failed_any, skipped_any = _convert_one(model_path, args)
        if ok_any:
            stats.ok += 1
        if failed_any:
            stats.failed += 1
            if not args.continue_on_error:
                break
        if skipped_any and not ok_any and not failed_any:
            stats.skipped += 1

    print(
        f"[INFO] Done. total={stats.total} ok={stats.ok} failed={stats.failed} skipped={stats.skipped}"
    )
    if interactive_used:
        replay_cmd = build_non_interactive_command("model convert", parser, args)
        print_replay_command("model convert", replay_cmd)

    if stats.failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
