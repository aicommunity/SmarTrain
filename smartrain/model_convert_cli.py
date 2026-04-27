from __future__ import annotations

import argparse
import ctypes
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smartrain.cli_argparse import CliArgumentParser
from smartrain.cli_prompts import print_numbered_options, prompt_choice, prompt_int, prompt_text, prompt_yes_no
from smartrain.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.interactive_contract import is_interactive_allowed
from smartrain.model_context import infer_img_size_with_source
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, resolve_workspace_root


@dataclass
class ConvertStats:
    total: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    artifacts_ok: int = 0
    artifacts_failed: int = 0
    artifacts_skipped: int = 0


_TRTEXEC_RUNTIME_CACHE: tuple[bool, str] | None = None


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
        choices=["onnx", "tensorrt-engine", "tensorrt-trt"],
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
        help="TensorRT precision profile (used only for TensorRT export)",
    )
    p.add_argument(
        "--imgsz",
        type=str,
        default=None,
        help="Image size: single value (640) or H,W",
    )
    p.add_argument("--opset", type=int, default=17, help="ONNX opset")
    p.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simplify ONNX graph",
    )
    p.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ONNX export dtype: FP16 when enabled, FP32 by default",
    )
    p.add_argument(
        "--nms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include NMS/postprocess into exported ONNX graph",
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


def _format_imgsz(value: int | tuple[int, int]) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value[0]},{value[1]}"


def _resolve_imgsz_from_args_and_model(args: argparse.Namespace, model_path: Path) -> tuple[int | tuple[int, int], str]:
    if args.imgsz is not None and str(args.imgsz).strip():
        return _parse_imgsz(str(args.imgsz)), "cli"
    inferred, source = infer_img_size_with_source(model_path)
    if inferred is not None:
        return int(inferred), source
    return 640, "fallback_640"


def _discover_models(workspace_root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    models_dir = workspace_root / "models"
    runs_dir = workspace_root / "runs"
    formats_order = (".pt",)

    def _append_group(root: Path, source: str) -> None:
        if not root.exists():
            return
        files = [p for p in sorted(root.rglob("*")) if p.is_file()]
        for suffix in formats_order:
            for p in files:
                if p.suffix.lower() == suffix:
                    found.append((source, p))

    # Preserve outer grouping: models first, then runs.
    _append_group(models_dir, "models")
    _append_group(runs_dir, "runs")
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

    format_options = ["onnx", "tensorrt-engine", "tensorrt-trt"]
    format_availability = _get_export_format_availability()
    displayed_options: list[str] = []
    display_to_value: dict[str, str] = {}
    selectable_labels: set[str] = set()
    use_dim = bool(sys.stdout.isatty())
    for raw in format_options:
        available, reason = format_availability.get(raw, (True, ""))
        if available:
            label = raw
            selectable_labels.add(label)
        else:
            plain = f"{raw} (unavailable: {reason})"
            label = f"\033[90m{plain}\033[0m" if use_dim else plain
        displayed_options.append(label)
        display_to_value[label] = raw
    if not selectable_labels:
        raise SystemExit("No export formats are available in current environment.")
    default_label = next((opt for opt in displayed_options if opt in selectable_labels), displayed_options[0])
    while True:
        selected_label = prompt_choice("Export format", displayed_options, default=default_label)
        selected_format = display_to_value[selected_label]
        selected_available, selected_reason = format_availability.get(selected_format, (True, ""))
        if selected_available:
            args.format = selected_format
            break
        print(f"[WARN] Format '{selected_format}' is unavailable: {selected_reason}")
    batch_mode = prompt_choice("Batch mode", ["static", "dynamic"], default="static")
    args.dynamic = batch_mode == "dynamic"
    args.batch = prompt_int("Batch size", default=1)
    if args.format in {"tensorrt-engine", "tensorrt-trt"}:
        args.precision = prompt_choice(
            "TensorRT precision (--precision)",
            ["fp32", "fp16", "int8"],
            default="fp32",
        )
    else:
        args.precision = "fp32"
    if args.format == "onnx":
        args.opset = prompt_int("ONNX opset", default=int(getattr(args, "opset", 17) or 17))
        args.simplify = prompt_yes_no("Simplify ONNX graph (--simplify)", default=bool(getattr(args, "simplify", True)))
        args.half = prompt_yes_no("Use FP16 for ONNX (--half)", default=bool(getattr(args, "half", False)))
        print("[INFO] Note: for end2end models Ultralytics may force --nms to False during export.")
        args.nms = prompt_yes_no("Include NMS in ONNX graph (--nms)", default=False)
    imgsz_mode = prompt_choice(
        "Image size mode",
        ["auto", "manual", "force-640"],
        default="auto",
    )
    args._imgsz_mode = imgsz_mode
    args.imgsz = None
    if imgsz_mode == "manual":
        default_manual = "640"
        try:
            probe_input = Path(str(args.input)).expanduser()
            if not probe_input.is_absolute():
                probe_input = (workspace_root / probe_input).resolve()
            probe_models = _collect_input_models(probe_input)
            if probe_models:
                auto_imgsz, _ = _resolve_imgsz_from_args_and_model(args, probe_models[0])
                default_manual = _format_imgsz(auto_imgsz)
        except Exception:
            pass
        args.imgsz = prompt_text("Image size (N or H,W)", default=default_manual).strip() or default_manual
    elif imgsz_mode == "force-640":
        args.imgsz = "640"
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
    if args.imgsz is not None and str(args.imgsz).strip():
        try:
            _ = _parse_imgsz(args.imgsz)
        except Exception as e:
            parser.error(f"invalid --imgsz: {e}")
    trt_requested = args.format in {"tensorrt-engine", "tensorrt-trt"}
    if trt_requested and args.precision in {"fp16", "int8"} and str(args.device).strip().lower() == "cpu":
        parser.error(f"--precision {args.precision} is incompatible with --device cpu")
    if args.format == "onnx" and bool(getattr(args, "half", False)) and str(args.device).strip().lower() == "cpu":
        parser.error("--half is incompatible with --device cpu for ONNX export. Use --no-half or GPU device.")
    if args.opset is not None and int(args.opset) <= 0:
        parser.error("--opset must be > 0")
    format_availability = _get_export_format_availability()
    available, reason = format_availability.get(str(args.format), (True, ""))
    if not available:
        parser.error(f"--format {args.format} is unavailable in current environment: {reason}")


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


def _check_trtexec_dependencies() -> tuple[bool, str]:
    trtexec_bin = _resolve_trtexec_bin()
    if not trtexec_bin:
        return False, "trtexec binary is not found"
    system_name = platform.system().lower()
    if system_name == "linux":
        try:
            proc = subprocess.run(["ldd", trtexec_bin], check=False, text=True, capture_output=True)
            output = f"{proc.stdout}\n{proc.stderr}"
            missing = [line.strip() for line in output.splitlines() if "not found" in line]
            if missing:
                return False, f"missing shared libs: {'; '.join(missing[:3])}"
        except Exception as e:
            return False, f"ldd check failed: {e}"
        # TensorRT often loads cuBLAS Lt dynamically at runtime.
        # If version 12 is unavailable, trtexec frequently fails before build.
        try:
            ctypes.CDLL("libcublasLt.so.12")
        except OSError:
            return False, "libcublasLt.so.12 is not found"
    elif system_name == "windows":
        try:
            ctypes.WinDLL("cublasLt64_12.dll")
        except Exception:
            return False, "cublasLt64_12.dll is not found"
    return True, ""


def _check_trtexec_gpu_ready() -> tuple[bool, str]:
    trtexec_bin = _resolve_trtexec_bin()
    if not trtexec_bin:
        return False, "trtexec binary is not found"
    # Fast environment preflight to avoid spending time on ONNX work when GPU
    # is not reachable from the current process namespace.
    if platform.system().lower() == "windows":
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                return False, "PyTorch CUDA check failed (cuda is unavailable)"
        except Exception as e:
            return False, f"PyTorch CUDA check failed: {e}"
        return True, ""
    try:
        probe = subprocess.run(["nvidia-smi", "-L"], check=False, text=True, capture_output=True)
    except Exception as e:
        return False, f"nvidia-smi probe failed: {e}"
    if probe.returncode != 0:
        details = (probe.stderr or probe.stdout or "").strip()
        return False, f"nvidia-smi probe failed: {details or f'exit={probe.returncode}'}"
    out = (probe.stdout or "").strip()
    if "GPU " not in out:
        return False, "nvidia-smi did not report any GPUs"
    return True, ""


def _check_trtexec_runtime_ready() -> tuple[bool, str]:
    global _TRTEXEC_RUNTIME_CACHE
    if _TRTEXEC_RUNTIME_CACHE is not None:
        return _TRTEXEC_RUNTIME_CACHE
    trtexec_bin = _resolve_trtexec_bin()
    if not trtexec_bin:
        _TRTEXEC_RUNTIME_CACHE = (False, "trtexec binary is not found")
        return _TRTEXEC_RUNTIME_CACHE
    try:
        import onnx  # type: ignore
        from onnx import TensorProto, helper  # type: ignore
    except Exception as e:
        _TRTEXEC_RUNTIME_CACHE = (False, f"onnx probe dependencies are unavailable: {e}")
        return _TRTEXEC_RUNTIME_CACHE
    probe_dir = Path(tempfile.mkdtemp(prefix="smartrain_trt_probe_"))
    onnx_path = probe_dir / "probe.onnx"
    engine_path = probe_dir / "probe.trt"
    try:
        x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])
        y = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node("Identity", inputs=["images"], outputs=["out"])
        graph = helper.make_graph([node], "smartrain_probe", [x], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
        onnx.save(model, str(onnx_path))
        cmd = [
            trtexec_bin,
            f"--onnx={onnx_path}",
            "--buildOnly",
            f"--saveEngine={engine_path}",
            "--workspace=64",
        ]
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            short = err.splitlines()[-1] if err else f"exit={proc.returncode}"
            _TRTEXEC_RUNTIME_CACHE = (False, f"trtexec runtime probe failed: {short}")
            return _TRTEXEC_RUNTIME_CACHE
        if not engine_path.exists():
            _TRTEXEC_RUNTIME_CACHE = (False, "trtexec runtime probe failed: engine is not produced")
            return _TRTEXEC_RUNTIME_CACHE
        _TRTEXEC_RUNTIME_CACHE = (True, "")
        return _TRTEXEC_RUNTIME_CACHE
    except Exception as e:
        _TRTEXEC_RUNTIME_CACHE = (False, f"trtexec runtime probe failed: {e}")
        return _TRTEXEC_RUNTIME_CACHE
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)


def _get_export_format_availability() -> dict[str, tuple[bool, str]]:
    result: dict[str, tuple[bool, str]] = {
        "onnx": (True, ""),
        "tensorrt-engine": (True, ""),
        "tensorrt-trt": (True, ""),
    }
    trt_ready, trt_reason = _check_tensorrt_ready()
    if not trt_ready:
        result["tensorrt-engine"] = (False, trt_reason)
        result["tensorrt-trt"] = (False, trt_reason)
        return result
    dep_ready, dep_reason = _check_trtexec_dependencies()
    if not dep_ready:
        result["tensorrt-trt"] = (False, dep_reason)
        return result
    gpu_ready, gpu_reason = _check_trtexec_gpu_ready()
    if not gpu_ready:
        result["tensorrt-trt"] = (False, gpu_reason)
        return result
    runtime_ready, runtime_reason = _check_trtexec_runtime_ready()
    if not runtime_ready:
        result["tensorrt-trt"] = (False, runtime_reason)
    return result


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


def _validate_onnx_export(onnx_path: Path) -> tuple[bool, str]:
    try:
        import onnx  # type: ignore

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
    except Exception as e:
        return False, f"onnx checker failed: {e}"
    return True, "ok"


def _imgsz_to_hw(imgsz: int | tuple[int, int]) -> tuple[int, int]:
    return (imgsz, imgsz) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))


def _extract_onnx_signature(onnx_path: Path) -> dict[str, Any] | None:
    try:
        import onnx  # type: ignore
    except Exception:
        return None
    try:
        model = onnx.load(str(onnx_path))
    except Exception:
        return None
    sig: dict[str, Any] = {
        "opset": None,
        "batch": None,
        "h": None,
        "w": None,
        "dynamic": None,
        "half": None,
        "simplify": None,
        "nms": None,
    }
    try:
        if model.opset_import:
            sig["opset"] = int(model.opset_import[0].version)
    except Exception:
        pass
    try:
        if model.graph.input:
            dims = model.graph.input[0].type.tensor_type.shape.dim
            if len(dims) >= 4:
                d0 = dims[0]
                d2 = dims[2]
                d3 = dims[3]
                dyn = bool(getattr(d0, "dim_param", "")) or bool(getattr(d2, "dim_param", "")) or bool(getattr(d3, "dim_param", ""))
                sig["dynamic"] = dyn
                if getattr(d0, "dim_value", 0):
                    sig["batch"] = int(d0.dim_value)
                if getattr(d2, "dim_value", 0):
                    sig["h"] = int(d2.dim_value)
                if getattr(d3, "dim_value", 0):
                    sig["w"] = int(d3.dim_value)
    except Exception:
        pass
    try:
        meta = {str(p.key): str(p.value) for p in model.metadata_props}
    except Exception:
        meta = {}
    for key in ("half", "simplify", "nms"):
        if key in meta:
            val = meta[key].strip().lower()
            if val in {"1", "true", "yes", "on"}:
                sig[key] = True
            elif val in {"0", "false", "no", "off"}:
                sig[key] = False
    return sig


def _expected_onnx_signature(args: argparse.Namespace, imgsz: int | tuple[int, int]) -> dict[str, Any]:
    h, w = _imgsz_to_hw(imgsz)
    return {
        "opset": int(args.opset),
        "batch": int(args.batch),
        "h": int(h),
        "w": int(w),
        "dynamic": bool(args.dynamic),
        "half": bool(getattr(args, "half", False)),
        "simplify": bool(getattr(args, "simplify", True)),
        "nms": bool(getattr(args, "nms", False)),
    }


def _compare_onnx_signature(existing: dict[str, Any] | None, expected: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    if not existing:
        return False, ["onnx_signature_unavailable"], []
    mismatches: list[str] = []
    warnings: list[str] = []
    for key in ("opset", "batch", "h", "w", "dynamic"):
        ev = expected.get(key)
        xv = existing.get(key)
        if xv is None:
            mismatches.append(f"{key}=unknown(expected:{ev})")
            continue
        if xv != ev:
            mismatches.append(f"{key}={xv}(expected:{ev})")
    # Optional fields are only strict when explicit metadata exists.
    # Missing metadata should not force a dedicated ONNX rebuild.
    for key in ("half", "simplify", "nms"):
        ev = expected.get(key)
        xv = existing.get(key)
        if xv is None:
            warnings.append(f"{key}=unknown(expected:{ev})")
            continue
        if xv != ev:
            mismatches.append(f"{key}={xv}(expected:{ev})")
    return len(mismatches) == 0, mismatches, warnings


def _make_dedicated_onnx_name(stem: str, expected: dict[str, Any]) -> str:
    dyn = "dynamic" if bool(expected.get("dynamic")) else "static"
    fp = "fp16" if bool(expected.get("half")) else "fp32"
    simplify = "simplify1" if bool(expected.get("simplify")) else "simplify0"
    nms = "nms1" if bool(expected.get("nms")) else "nms0"
    return (
        f"{stem}_imgsz{expected.get('h')}x{expected.get('w')}"
        f"_b{expected.get('batch')}_{dyn}_op{expected.get('opset')}_{fp}_{simplify}_{nms}_trtprep"
    )


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_v{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path


def _export_named_onnx_from_pt(
    model: Any,
    *,
    source_path: Path,
    target_path: Path,
    base_common: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    onnx_kw = {
        **base_common,
        "format": "onnx",
        "simplify": bool(args.simplify),
        "opset": int(args.opset),
        "half": bool(getattr(args, "half", False)),
        "nms": bool(getattr(args, "nms", False)),
    }
    print(
        f"[INFO] ONNX export profile: opset={onnx_kw['opset']} simplify={onnx_kw['simplify']} "
        f"half={onnx_kw['half']} nms={onnx_kw['nms']}"
    )
    try:
        exported = Path(str(model.export(**onnx_kw))).expanduser().resolve()
        ok_move, reason = _maybe_move_output(exported, target_path, force=True)
        if not ok_move:
            return False, f"cannot persist dedicated ONNX: {reason}"
        valid, validation_reason = _validate_onnx_export(target_path)
        if not valid:
            return False, validation_reason
        return True, "ok"
    except Exception as e:
        return False, str(e)


def _maybe_move_output(exported: Path, target: Path, force: bool) -> tuple[bool, str]:
    # Handle same-path exports first: Ultralytics can already write directly to
    # the final target location. In this case, do not unlink on --force.
    if exported.resolve() == target.resolve():
        if target.exists():
            return True, "same-path"
        return False, "missing-after-export"
    if target.exists():
        if not force:
            return False, "exists"
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), str(target))
    return True, "moved"


def _collect_existing_output_conflicts(models: list[Path], args: argparse.Namespace) -> list[Path]:
    conflicts: list[Path] = []
    for source_path in models:
        source_ext = source_path.suffix.lower()
        out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else source_path.parent
        if args.format == "onnx" and source_ext == ".pt":
            onnx_target = out_dir / f"{source_path.stem}.onnx"
            if onnx_target.exists():
                conflicts.append(onnx_target)
        # ONNX->ONNX does not produce a new artifact, but users still expect
        # explicit confirmation when the "output" file already exists.
        if args.format == "onnx" and source_ext == ".onnx":
            conflicts.append(source_path)
        if args.format == "tensorrt-engine":
            engine_target = out_dir / f"{source_path.stem}.engine"
            if engine_target.exists():
                conflicts.append(engine_target)
        if args.format == "tensorrt-trt":
            trt_target = out_dir / f"{source_path.stem}.trt"
            if trt_target.exists():
                conflicts.append(trt_target)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in conflicts:
        rp = path.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(rp)
    return deduped


def _convert_one(pt_path: Path, args: argparse.Namespace) -> tuple[bool, bool, bool, int, int, int]:
    """
    Returns tuple: (ok_any, failed_any, skipped_any, artifacts_ok, artifacts_failed, artifacts_skipped).
    """
    source_path = pt_path
    source_ext = source_path.suffix.lower()
    if source_ext not in {".pt", ".onnx"}:
        print(f"[ERROR] Unsupported input extension for {source_path}. Expected .pt or .onnx")
        return False, True, False, 0, 1, 0

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else source_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    imgsz, imgsz_source = _resolve_imgsz_from_args_and_model(args, source_path)
    print(f"[INFO] Resolved image size: {_format_imgsz(imgsz)} (source: {imgsz_source})")
    if imgsz_source == "fallback_640":
        print("[WARN] Training image size not found. Using fallback 640. Set --imgsz to override.")
    do_onnx = args.format == "onnx" and source_ext == ".pt"
    do_trt = args.format in {"tensorrt-engine", "tensorrt-trt"}
    use_trtexec = args.format == "tensorrt-trt"
    ok_any = False
    failed_any = False
    skipped_any = False
    artifacts_ok = 0
    artifacts_failed = 0
    artifacts_skipped = 0

    if source_ext == ".onnx" and args.format == "onnx":
        print(f"[WARN] Skip ONNX export for {source_path}: input is already ONNX.")
        return False, False, True, 0, 0, 1

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
            print(f"[WARN] Skip ONNX (exists): {onnx_target}. Use --force to rebuild.")
            skipped_any = True
            artifacts_skipped += 1
        else:
            onnx_kw = {
                **base_common,
                "format": "onnx",
                "simplify": bool(args.simplify),
                "opset": int(args.opset),
                "half": bool(getattr(args, "half", False)),
                "nms": bool(getattr(args, "nms", False)),
            }
            print(
                f"[INFO] ONNX export profile: opset={onnx_kw['opset']} simplify={onnx_kw['simplify']} "
                f"half={onnx_kw['half']} nms={onnx_kw['nms']}"
            )
            try:
                exported = Path(str(model.export(**onnx_kw))).expanduser().resolve()
                ok_move, reason = _maybe_move_output(exported, onnx_target, args.force)
                if ok_move:
                    valid_onnx, validation_reason = _validate_onnx_export(onnx_target)
                    if valid_onnx:
                        print(f"[OK] ONNX: {onnx_target}")
                        print(
                            "[INFO] ONNX post-check passed. PyTorch export warnings (e.g. aten::index) are treated as non-fatal unless validation fails."
                        )
                        ok_any = True
                        artifacts_ok += 1
                    else:
                        print(f"[ERROR] ONNX validation failed for {onnx_target}: {validation_reason}")
                        failed_any = True
                        artifacts_failed += 1
                else:
                    print(f"[WARN] Skip ONNX ({reason}): {onnx_target}")
                    skipped_any = True
                    artifacts_skipped += 1
            except Exception as e:
                print(f"[ERROR] ONNX export failed for {source_path}: {e}")
                failed_any = True
                artifacts_failed += 1

    if do_trt:
        ready, reason = _check_tensorrt_ready()
        if not ready:
            print(f"[WARN] TensorRT export unavailable for {source_path}: {reason}")
            failed_any = True
        else:
            if use_trtexec:
                trt_gpu_ready, trt_gpu_reason = _check_trtexec_gpu_ready()
                if not trt_gpu_ready:
                    print(
                        f"[WARN] TensorRT trtexec preflight failed for {source_path}: {trt_gpu_reason}. "
                        "Skipping ONNX preparation and TRT build."
                    )
                    failed_any = True
                    return ok_any, failed_any, skipped_any, artifacts_ok, artifacts_failed, artifacts_skipped
            engine_suffix = ".trt" if use_trtexec else ".engine"
            engine_target = out_dir / f"{source_path.stem}{engine_suffix}"
            if engine_target.exists() and not args.force:
                print(f"[WARN] Skip TensorRT (exists): {engine_target}. Use --force to rebuild.")
                skipped_any = True
                artifacts_skipped += 1
            elif source_ext == ".onnx":
                if args.force and engine_target.exists():
                    engine_target.unlink()
                if use_trtexec:
                    ok_trt, trt_reason = _trtexec_export_from_onnx(source_path, engine_target, args, imgsz)
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
                            ok_trt, trt_reason = True, "ok"
                        else:
                            ok_trt, trt_reason = False, f"skip ({move_reason})"
                    except Exception as e:
                        ok_trt, trt_reason = False, str(e)
                if ok_trt:
                    print(f"[OK] TensorRT: {engine_target}")
                    ok_any = True
                    artifacts_ok += 1
                else:
                    print(f"[ERROR] TensorRT export failed for {source_path}: {trt_reason}")
                    failed_any = True
                    artifacts_failed += 1
            else:
                if not use_trtexec:
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
                            artifacts_ok += 1
                        else:
                            print(f"[WARN] Skip TensorRT ({move_reason}): {engine_target}")
                            skipped_any = True
                            artifacts_skipped += 1
                    except Exception as e:
                        print(f"[ERROR] TensorRT export failed for {source_path}: {e}")
                        failed_any = True
                        artifacts_failed += 1
                    return ok_any, failed_any, skipped_any, artifacts_ok, artifacts_failed, artifacts_skipped

                existing_onnx = out_dir / f"{source_path.stem}.onnx"
                expected_sig = _expected_onnx_signature(args, imgsz)
                onnx_for_trt: Path | None = None
                if existing_onnx.exists():
                    existing_sig = _extract_onnx_signature(existing_onnx)
                    sig_ok, mismatches, warnings = _compare_onnx_signature(existing_sig, expected_sig)
                    if sig_ok:
                        onnx_for_trt = existing_onnx
                        print(f"[INFO] Using existing ONNX for TRT: {existing_onnx}")
                        if warnings:
                            print(f"[WARN] Existing ONNX has incomplete metadata: {'; '.join(warnings[:8])}")
                    else:
                        details = "; ".join(mismatches[:8])
                        if len(mismatches) > 8:
                            details += f"; ... (+{len(mismatches)-8} more)"
                        print(f"[WARN] Existing ONNX mismatch: {details}")

                if onnx_for_trt is None:
                    if model is None:
                        try:
                            from ultralytics import YOLO
                        except Exception as e:
                            print(f"[ERROR] ultralytics import failed: {e}")
                            return False, True, False
                        model = YOLO(str(source_path))
                    named = _make_dedicated_onnx_name(source_path.stem, expected_sig) + ".onnx"
                    dedicated_onnx = _next_available_path(out_dir / named)
                    ok_onnx, onnx_reason = _export_named_onnx_from_pt(
                        model,
                        source_path=source_path,
                        target_path=dedicated_onnx,
                        base_common=base_common,
                        args=args,
                    )
                    if not ok_onnx:
                        print(f"[ERROR] Failed to build dedicated ONNX for TRT from {source_path}: {onnx_reason}")
                        failed_any = True
                        artifacts_failed += 1
                        return ok_any, failed_any, skipped_any, artifacts_ok, artifacts_failed, artifacts_skipped
                    print(f"[OK] Created dedicated ONNX for TRT: {dedicated_onnx}")
                    onnx_for_trt = dedicated_onnx
                    ok_any = True
                    artifacts_ok += 1

                ok_trt, trt_reason = _trtexec_export_from_onnx(onnx_for_trt, engine_target, args, imgsz)
                if ok_trt:
                    print(f"[OK] TensorRT: {engine_target}")
                    ok_any = True
                    artifacts_ok += 1
                else:
                    print(f"[ERROR] TensorRT export failed for {source_path}: {trt_reason}")
                    failed_any = True
                    artifacts_failed += 1

    return ok_any, failed_any, skipped_any, artifacts_ok, artifacts_failed, artifacts_skipped


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

    if interactive_used:
        preview_imgsz, preview_source = _resolve_imgsz_from_args_and_model(args, models[0])
        print(f"[INFO] Interactive summary: imgsz={_format_imgsz(preview_imgsz)} (source: {preview_source})")
        if args.imgsz is None:
            args.imgsz = _format_imgsz(preview_imgsz)
        if not args.force:
            conflicts = _collect_existing_output_conflicts(models, args)
            if conflicts:
                preview = ", ".join(str(p) for p in conflicts[:3])
                if len(conflicts) > 3:
                    preview += f", ... (+{len(conflicts) - 3} more)"
                print(f"[WARN] Existing output files detected: {preview}")
                args.force = prompt_yes_no("Overwrite existing output files (--force)?", default=False)

    stats = ConvertStats(total=len(models))
    print(f"[INFO] Found {len(models)} model(s) for conversion.")
    for model_path in models:
        print(f"[INFO] Convert: {model_path}")
        ok_any, failed_any, skipped_any, artifacts_ok, artifacts_failed, artifacts_skipped = _convert_one(model_path, args)
        if ok_any:
            stats.ok += 1
        if failed_any:
            stats.failed += 1
            if not args.continue_on_error:
                break
        if skipped_any and not ok_any and not failed_any:
            stats.skipped += 1
        stats.artifacts_ok += artifacts_ok
        stats.artifacts_failed += artifacts_failed
        stats.artifacts_skipped += artifacts_skipped

    print(
        f"[INFO] Done. total={stats.total} ok={stats.ok} failed={stats.failed} skipped={stats.skipped} "
        f"(artifacts: ok={stats.artifacts_ok} failed={stats.artifacts_failed} skipped={stats.artifacts_skipped})"
    )
    if stats.ok == 0 and stats.failed == 0 and stats.skipped > 0:
        print("[INFO] All conversions were skipped because output artifacts already exist. Use --force to rebuild.")
    if interactive_used:
        if args.imgsz is None and models:
            resolved_imgsz, _ = _resolve_imgsz_from_args_and_model(args, models[0])
            args.imgsz = _format_imgsz(resolved_imgsz)
        replay_cmd = build_non_interactive_command("model convert", parser, args)
        print_replay_command("model convert", replay_cmd)

    if stats.failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
