from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import (
    print_numbered_options,
    prompt_choice,
    prompt_int,
    prompt_multi_choice_csv,
    prompt_text,
    prompt_yes_no,
)
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.workflows.models.model_context import infer_img_size_with_source
from smartrain.core.runtime.run_artifacts import (
    preferred_run_model_path,
    materialize_preferred_run_model,
    run_models_dir,
    run_tmp_dir,
    write_model_sidecar_metadata,
)
from smartrain.workflows.models import tensorrt_checks as trt_checks
from smartrain.core.runtime.run_discovery import find_run_directories
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, resolve_workspace_root


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
_TRTEXEC_CAPS_CACHE: tuple[str, "TrtexecCapabilities"] | None = None


SourceKind = Literal["pt", "onnx"]


@dataclass
class InteractiveContext:
    source_kind: SourceKind
    source_path: Path
    # targets are conceptual: onnx build, ultralytics engine build, trtexec trt build
    target_onnx: bool
    target_engine: bool
    target_trt: bool

    output_dir: Path | None
    force: bool
    force_onnx: bool
    force_engine: bool
    force_trt: bool

    onnx_imgsz: int | tuple[int, int]
    onnx_imgsz_source: str
    onnx_batch: int
    onnx_dynamic: bool
    device: str | None

    engine_precision: str
    engine_workspace_gib: float | None
    trt_precision: str
    trt_workspace_gib: float | None
    data: str | None
    fraction: float

    opset: int
    simplify: bool
    half: bool
    nms: bool


@dataclass
class InteractiveResult:
    stats: ConvertStats
    session_onnx: Path | None = None
    engine_path: Path | None = None
    trt_path: Path | None = None


TrtexecCapabilities = trt_checks.TrtexecCapabilities


def _print_stage_header(title: str) -> None:
    print(f"[INFO] ===== {title} =====")


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


def _discover_models(workspace_root: Path, *, allowed_suffixes: tuple[str, ...]) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    models_dir = workspace_root / "models"
    runs_dir = workspace_root / "runs"

    def _append_group(root: Path, source: str) -> None:
        if not root.exists():
            return
        files = [p for p in sorted(root.rglob("*")) if p.is_file()]
        for suffix in allowed_suffixes:
            for p in files:
                if p.suffix.lower() == suffix:
                    found.append((source, p))

    def _append_run_canonical(root: Path) -> None:
        if not root.exists():
            return
        run_dirs = [Path(p) for p in find_run_directories(str(root))]
        for run_dir in run_dirs:
            for suffix in allowed_suffixes:
                canonical = Path(preferred_run_model_path(str(run_dir), suffix))
                if canonical.is_file():
                    found.append(("runs", canonical))
                    continue
                # Transparent legacy workspace migration on first discovery.
                migrated = materialize_preferred_run_model(str(run_dir), ext=suffix, move=True, normalize_metadata=True)
                if migrated is not None and migrated.is_file():
                    found.append(("runs", migrated))

    # Preserve outer grouping: models first, then runs.
    _append_group(models_dir, "models")
    _append_run_canonical(runs_dir)
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


def _fmt_unavailable(label: str, reason: str) -> str:
    plain = f"{label} (unavailable: {reason})"
    if sys.stdout.isatty():
        return f"\033[90m{plain}\033[0m"
    return plain


def _prompt_source_kind() -> SourceKind:
    selected = prompt_choice("Source model type", ["pt", "onnx"], default="pt")
    return "pt" if selected == "pt" else "onnx"


def _prompt_source_path(workspace_root: Path, source_kind: SourceKind) -> Path:
    suffix = ".pt" if source_kind == "pt" else ".onnx"
    discovered = _discover_models(workspace_root, allowed_suffixes=(suffix,))
    if discovered:
        printable: list[str] = []
        options: list[str] = []
        for source, path in discovered:
            rel = path.relative_to(workspace_root) if path.is_relative_to(workspace_root) else path
            label = f"{source}: {rel}"
            printable.append(label)
            options.append(label)
        options.append("<manual path>")
        print_numbered_options("available models", printable)
        selected = prompt_choice("Select input model", options, default=options[0], show_options=False)
        if selected == "<manual path>":
            raw = prompt_text(f"Input path ({suffix} file or dir)", default="models").strip() or "models"
            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = (workspace_root / p).resolve()
            return p
        idx = options.index(selected)
        return discovered[idx][1]

    print(f"[WARN] No {suffix} models discovered in workspace/models or workspace/runs.")
    raw = prompt_text(f"Input path ({suffix} file or dir)", default="models").strip() or "models"
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (workspace_root / p).resolve()
    return p


def _prompt_target_models(
    source_kind: SourceKind, *, engine_available: bool, engine_reason: str, trt_available: bool, trt_reason: str
) -> tuple[bool, bool, bool]:
    if source_kind == "pt":
        # ONNX from pt is always available; engine/trt depend on local runtime.
        options = [
            "onnx",
            "engine" if engine_available else _fmt_unavailable("engine", engine_reason),
            "trt" if trt_available else _fmt_unavailable("trt", trt_reason),
        ]
        valid_values = {"onnx"}
        if engine_available:
            valid_values.add("engine")
        if trt_available:
            valid_values.add("trt")
        while True:
            picked = prompt_multi_choice_csv("Targets", options, default_values=["onnx"])
            if not picked:
                print("[WARN] Select at least one target.")
                continue
            unavailable_selected = [item for item in picked if item not in valid_values]
            if unavailable_selected:
                print(f"[WARN] Selected targets are unavailable in current environment: {', '.join(unavailable_selected)}")
                continue
            return "onnx" in picked, "engine" in picked, "trt" in picked
    else:
        # For ONNX source, only TRT is a supported target in current architecture.
        if not trt_available:
            raise SystemExit(
                "No target models are available for onnx source in current environment: "
                f"trt: {trt_reason}"
            )
        options2: list[str] = ["trt"]

        while True:
            picked = prompt_multi_choice_csv("Targets", options2, default_values=options2)
            if not picked:
                print("[WARN] Select at least one target.")
                continue
            if any(item != "trt" for item in picked):
                print("[WARN] Invalid selection; retry.")
                continue
            if "trt" in picked:
                return False, False, True
            print("[WARN] Invalid selection; retry.")


def _run_interactive_pipeline(ctx: InteractiveContext) -> InteractiveResult:
    stats = ConvertStats(total=1)
    result = InteractiveResult(stats=stats)
    source_path = ctx.source_path
    out_dir = ctx.output_dir if ctx.output_dir else _default_output_dir_for_source(source_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Found 1 model(s) for conversion.")
    _print_stage_header("Model 1/1")
    print(f"[INFO] Convert: {source_path}")

    # Resolve concrete source model file: if directory provided, pick first matching file.
    source_models = _collect_input_models(source_path)
    if source_path.is_dir():
        allowed_ext = ".pt" if ctx.source_kind == "pt" else ".onnx"
        source_models = [p for p in source_models if p.suffix.lower() == allowed_ext]
        if not source_models:
            print(f"[ERROR] No {allowed_ext} models found by input path: {source_path}")
            stats.failed += 1
            return result
        source_path = source_models[0]

    base_common: dict[str, Any] = {
        "imgsz": ctx.onnx_imgsz,
        "batch": ctx.onnx_batch,
        "dynamic": bool(ctx.onnx_dynamic),
    }
    if ctx.device is not None:
        base_common["device"] = ctx.device

    session_onnx: Path | None = None
    expected_sig = _expected_onnx_signature(
        argparse.Namespace(
            opset=ctx.opset,
            batch=ctx.onnx_batch,
            dynamic=ctx.onnx_dynamic,
            half=ctx.half,
            simplify=ctx.simplify,
            nms=ctx.nms,
        ),
        ctx.onnx_imgsz,
    )
    public_onnx_target, session_onnx_target, engine_target, trt_target = _resolve_interactive_variant_targets(
        source_path=source_path,
        out_dir=out_dir,
        onnx_expected=expected_sig,
        onnx_imgsz=ctx.onnx_imgsz,
        onnx_batch=ctx.onnx_batch,
        onnx_dynamic=ctx.onnx_dynamic,
        engine_precision=ctx.engine_precision,
        engine_workspace_gib=ctx.engine_workspace_gib,
        trt_precision=ctx.trt_precision,
        trt_workspace_gib=ctx.trt_workspace_gib,
    )
    if ctx.source_kind == "pt" and ctx.target_onnx:
        _print_stage_header("Step ONNX")
        try:
            from ultralytics import YOLO
        except Exception as e:
            print(f"[ERROR] ultralytics import failed: {e}")
            stats.failed += 1
            return result
        model = YOLO(str(source_path))
        public_onnx = public_onnx_target
        session_onnx = session_onnx_target
        if (public_onnx.exists() or session_onnx.exists()) and not ctx.force_onnx:
            print(
                f"[WARN] Skip ONNX (exists): {public_onnx.name} / {session_onnx.name}. Use --force to rebuild."
            )
            stats.skipped += 1
            stats.artifacts_skipped += 1
        else:
            _cleanup_trtprep_artifacts(out_dir, source_path.stem)
            ok_onnx, onnx_reason = _export_named_onnx_from_pt(
                model,
                source_path=source_path,
                target_path=session_onnx,
                base_common=base_common,
                args=argparse.Namespace(
                    simplify=ctx.simplify,
                    opset=ctx.opset,
                    half=ctx.half,
                    nms=ctx.nms,
                ),
            )
            if not ok_onnx:
                print(f"[ERROR] ONNX export failed for {source_path}: {onnx_reason}")
                stats.failed += 1
                return result
            print(f"[OK] ONNX: {session_onnx}")
            run_root = _guess_run_root_for_path(source_path)
            onnx_params = {
                "imgsz": _format_imgsz(ctx.onnx_imgsz),
                "batch": ctx.onnx_batch,
                "dynamic": bool(ctx.onnx_dynamic),
                "opset": ctx.opset,
                "simplify": bool(ctx.simplify),
                "half": bool(ctx.half),
                "nms": bool(ctx.nms),
            }
            write_model_sidecar_metadata(
                session_onnx,
                format_name="onnx",
                run_dir=str(run_root) if run_root else None,
                source_path=str(source_path),
                tool="ultralytics",
                params=onnx_params,
            )
            if public_onnx.resolve() != session_onnx.resolve():
                ok_sync, sync_reason = _sync_onnx_artifact(
                    session_onnx,
                    public_onnx,
                    force=True,
                    source_path_for_meta=source_path,
                    run_root=run_root,
                    params=onnx_params,
                )
                if not ok_sync:
                    print(f"[ERROR] Failed to materialize public ONNX from dedicated cache: {sync_reason}")
                    stats.failed += 1
                    return result
                print(f"[OK] Public ONNX: {public_onnx}")
            stats.ok += 1
            stats.artifacts_ok += 1
        result.session_onnx = session_onnx

    if ctx.source_kind == "pt" and (ctx.target_engine or ctx.target_trt) and session_onnx is None:
        if (not ctx.target_engine or (engine_target.exists() and not ctx.force_engine)) and (
            not ctx.target_trt or (trt_target.exists() and not ctx.force_trt)
        ):
            print("[INFO] Skip ONNX cache: all requested TensorRT outputs already exist and overwrite is disabled.")
            session_onnx = session_onnx_target
            result.session_onnx = session_onnx
        else:
            _print_stage_header("Step ONNX cache")
            try:
                from ultralytics import YOLO
            except Exception as e:
                print(f"[ERROR] ultralytics import failed: {e}")
                stats.failed += 1
                return result
            model = YOLO(str(source_path))
            public_onnx = public_onnx_target
            session_onnx = session_onnx_target
            if session_onnx.exists():
                print(f"[INFO] Reuse ONNX cache: {session_onnx}")
            else:
                _cleanup_trtprep_artifacts(out_dir, source_path.stem)
                ok_onnx, onnx_reason = _export_named_onnx_from_pt(
                    model,
                    source_path=source_path,
                    target_path=session_onnx,
                    base_common=base_common,
                    args=argparse.Namespace(
                        simplify=ctx.simplify,
                        opset=ctx.opset,
                        half=ctx.half,
                        nms=ctx.nms,
                    ),
                )
                if not ok_onnx:
                    print(f"[ERROR] ONNX cache export failed for {source_path}: {onnx_reason}")
                    stats.failed += 1
                    return result
            run_root = _guess_run_root_for_path(source_path)
            onnx_params = {
                "imgsz": _format_imgsz(ctx.onnx_imgsz),
                "batch": ctx.onnx_batch,
                "dynamic": bool(ctx.onnx_dynamic),
                "opset": ctx.opset,
                "simplify": bool(ctx.simplify),
                "half": bool(ctx.half),
                "nms": bool(ctx.nms),
            }
            write_model_sidecar_metadata(
                session_onnx,
                format_name="onnx",
                run_dir=str(run_root) if run_root else None,
                source_path=str(source_path),
                tool="ultralytics",
                params=onnx_params,
            )
            if public_onnx.resolve() != session_onnx.resolve():
                ok_sync, sync_reason = _sync_onnx_artifact(
                    session_onnx,
                    public_onnx,
                    force=True,
                    source_path_for_meta=source_path,
                    run_root=run_root,
                    params=onnx_params,
                )
                if not ok_sync:
                    print(f"[ERROR] Failed to materialize public ONNX from ONNX cache: {sync_reason}")
                    stats.failed += 1
                    return result
            print(f"[OK] Dedicated ONNX cache: {session_onnx}")
            if public_onnx.resolve() != session_onnx.resolve():
                print(f"[OK] Public ONNX: {public_onnx}")
            stats.ok += 1
            stats.artifacts_ok += 1
            result.session_onnx = session_onnx

    # engine build
    if ctx.target_engine:
        _print_stage_header("Step TensorRT engine")
        if engine_target.exists() and not ctx.force_engine:
            print(f"[WARN] Skip TensorRT engine (exists): {engine_target}. Use --force to rebuild.")
            stats.skipped += 1
            stats.artifacts_skipped += 1
        else:
            try:
                from ultralytics import YOLO
            except Exception as e:
                print(f"[ERROR] ultralytics import failed: {e}")
                stats.failed += 1
                return result
            if ctx.source_kind != "pt":
                print("[ERROR] TensorRT engine export is supported only for .pt source models.")
                stats.failed += 1
                return result
            engine_input = source_path
            model = YOLO(str(engine_input))
            engine_kw: dict[str, Any] = {
                **base_common,
                **_precision_kwargs(
                    argparse.Namespace(precision=ctx.engine_precision, data=ctx.data, fraction=ctx.fraction), "engine"
                ),
                "format": "engine",
                "simplify": bool(ctx.simplify),
            }
            if ctx.engine_workspace_gib is not None:
                engine_kw["workspace"] = float(ctx.engine_workspace_gib)
            if ctx.engine_precision == "int8":
                if ctx.data:
                    engine_kw["data"] = ctx.data
                else:
                    print("[WARN] INT8 selected without --data; Ultralytics fallback dataset will be used.")
                engine_kw["fraction"] = float(ctx.fraction)
            print(f"[INFO] [START] TensorRT engine export: {engine_input} -> {engine_target}")
            exported = Path(str(model.export(**engine_kw))).expanduser().resolve()
            ok_move, move_reason = _maybe_move_output(exported, engine_target, ctx.force_engine)
            if not ok_move:
                print(f"[ERROR] TensorRT engine export failed for {engine_input}: {move_reason}")
                stats.failed += 1
                stats.artifacts_failed += 1
                return result
            print(f"[OK] TensorRT engine: {engine_target}")
            print(f"[INFO] [DONE] TensorRT engine export: {engine_target}")
            run_root = _guess_run_root_for_path(source_path)
            write_model_sidecar_metadata(
                engine_target,
                format_name="engine",
                run_dir=str(run_root) if run_root else None,
                source_path=str(engine_input),
                tool="ultralytics",
                params={
                    "imgsz": _format_imgsz(ctx.onnx_imgsz),
                    "batch": ctx.onnx_batch,
                    "dynamic": bool(ctx.onnx_dynamic),
                    "precision": ctx.engine_precision,
                    "workspace_gib": ctx.engine_workspace_gib,
                },
            )
            stats.ok += 1
            stats.artifacts_ok += 1
            result.engine_path = engine_target

    # trt build
    if ctx.target_trt:
        _print_stage_header("Step TensorRT trt")
        if trt_target.exists() and not ctx.force_trt:
            print(f"[WARN] Skip TensorRT trt (exists): {trt_target}. Use --force to rebuild.")
            stats.skipped += 1
            stats.artifacts_skipped += 1
        else:
            trt_input = session_onnx if (ctx.source_kind == "pt") else source_path
            if trt_input is None:
                print("[ERROR] Internal error: trt requested but session ONNX is missing.")
                stats.failed += 1
                return result
            ok_trt, trt_reason = _trtexec_export_from_onnx(
                trt_input,
                trt_target,
                argparse.Namespace(
                    batch=ctx.onnx_batch,
                    dynamic=ctx.onnx_dynamic,
                    precision=ctx.trt_precision,
                    workspace_gib=ctx.trt_workspace_gib,
                ),
                ctx.onnx_imgsz,
            )
            if not ok_trt:
                print(f"[ERROR] TensorRT trt export failed for {trt_input}: {trt_reason}")
                stats.failed += 1
                stats.artifacts_failed += 1
                return result
            print(f"[OK] TensorRT trt: {trt_target}")
            run_root = _guess_run_root_for_path(source_path)
            write_model_sidecar_metadata(
                trt_target,
                format_name="trt",
                run_dir=str(run_root) if run_root else None,
                source_path=str(trt_input),
                tool="trtexec",
                params={
                    "imgsz": _format_imgsz(ctx.onnx_imgsz),
                    "batch": ctx.onnx_batch,
                    "dynamic": bool(ctx.onnx_dynamic),
                    "precision": ctx.trt_precision,
                    "workspace_gib": ctx.trt_workspace_gib,
                },
            )
            stats.ok += 1
            stats.artifacts_ok += 1
            result.trt_path = trt_target

    return result


def _interactive_fill(args: argparse.Namespace, workspace_root: Path) -> None:
    # 1) source kind
    args._source_kind = _prompt_source_kind()
    source_kind: SourceKind = args._source_kind
    # 2) source path (file or dir)
    source_path = _prompt_source_path(workspace_root, source_kind)
    args.input = str(source_path)

    # availability
    engine_ready, engine_reason = _check_tensorrt_ready()
    availability = _get_export_format_availability()
    trt_ready, trt_reason = availability.get("tensorrt-trt", (True, ""))

    # 3) targets (separate model list)
    target_onnx, target_engine, target_trt = _prompt_target_models(
        source_kind, engine_available=engine_ready, engine_reason=engine_reason, trt_available=trt_ready, trt_reason=trt_reason
    )
    args._target_onnx = bool(target_onnx)
    args._target_engine = bool(target_engine)
    args._target_trt = bool(target_trt)

    # 4) ONNX block first for pt source
    imgsz_mode = "auto"
    args.imgsz = None
    if source_kind == "pt":
        print("[INFO] ONNX settings")
        imgsz_mode = prompt_choice("ONNX image size mode", ["auto", "manual", "force-640"], default="auto")
        args._imgsz_mode = imgsz_mode
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
            args.imgsz = prompt_text("ONNX image size (N or H,W)", default=default_manual).strip() or default_manual
        elif imgsz_mode == "force-640":
            args.imgsz = "640"
        batch_mode = prompt_choice("ONNX batch mode", ["static", "dynamic"], default="static")
        args.dynamic = batch_mode == "dynamic"
        args.batch = prompt_int("ONNX batch size", default=1)
        args.opset = prompt_int("ONNX opset", default=int(getattr(args, "opset", 17) or 17))
        args.simplify = prompt_yes_no("Simplify ONNX graph (--simplify)", default=bool(getattr(args, "simplify", True)))
        args.half = prompt_yes_no("Use FP16 for ONNX (--half)", default=bool(getattr(args, "half", False)))
        print("[INFO] Note: for end2end models Ultralytics may force --nms to False during export.")
        args.nms = prompt_yes_no("Include NMS in ONNX graph (--nms)", default=False)
    else:
        # For strict ONNX input, downstream formats still need shape/profile settings.
        print("[INFO] Target shape settings")
        imgsz_mode = prompt_choice("Target image size mode", ["auto", "manual", "force-640"], default="auto")
        args._imgsz_mode = imgsz_mode
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
            args.imgsz = prompt_text("Target image size (N or H,W)", default=default_manual).strip() or default_manual
        elif imgsz_mode == "force-640":
            args.imgsz = "640"
        batch_mode = prompt_choice("Target batch mode", ["static", "dynamic"], default="static")
        args.dynamic = batch_mode == "dynamic"
        args.batch = prompt_int("Target batch size", default=1)

    # 5) Engine block
    args._engine_precision = "fp32"
    args._engine_workspace_gib = None
    if target_engine:
        print("[INFO] Engine settings")
        args._engine_precision = prompt_choice("Engine precision (--precision)", ["fp32", "fp16", "int8"], default="fp32")
        engine_ws = prompt_text("Engine workspace GiB (empty = default)", default="").strip()
        args._engine_workspace_gib = float(engine_ws) if engine_ws else None

    # 6) TRT block
    args._trt_precision = "fp32"
    args._trt_workspace_gib = None
    if target_trt:
        print("[INFO] TRT settings")
        args._trt_precision = prompt_choice("TRT precision (--precision)", ["fp32", "fp16", "int8"], default="fp32")
        trt_ws = prompt_text("TRT workspace GiB (empty = default)", default="").strip()
        args._trt_workspace_gib = float(trt_ws) if trt_ws else None

    # Preserve legacy single-value args for non-interactive/validation compatibility.
    args.precision = args._engine_precision if target_engine else args._trt_precision
    args.workspace_gib = args._engine_workspace_gib if target_engine else args._trt_workspace_gib

    args.output_dir = prompt_text("Output directory (empty = source model dir)", default="").strip() or None
    args._force_onnx = bool(args.force)
    args._force_engine = bool(args.force)
    args._force_trt = bool(args.force)
    if bool(args.force):
        return

    source_input = Path(str(args.input)).expanduser()
    if not source_input.is_absolute():
        source_input = (workspace_root / source_input).resolve()
    chosen_models = _collect_input_models(source_input)
    if source_input.is_dir():
        expected_ext = ".pt" if source_kind == "pt" else ".onnx"
        chosen_models = [p for p in chosen_models if p.suffix.lower() == expected_ext]
    if not chosen_models:
        return
    chosen_source = chosen_models[0]
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else chosen_source.parent
    resolved_imgsz = _parse_imgsz(str(args.imgsz)) if args.imgsz else _resolve_imgsz_from_args_and_model(args, chosen_source)[0]
    expected_sig = _expected_onnx_signature(args, resolved_imgsz)
    public_onnx, dedicated_onnx, engine_target_variant, trt_target_variant = _resolve_interactive_variant_targets(
        source_path=chosen_source,
        out_dir=out_dir,
        onnx_expected=expected_sig,
        onnx_imgsz=resolved_imgsz,
        onnx_batch=int(args.batch),
        onnx_dynamic=bool(args.dynamic),
        engine_precision=str(getattr(args, "_engine_precision", "fp32")),
        engine_workspace_gib=getattr(args, "_engine_workspace_gib", None),
        trt_precision=str(getattr(args, "_trt_precision", "fp32")),
        trt_workspace_gib=getattr(args, "_trt_workspace_gib", None),
    )

    if source_kind == "pt" and bool(target_onnx):
        onnx_target = dedicated_onnx
        if onnx_target.is_file() or public_onnx.is_file():
            args._force_onnx = prompt_yes_no(
                f"Overwrite existing ONNX targets ({public_onnx.name}, {onnx_target.name})",
                default=False,
            )
    if bool(target_engine):
        engine_target = engine_target_variant
        if engine_target.is_file():
            args._force_engine = prompt_yes_no(
                f"Overwrite existing TensorRT engine ({engine_target.name})",
                default=False,
            )
    if bool(target_trt):
        trt_target = trt_target_variant
        if trt_target.is_file():
            args._force_trt = prompt_yes_no(
                f"Overwrite existing TensorRT trt ({trt_target.name})",
                default=False,
            )


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
    input_value = str(getattr(args, "input", "") or "").strip().lower()
    if args.format == "tensorrt-engine" and input_value.endswith(".onnx"):
        parser.error("--format tensorrt-engine requires a .pt input model; .onnx is not supported for Ultralytics engine export")
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
    return trt_checks.check_tensorrt_ready()


def _detect_trtexec_capabilities(trtexec_bin: str) -> TrtexecCapabilities:
    return trt_checks.detect_trtexec_capabilities(trtexec_bin)


def _get_trtexec_capabilities(trtexec_bin: str) -> TrtexecCapabilities:
    global _TRTEXEC_CAPS_CACHE
    if _TRTEXEC_CAPS_CACHE is not None and _TRTEXEC_CAPS_CACHE[0] == trtexec_bin:
        return _TRTEXEC_CAPS_CACHE[1]
    caps = trt_checks.get_trtexec_capabilities(trtexec_bin)
    _TRTEXEC_CAPS_CACHE = (trtexec_bin, caps)
    return caps


def _append_trtexec_workspace_arg(cmd: list[str], workspace_mib: int, caps: TrtexecCapabilities) -> None:
    trt_checks.append_trtexec_workspace_arg(cmd, workspace_mib, caps)


def _check_trtexec_dependencies() -> tuple[bool, str]:
    return trt_checks.check_trtexec_dependencies()


def _check_trtexec_gpu_ready() -> tuple[bool, str]:
    return trt_checks.check_trtexec_gpu_ready()


def _check_trtexec_runtime_ready() -> tuple[bool, str]:
    global _TRTEXEC_RUNTIME_CACHE
    if _TRTEXEC_RUNTIME_CACHE is not None:
        return _TRTEXEC_RUNTIME_CACHE
    trt_checks._TRTEXEC_RUNTIME_CACHE = None
    _TRTEXEC_RUNTIME_CACHE = trt_checks.check_trtexec_runtime_ready()
    return _TRTEXEC_RUNTIME_CACHE


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
        hint = "try format=tensorrt-engine as a fallback path"
        result["tensorrt-trt"] = (False, f"{runtime_reason}; {hint}")
    return result


def _resolve_trtexec_bin() -> str | None:
    return trt_checks.resolve_trtexec_bin()


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
    caps = _get_trtexec_capabilities(trtexec_bin)
    h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    input_name = _guess_onnx_input_name(onnx_path)
    batch = int(args.batch)
    if batch < 1:
        return False, "batch must be >= 1"

    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_target}",
        "--verbose",
    ]
    if caps.supports_explicit_batch:
        cmd.append("--explicitBatch")
    if args.workspace_gib is not None:
        _append_trtexec_workspace_arg(cmd, max(1, int(float(args.workspace_gib) * 1024)), caps)
    if args.precision == "fp16":
        cmd.append("--fp16")
    elif args.precision == "int8":
        cmd.append("--int8")
    if bool(args.dynamic):
        shape = f"{input_name}:{batch}x3x{h}x{w}"
        cmd.append(f"--minShapes={input_name}:1x3x{h}x{w}")
        cmd.append(f"--optShapes={shape}")
        cmd.append(f"--maxShapes={shape}")

    print(f"[INFO] [START] TensorRT trt build (trtexec): {onnx_path} -> {engine_target}")
    # Ultralytics ONNX export may leave CUDA visibility in a CPU-only state
    # for the current process. Run trtexec with a sanitized environment to
    # keep direct GPU access deterministic.
    trt_env = os.environ.copy()
    cvd = trt_env.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and not str(cvd).strip():
        trt_env.pop("CUDA_VISIBLE_DEVICES", None)
    run_tmp = _guess_run_tmp_dir_for_path(engine_target)
    if run_tmp is not None:
        run_tmp.mkdir(parents=True, exist_ok=True)
        log_file = Path(tempfile.mkstemp(prefix="smartrain_trtexec_", suffix=".log", dir=str(run_tmp))[1])
    else:
        log_file = Path(tempfile.mkstemp(prefix="smartrain_trtexec_", suffix=".log")[1])
    try:
        with log_file.open("w", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, env=trt_env)
            started = time.monotonic()
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                elapsed = int(time.monotonic() - started)
                print(f"[INFO] [LIVE] TensorRT trt build is running... {elapsed}s elapsed")
                time.sleep(5)
    except Exception as e:
        return False, f"failed to run trtexec: {e}"
    if proc.returncode != 0:
        try:
            lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            tail = lines[-1] if lines else ""
        except Exception:
            tail = ""
        details = tail or f"exit={proc.returncode}"
        return False, f"trtexec failed: {details} (full log: {log_file})"
    if not engine_target.exists():
        return False, "trtexec finished without engine artifact"
    try:
        log_file.unlink(missing_ok=True)
    except Exception:
        pass
    print(f"[INFO] [DONE] TensorRT trt build (trtexec): {engine_target}")
    return True, "ok"


def _guess_run_root_for_path(path: Path) -> Path | None:
    current = path.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for cand in [current, *current.parents]:
        # Some workspaces can prune train artifacts after export; in this case
        # the run context must still be detected from stable run markers.
        if (cand / "training_metadata.json").is_file():
            return cand
        if (cand / "models").is_dir() and (cand / "test").is_dir():
            return cand
        # Canonical layout fallback: runs/<dataset>/<run>/models/*
        # Keep run-context even if metadata/test dir is temporarily absent.
        if cand.name == "models":
            run_dir = cand.parent
            if run_dir.is_dir() and run_dir.parent.is_dir() and run_dir.parent.parent.name == "runs":
                return run_dir
    return None


def _guess_run_tmp_dir_for_path(path: Path) -> Path | None:
    run_root = _guess_run_root_for_path(path)
    if run_root is None:
        return None
    return run_tmp_dir(str(run_root))


def _default_output_dir_for_source(source_path: Path) -> Path:
    run_root = _guess_run_root_for_path(source_path)
    if run_root is not None:
        return run_models_dir(str(run_root))
    return source_path.parent


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


def _make_variant_tensor_name(stem: str, ext: str, args: argparse.Namespace, imgsz: int | tuple[int, int]) -> str:
    h, w = _imgsz_to_hw(imgsz)
    dyn = "dynamic" if bool(args.dynamic) else "static"
    precision = str(getattr(args, "precision", "fp32"))
    ws = getattr(args, "workspace_gib", None)
    ws_tag = f"_ws{str(ws).replace('.', 'p')}" if ws is not None else ""
    base = f"{stem}_imgsz{h}x{w}_b{int(args.batch)}_{dyn}_{precision}{ws_tag}"
    return f"{base}{ext}"


def _variant_or_legacy_name(source_path: Path, *, ext: str, args: argparse.Namespace, imgsz: int | tuple[int, int]) -> str:
    run_root = _guess_run_root_for_path(source_path)
    if run_root is None:
        return f"{source_path.stem}{ext}"
    return _make_variant_tensor_name(source_path.stem, ext, args, imgsz)


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_v{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path


def _cleanup_trtprep_artifacts(out_dir: Path, source_stem: str) -> None:
    pattern = f"{source_stem}*trtprep*.onnx"
    for candidate in sorted(out_dir.glob(pattern)):
        if not candidate.is_file():
            continue
        try:
            candidate.unlink()
        except Exception:
            # Best-effort cleanup: stale cache files must not fail conversion.
            pass


def _resolve_public_onnx_target(source_path: Path, out_dir: Path, expected: dict[str, Any]) -> Path:
    run_root = _guess_run_root_for_path(source_path)
    in_runs_models = out_dir.name == "models" and out_dir.parent.parent.name == "runs"
    if run_root is None and not in_runs_models:
        return out_dir / f"{source_path.stem}.onnx"
    public_name = _make_dedicated_onnx_name(source_path.stem, expected).replace("_trtprep", "")
    return out_dir / f"{public_name}.onnx"


def _cleanup_legacy_plain_onnx_for_run(source_path: Path, target_onnx: Path) -> None:
    try:
        target = target_onnx.expanduser().resolve()
        out_dir = target.parent
        if out_dir.name != "models" or out_dir.parent.parent.parent.name != "runs":
            return
        plain = out_dir / f"{source_path.stem}.onnx"
        plain = plain.resolve()
        if plain == target or not plain.exists():
            return
        plain.unlink(missing_ok=True)
        plain_meta = plain.with_suffix(plain.suffix + ".meta.json")
        plain_meta.unlink(missing_ok=True)
        print(f"[INFO] Removed legacy plain ONNX artifact: {plain}")
    except Exception:
        # Best-effort cleanup: must not fail conversion.
        return


def _resolve_interactive_variant_targets(
    *,
    source_path: Path,
    out_dir: Path,
    onnx_expected: dict[str, Any],
    onnx_imgsz: int | tuple[int, int],
    onnx_batch: int,
    onnx_dynamic: bool,
    engine_precision: str,
    engine_workspace_gib: float | None,
    trt_precision: str,
    trt_workspace_gib: float | None,
) -> tuple[Path, Path, Path, Path]:
    dedicated_onnx = out_dir / (_make_dedicated_onnx_name(source_path.stem, onnx_expected) + ".onnx")
    public_onnx = _resolve_public_onnx_target(source_path, out_dir, onnx_expected)
    engine_target = out_dir / _variant_or_legacy_name(
        source_path,
        ext=".engine",
        args=argparse.Namespace(
            batch=onnx_batch,
            dynamic=onnx_dynamic,
            precision=engine_precision,
            workspace_gib=engine_workspace_gib,
        ),
        imgsz=onnx_imgsz,
    )
    trt_target = out_dir / _variant_or_legacy_name(
        source_path,
        ext=".trt",
        args=argparse.Namespace(
            batch=onnx_batch,
            dynamic=onnx_dynamic,
            precision=trt_precision,
            workspace_gib=trt_workspace_gib,
        ),
        imgsz=onnx_imgsz,
    )
    return public_onnx, dedicated_onnx, engine_target, trt_target


def _sync_onnx_artifact(
    source_onnx: Path,
    target_onnx: Path,
    *,
    force: bool,
    source_path_for_meta: Path,
    run_root: Path | None,
    params: dict[str, Any],
) -> tuple[bool, str]:
    try:
        src = source_onnx.expanduser().resolve()
        dst = target_onnx.expanduser().resolve()
        if src != dst:
            if dst.exists():
                if not force:
                    return False, "exists"
                dst.unlink()
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
        write_model_sidecar_metadata(
            target_onnx,
            format_name="onnx",
            run_dir=str(run_root) if run_root else None,
            source_path=str(source_path_for_meta),
            tool="ultralytics",
            params=params,
        )
        _cleanup_legacy_plain_onnx_for_run(source_path_for_meta, target_onnx)
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


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
            expected = _expected_onnx_signature(args, _resolve_imgsz_from_args_and_model(args, source_path)[0])
            onnx_target = _resolve_public_onnx_target(source_path, out_dir, expected)
            if onnx_target.exists():
                conflicts.append(onnx_target)
        # ONNX->ONNX does not produce a new artifact, but users still expect
        # explicit confirmation when the "output" file already exists.
        if args.format == "onnx" and source_ext == ".onnx":
            conflicts.append(source_path)
        if args.format == "tensorrt-engine":
            engine_target = out_dir / _variant_or_legacy_name(
                source_path,
                ext=".engine",
                args=args,
                imgsz=_resolve_imgsz_from_args_and_model(args, source_path)[0],
            )
            if engine_target.exists():
                conflicts.append(engine_target)
        if args.format == "tensorrt-trt":
            trt_target = out_dir / _variant_or_legacy_name(
                source_path,
                ext=".trt",
                args=args,
                imgsz=_resolve_imgsz_from_args_and_model(args, source_path)[0],
            )
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

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir_for_source(source_path)
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
        _print_stage_header("Step ONNX")
        if model is None:
            try:
                from ultralytics import YOLO
            except Exception as e:
                print(f"[ERROR] ultralytics import failed: {e}")
                return False, True, False
            model = YOLO(str(source_path))
        expected = _expected_onnx_signature(args, imgsz)
        run_root = _guess_run_root_for_path(source_path)
        public_onnx = _resolve_public_onnx_target(source_path, out_dir, expected)
        dedicated_onnx = out_dir / (_make_dedicated_onnx_name(source_path.stem, expected) + ".onnx")
        if public_onnx.exists() and not args.force:
            print(f"[WARN] Skip ONNX (exists): {public_onnx}. Use --force to rebuild.")
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
                onnx_params = {
                    "imgsz": _format_imgsz(imgsz),
                    "batch": int(args.batch),
                    "dynamic": bool(args.dynamic),
                    "opset": int(args.opset),
                    "simplify": bool(getattr(args, "simplify", True)),
                    "half": bool(getattr(args, "half", False)),
                    "nms": bool(getattr(args, "nms", False)),
                }
                target_primary = public_onnx
                ok_primary, primary_reason = _sync_onnx_artifact(
                    exported,
                    target_primary,
                    force=True,
                    source_path_for_meta=source_path,
                    run_root=run_root,
                    params=onnx_params,
                )
                if not ok_primary:
                    print(f"[ERROR] ONNX export failed for {source_path}: {primary_reason}")
                    failed_any = True
                    artifacts_failed += 1
                else:
                    print(f"[OK] ONNX: {target_primary}")
                    # Keep trtprep cache synchronized without second export.
                    if dedicated_onnx.resolve() != target_primary.resolve():
                        _cleanup_trtprep_artifacts(out_dir, source_path.stem)
                        ok_cache, cache_reason = _sync_onnx_artifact(
                            target_primary,
                            dedicated_onnx,
                            force=True,
                            source_path_for_meta=source_path,
                            run_root=run_root,
                            params=onnx_params,
                        )
                        if not ok_cache:
                            print(f"[ERROR] Failed to update dedicated ONNX cache from public ONNX: {cache_reason}")
                            failed_any = True
                            artifacts_failed += 1
                        else:
                            print(f"[OK] Dedicated ONNX cache: {dedicated_onnx}")
                    print(
                        "[INFO] ONNX post-check passed. PyTorch export warnings (e.g. aten::index) are treated as non-fatal unless validation fails."
                    )
                    ok_any = True
                    artifacts_ok += 1
            except Exception as e:
                print(f"[ERROR] ONNX export failed for {source_path}: {e}")
                failed_any = True
                artifacts_failed += 1

    if do_trt:
        stage_title = "Step TensorRT trt" if use_trtexec else "Step TensorRT engine"
        _print_stage_header(stage_title)
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
            engine_target = out_dir / _variant_or_legacy_name(source_path, ext=engine_suffix, args=args, imgsz=imgsz)
            if engine_target.exists() and not args.force:
                print(f"[WARN] Skip TensorRT (exists): {engine_target}. Use --force to rebuild.")
                skipped_any = True
                artifacts_skipped += 1
            elif source_ext == ".onnx":
                if not use_trtexec:
                    print(
                        f"[ERROR] TensorRT engine export is unsupported for ONNX input {source_path}: "
                        "Ultralytics engine export requires a .pt model."
                    )
                    failed_any = True
                    artifacts_failed += 1
                    return ok_any, failed_any, skipped_any, artifacts_ok, artifacts_failed, artifacts_skipped
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
                        print(f"[INFO] [START] TensorRT engine export: {source_path} -> {engine_target}")
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
                    if not use_trtexec:
                        print(f"[INFO] [DONE] TensorRT engine export: {engine_target}")
                        run_root = _guess_run_root_for_path(source_path)
                        write_model_sidecar_metadata(
                            engine_target,
                            format_name="engine",
                            run_dir=str(run_root) if run_root else None,
                            source_path=str(source_path),
                            tool="ultralytics",
                            params={
                                "imgsz": _format_imgsz(imgsz),
                                "batch": int(args.batch),
                                "dynamic": bool(args.dynamic),
                                "precision": str(args.precision),
                                "workspace_gib": getattr(args, "workspace_gib", None),
                            },
                        )
                    ok_any = True
                    artifacts_ok += 1
                else:
                    print(f"[ERROR] TensorRT export failed for {source_path}: {trt_reason}")
                    failed_any = True
                    artifacts_failed += 1
            else:
                expected_sig = _expected_onnx_signature(args, imgsz)
                public_onnx = _resolve_public_onnx_target(source_path, out_dir, expected_sig)
                dedicated_onnx = out_dir / (_make_dedicated_onnx_name(source_path.stem, expected_sig) + ".onnx")
                onnx_params = {
                    "imgsz": _format_imgsz(imgsz),
                    "batch": int(args.batch),
                    "dynamic": bool(args.dynamic),
                    "opset": int(args.opset),
                    "simplify": bool(getattr(args, "simplify", True)),
                    "half": bool(getattr(args, "half", False)),
                    "nms": bool(getattr(args, "nms", False)),
                }
                onnx_for_trt: Path | None = None
                if public_onnx.exists():
                    existing_sig = _extract_onnx_signature(public_onnx)
                    sig_ok, mismatches, warnings = _compare_onnx_signature(existing_sig, expected_sig)
                    if sig_ok:
                        onnx_for_trt = public_onnx
                        print(f"[INFO] Using existing public ONNX for TRT cache: {public_onnx}")
                        if warnings:
                            print(f"[WARN] Existing ONNX has incomplete metadata: {'; '.join(warnings[:8])}")
                    else:
                        details = "; ".join(mismatches[:8])
                        if len(mismatches) > 8:
                            details += f"; ... (+{len(mismatches)-8} more)"
                        print(f"[WARN] Existing public ONNX mismatch: {details}")

                if onnx_for_trt is None:
                    if model is None:
                        try:
                            from ultralytics import YOLO
                        except Exception as e:
                            print(f"[ERROR] ultralytics import failed: {e}")
                            return False, True, False
                        model = YOLO(str(source_path))
                    _cleanup_trtprep_artifacts(out_dir, source_path.stem)
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
                    run_root = _guess_run_root_for_path(source_path)
                    write_model_sidecar_metadata(
                        dedicated_onnx,
                        format_name="onnx",
                        run_dir=str(run_root) if run_root else None,
                        source_path=str(source_path),
                        tool="ultralytics",
                        params=onnx_params,
                    )
                    onnx_for_trt = dedicated_onnx
                    print(f"[OK] Created dedicated ONNX for TRT: {dedicated_onnx}")
                    ok_pub, pub_reason = _sync_onnx_artifact(
                        dedicated_onnx,
                        public_onnx,
                        force=True,
                        source_path_for_meta=source_path,
                        run_root=run_root,
                        params=onnx_params,
                    )
                    if not ok_pub:
                        print(f"[ERROR] Failed to update public ONNX from dedicated cache: {pub_reason}")
                        failed_any = True
                        artifacts_failed += 1
                        return ok_any, failed_any, skipped_any, artifacts_ok, artifacts_failed, artifacts_skipped
                    print(f"[OK] Public ONNX: {public_onnx}")
                    ok_any = True
                    artifacts_ok += 1
                elif dedicated_onnx.resolve() != onnx_for_trt.resolve():
                    _cleanup_trtprep_artifacts(out_dir, source_path.stem)
                    run_root = _guess_run_root_for_path(source_path)
                    ok_cache, cache_reason = _sync_onnx_artifact(
                        onnx_for_trt,
                        dedicated_onnx,
                        force=True,
                        source_path_for_meta=source_path,
                        run_root=run_root,
                        params=onnx_params,
                    )
                    if not ok_cache:
                        print(f"[ERROR] Failed to refresh dedicated ONNX cache: {cache_reason}")
                        failed_any = True
                        artifacts_failed += 1
                        return ok_any, failed_any, skipped_any, artifacts_ok, artifacts_failed, artifacts_skipped
                    print(f"[OK] Dedicated ONNX cache: {dedicated_onnx}")
                    onnx_for_trt = dedicated_onnx

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
                        print(f"[INFO] [START] TensorRT engine export: {source_path} -> {engine_target}")
                        exported = Path(str(model.export(**engine_kw))).expanduser().resolve()
                        ok_move, move_reason = _maybe_move_output(exported, engine_target, args.force)
                        if ok_move:
                            print(f"[OK] TensorRT: {engine_target}")
                            print(f"[INFO] [DONE] TensorRT engine export: {engine_target}")
                            run_root = _guess_run_root_for_path(source_path)
                            write_model_sidecar_metadata(
                                engine_target,
                                format_name="engine",
                                run_dir=str(run_root) if run_root else None,
                                source_path=str(source_path),
                                tool="ultralytics",
                                params={
                                    "imgsz": _format_imgsz(imgsz),
                                    "batch": int(args.batch),
                                    "dynamic": bool(args.dynamic),
                                    "precision": str(args.precision),
                                    "workspace_gib": getattr(args, "workspace_gib", None),
                                },
                            )
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

                ok_trt, trt_reason = _trtexec_export_from_onnx(onnx_for_trt, engine_target, args, imgsz)
                if ok_trt:
                    print(f"[OK] TensorRT: {engine_target}")
                    run_root = _guess_run_root_for_path(source_path)
                    write_model_sidecar_metadata(
                        engine_target,
                        format_name="trt",
                        run_dir=str(run_root) if run_root else None,
                        source_path=str(onnx_for_trt),
                        tool="trtexec",
                        params={
                            "imgsz": _format_imgsz(imgsz),
                            "batch": int(args.batch),
                            "dynamic": bool(args.dynamic),
                            "precision": str(args.precision),
                            "workspace_gib": getattr(args, "workspace_gib", None),
                        },
                    )
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

    if interactive_used:
        input_path = Path(str(args.input)).expanduser().resolve()
        imgsz, imgsz_source = _resolve_imgsz_from_args_and_model(args, input_path if input_path.is_file() else input_path)
        if args.imgsz is None:
            args.imgsz = _format_imgsz(imgsz)
        print(f"[INFO] Interactive summary: imgsz={_format_imgsz(imgsz)} (source: {imgsz_source})")

        # Build interactive context and run pipeline (single selection).
        source_kind: SourceKind = getattr(args, "_source_kind", "pt")
        target_onnx = bool(getattr(args, "_target_onnx", False))
        target_engine = bool(getattr(args, "_target_engine", False))
        target_trt = bool(getattr(args, "_target_trt", False))
        out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

        ctx = InteractiveContext(
            source_kind=source_kind,
            source_path=input_path,
            target_onnx=target_onnx,
            target_engine=target_engine,
            target_trt=target_trt,
            output_dir=out_dir,
            force=bool(args.force),
            force_onnx=bool(getattr(args, "_force_onnx", args.force)),
            force_engine=bool(getattr(args, "_force_engine", args.force)),
            force_trt=bool(getattr(args, "_force_trt", args.force)),
            onnx_imgsz=imgsz,
            onnx_imgsz_source=imgsz_source,
            onnx_batch=int(args.batch),
            onnx_dynamic=bool(args.dynamic),
            device=str(args.device) if args.device is not None else None,
            engine_precision=str(getattr(args, "_engine_precision", "fp32")),
            engine_workspace_gib=(
                float(getattr(args, "_engine_workspace_gib")) if getattr(args, "_engine_workspace_gib", None) is not None else None
            ),
            trt_precision=str(getattr(args, "_trt_precision", "fp32")),
            trt_workspace_gib=(
                float(getattr(args, "_trt_workspace_gib")) if getattr(args, "_trt_workspace_gib", None) is not None else None
            ),
            data=str(args.data) if args.data is not None else None,
            fraction=float(args.fraction),
            opset=int(getattr(args, "opset", 17) or 17),
            simplify=bool(getattr(args, "simplify", True)),
            half=bool(getattr(args, "half", False)),
            nms=bool(getattr(args, "nms", False)),
        )

        result = _run_interactive_pipeline(ctx)
        stats = result.stats
        print(
            f"[INFO] Done. total={stats.total} ok={stats.ok} failed={stats.failed} skipped={stats.skipped} "
            f"(artifacts: ok={stats.artifacts_ok} failed={stats.artifacts_failed} skipped={stats.artifacts_skipped})"
        )

        # Replay commands: emit current-step series (best-effort).
        cmds: list[str] = []
        if ctx.source_kind == "pt":
            if ctx.target_onnx:
                cmds.append(
                    build_non_interactive_command(
                        "model convert",
                        parser,
                        argparse.Namespace(**{**vars(args), "format": "onnx", "input": str(input_path)}),
                    )
                )
            if ctx.target_engine:
                cmds.append(
                    build_non_interactive_command(
                        "model convert",
                        parser,
                        argparse.Namespace(
                            **{
                                **vars(args),
                                "format": "tensorrt-engine",
                                "input": str(input_path),
                                "precision": getattr(args, "_engine_precision", "fp32"),
                                "workspace_gib": getattr(args, "_engine_workspace_gib", None),
                            }
                        ),
                    )
                )
            if ctx.target_trt:
                cmds.append(
                    build_non_interactive_command(
                        "model convert",
                        parser,
                        argparse.Namespace(
                            **{
                                **vars(args),
                                "format": "tensorrt-trt",
                                "input": str(input_path),
                                "precision": getattr(args, "_trt_precision", "fp32"),
                                "workspace_gib": getattr(args, "_trt_workspace_gib", None),
                            }
                        ),
                    )
                )
        else:
            if ctx.target_engine:
                cmds.append(
                    build_non_interactive_command(
                        "model convert",
                        parser,
                        argparse.Namespace(
                            **{
                                **vars(args),
                                "format": "tensorrt-engine",
                                "input": str(input_path),
                                "precision": getattr(args, "_engine_precision", "fp32"),
                                "workspace_gib": getattr(args, "_engine_workspace_gib", None),
                            }
                        ),
                    )
                )
            if ctx.target_trt:
                cmds.append(
                    build_non_interactive_command(
                        "model convert",
                        parser,
                        argparse.Namespace(
                            **{
                                **vars(args),
                                "format": "tensorrt-trt",
                                "input": str(input_path),
                                "precision": getattr(args, "_trt_precision", "fp32"),
                                "workspace_gib": getattr(args, "_trt_workspace_gib", None),
                            }
                        ),
                    )
                )
        if cmds:
            print("[INFO] Command(s) for non-interactive retry:")
            for c in cmds:
                print(c)

        if stats.failed > 0:
            raise SystemExit(1)
        return

    input_path = Path(str(args.input)).expanduser().resolve()
    models = _collect_input_models(input_path)
    if not models:
        print(f"[ERROR] No .pt/.onnx models found by input path: {input_path}")
        raise SystemExit(2)

    stats = ConvertStats(total=len(models))
    total_models = len(models)
    print(f"[INFO] Found {total_models} model(s) for conversion.")
    for idx, model_path in enumerate(models, start=1):
        _print_stage_header(f"Model {idx}/{total_models}")
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
    if stats.failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
