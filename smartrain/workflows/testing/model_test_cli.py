from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from smartrain.cli_support.cli_argparse import CliArgumentParser
from smartrain.cli_support.cli_prompts import print_numbered_options, prompt_choice, prompt_text, prompt_yes_no
from smartrain.cli_support.cli_contracts import emit_replay, make_command_request
from smartrain.workflows.inference.inference_cli import _resolve_model_from_name, _resolve_run_ref
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.workflows.models import tensorrt_checks as trt_checks
from smartrain.workflows.testing.model_test_backends import run_native_format_backend, run_ultralytics_backend
from smartrain.workflows.testing.model_test_service import (
    SUPPORTED_TEST_FORMATS,
    complete_missing_test_artifacts,
    has_matching_test_artifacts,
    has_complete_test_artifacts,
    format_test_dir,
    resolve_root_dir_for_target,
)
from smartrain.services.model_test_orchestrator import run_model_test_after_setup
from smartrain.core.runtime.mpl_runtime import ensure_matplotlib_training_runtime
from smartrain.core.runtime.ultralytics_ephemeral import best_effort_prune_workspace_runs_detect
from smartrain.workflows.training.train_resume import resolve_dataset_path_for_resume
from smartrain.core.runtime.run_artifacts import (
    canonical_run_model_path,
    ensure_run_layout,
    is_internal_conversion_artifact,
    materialize_canonical_run_model,
    scan_run_models,
    run_models_dir,
)
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.core.runtime.device_selector import (
    default_device_value,
    device_display_name,
    prompt_device_selection,
    resolve_device_request,
    validate_device_available,
)
from smartrain.canonical.policy import emit_legacy_read_deprecation_warnings


def build_model_test_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Complete missing test artifacts for runs/models and compare formats (empty call starts interactive mode)."
    )
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR})")
    p.add_argument("--run", type=str, default=None, help="Run path or run index from workspace/runs.")
    p.add_argument("--model-name", type=str, default=None, help="Promoted model directory name from workspace/models.")
    p.add_argument("--weights", type=str, default=None, help="Explicit weights path (.pt/.onnx/.engine/.trt).")
    p.add_argument("--data", type=str, default=None, help="Dataset directory or path to data.yaml.")
    p.add_argument(
        "--formats",
        type=str,
        default=None,
        help="Comma-separated formats: pt,onnx,engine,trt. "
        "Default: pt in batch (-y); interactive + --run: all export formats found under the run unless you pass this flag.",
    )
    p.add_argument("--missing-only", action="store_true", help="Only build artifacts that are currently missing.")
    p.add_argument("--force", action="store_true", help="Force re-test even if matching artifacts already exist.")
    p.add_argument("--imgsz", type=int, default=None, help="Validation image size.")
    p.add_argument("--conf", type=float, default=None, help="Validation confidence threshold.")
    p.add_argument("--iou", type=float, default=None, help="Validation IoU threshold.")
    p.add_argument("--batch", type=int, default=None, help="Validation batch size.")
    p.add_argument(
        "--task",
        type=str,
        choices=["detect", "segment", "classify", "detection", "segmentation", "classification"],
        default=None,
        help="Task for backend routing (default: inferred from training metadata, else detect).",
    )
    p.add_argument("--device", type=str, default=None, help="Compute device: cpu, 0, cuda:0, or GPU name.")
    p.add_argument("--deep-diagnostics", action="store_true", help="Save deep per-image diagnostics artifacts.")
    p.add_argument(
        "--perf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect inference performance metrics.",
    )
    p.add_argument("--perf-warmup-images", type=int, default=5, help="Warmup images excluded from steady perf stats.")
    p.add_argument(
        "--onnx-provider-policy",
        type=str,
        choices=["gpu_strict", "gpu_preferred", "cpu_only"],
        default=None,
        help="ONNX provider policy: gpu_strict, gpu_preferred, cpu_only.",
    )
    p.add_argument(
        "--non-interactive",
        "-y",
        "--nit",
        action="store_true",
        dest="non_interactive",
        help="Disable interactive prompts (Typer also accepts --nit before subcommand flags).",
    )
    return p


def _parse_formats(raw: str) -> list[str]:
    out: list[str] = []
    for part in str(raw or "pt").split(","):
        fmt = part.strip().lower()
        if not fmt:
            continue
        if fmt not in SUPPORTED_TEST_FORMATS:
            raise ValueError(f"Unsupported format: {fmt}")
        if fmt not in out:
            out.append(fmt)
    return out or ["pt"]


def _discover_run_artifact_candidates(root_dir: str, formats: list[str] | None = None) -> dict[str, list[str]]:
    requested = list(formats or list(SUPPORTED_TEST_FORMATS))
    out: dict[str, list[str]] = {fmt: [] for fmt in requested}
    scanned = scan_run_models(root_dir)
    by_fmt: dict[str, list[str]] = {}
    for rec in scanned:
        fmt = str(rec.get("format") or "").strip().lower()
        path = str(rec.get("path") or "").strip()
        if not fmt or not path:
            continue
        if is_internal_conversion_artifact(path):
            continue
        by_fmt.setdefault(fmt, []).append(path)
    for fmt in requested:
        if fmt == "pt":
            best = canonical_run_model_path(root_dir, ".pt")
            if os.path.isfile(best):
                out[fmt] = [best]
            continue
        if fmt in by_fmt:
            out[fmt] = sorted(set(by_fmt[fmt]))
            continue
        try:
            one = _resolve_existing_artifact(
                root_dir=root_dir,
                primary_path=canonical_run_model_path(root_dir, ".pt"),
                format_name=fmt,
                target_kind="runs",
            )
            out[fmt] = [one]
        except Exception:
            out[fmt] = []
    # Keep deterministic order by format and path.
    ordered: dict[str, list[str]] = {}
    for fmt in SUPPORTED_TEST_FORMATS:
        if fmt in out and out[fmt]:
            ordered[fmt] = sorted(set(out[fmt]))
    return ordered


def _prompt_export_backends_interactive(root_dir: str, candidates: dict[str, list[str]]) -> list[str]:
    """Interactive step: choose pt/onnx/engine/trt. Always lists all four; marks missing weights."""
    entries: list[tuple[str, str, bool]] = []
    for fmt in SUPPORTED_TEST_FORMATS:
        paths = [p for p in (candidates.get(fmt) or []) if p and os.path.isfile(str(p))]
        if paths:
            try:
                rel = os.path.relpath(paths[0], root_dir)
            except ValueError:
                rel = str(paths[0])
            extra = f" (+{len(paths) - 1} more)" if len(paths) > 1 else ""
            entries.append((fmt, f"{fmt} — {rel}{extra}", True))
        else:
            entries.append((fmt, f"{fmt} — no artifact under this run", False))

    options = [e[1] for e in entries]
    print_numbered_options("test backends", options)
    default_nums = [str(i + 1) for i, e in enumerate(entries) if e[2]]
    default = ",".join(default_nums) if default_nums else "1"
    raw = prompt_text("Select backends to test (comma-separated numbers)", default=default).strip() or default
    out: list[str] = []
    seen: set[str] = set()
    for token in [t.strip() for t in raw.split(",") if t.strip()]:
        if not token.isdigit():
            continue
        i = int(token)
        if i < 1 or i > len(entries):
            continue
        fmt, _line, ok = entries[i - 1]
        if not ok:
            print(f"[WARN] {fmt}: skipped — no weights file for this format in the run.")
            continue
        if fmt not in seen:
            seen.add(fmt)
            out.append(fmt)
    if not out:
        out = [e[0] for e in entries if e[2]]
    return out


def _prompt_artifact_selection_interactive(candidates: dict[str, list[str]]) -> list[tuple[str, str]]:
    options: list[str] = []
    mapping: dict[int, tuple[str, str]] = {}
    idx = 1
    for fmt in SUPPORTED_TEST_FORMATS:
        paths = candidates.get(fmt, [])
        if not paths:
            continue
        for p in paths:
            rel = os.path.relpath(p, os.path.dirname(os.path.dirname(p))) if os.path.isabs(p) else p
            label = f"{fmt}: {rel}"
            options.append(label)
            mapping[idx] = (fmt, p)
            idx += 1
    if not options:
        return []
    print_numbered_options("models", options)
    default = ",".join(str(i) for i in mapping.keys())
    raw = prompt_text("Select models for test (comma-separated numbers)", default=default).strip() or default
    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for token in [t.strip() for t in raw.split(",") if t.strip()]:
        if not token.isdigit():
            continue
        i = int(token)
        item = mapping.get(i)
        if item is None or item in seen:
            continue
        seen.add(item)
        selected.append(item)
    return selected


def _artifact_key(fmt: str, path: str) -> str:
    return f"{fmt}|{os.path.abspath(path)}"


def _select_preferred_artifact(candidates: list[Path], fmt: str) -> Path | None:
    if not candidates:
        return None
    fmt_l = str(fmt or "").strip().lower()
    filtered = [p for p in candidates if p.is_file() and not is_internal_conversion_artifact(p)]
    if not filtered:
        return None

    def _score(p: Path) -> tuple[int, float, str]:
        name = p.name.lower()
        variant_hint = 1 if "_imgsz" in name and "_b" in name else 0
        # Prefer profile variants for ONNX, then freshest artifact.
        if fmt_l == "onnx":
            return (variant_hint, p.stat().st_mtime, name)
        return (0, p.stat().st_mtime, name)

    return max(filtered, key=_score)


def _normalize_data_to_yaml(data_value: str) -> str:
    candidate = Path(str(data_value)).expanduser().resolve()
    if candidate.is_file():
        return str(candidate)
    data_yaml = candidate / "data.yaml"
    if not data_yaml.is_file():
        raw = str(data_value or "").strip()
        hint = ""
        if "..." in raw:
            hint = " Path contains '...'; replace with full real path."
        raise FileNotFoundError(f"data.yaml not found for dataset: {candidate}.{hint}")
    return str(data_yaml)


def _suggest_convert_cmd(input_path: str, fmt: str) -> str:
    convert_fmt = {"onnx": "onnx", "engine": "tensorrt-engine", "trt": "tensorrt-trt"}.get(fmt, fmt)
    return f"smartrain model convert --input {input_path} --format {convert_fmt}"


def _resolve_existing_artifact(
    *,
    root_dir: str,
    primary_path: str,
    format_name: str,
    target_kind: str,
) -> str:
    fmt = str(format_name).strip().lower()
    if fmt in {"pt", "pt_uni"}:
        return primary_path
    ext_map = {"onnx": ".onnx", "engine": ".engine", "trt": ".trt"}
    ext = ext_map.get(fmt)
    if not ext:
        raise ValueError(f"Unsupported format: {format_name}")

    p = Path(primary_path)
    if p.suffix.lower() == ext and p.is_file():
        return str(p.resolve())

    root = Path(root_dir)
    candidates: list[Path] = []
    if target_kind == "runs":
        models_dir = run_models_dir(str(root))
        candidates.extend(sorted(models_dir.glob(f"*{ext}")))
        run_pt = Path(canonical_run_model_path(str(root), ".pt"))
        candidates.extend(
            [
                run_pt.with_suffix(ext),
            ]
        )
        candidates.extend(sorted(root.glob(f"*{ext}")))
        # Some conversion workflows place artifacts into nested folders under the run dir.
        candidates.extend(sorted(root.rglob(f"*{ext}")))
    else:
        candidates.extend(sorted(root.glob(f"*{ext}")))
        candidates.extend(sorted(root.rglob(f"*{ext}")))

    preferred = _select_preferred_artifact(candidates, fmt)
    if preferred is not None:
        return str(preferred.resolve())

    if primary_path.lower().endswith(".pt"):
        raise RuntimeError(
            f"Missing {fmt} artifact for target. Expected an existing {ext} file under {root_dir}. "
            f"Convert first: {_suggest_convert_cmd(primary_path, fmt)}"
        )
    raise RuntimeError(
        f"Missing {fmt} artifact for target. Expected an existing {ext} file under {root_dir} "
        f"or pass explicit --weights {ext}."
    )


def _resolve_target(args: argparse.Namespace, layout: WorkspaceLayout) -> tuple[str, str, str, str | None]:
    if args.run:
        run_dir = _resolve_run_ref(layout, str(args.run))
        if not run_dir.is_dir():
            raw = str(args.run or "").strip()
            hint = " Path contains '...'; replace with full real path." if "..." in raw else ""
            raise FileNotFoundError(f"Run directory not found: {run_dir}.{hint}")
        ensure_run_layout(str(run_dir))
        best_pt = Path(canonical_run_model_path(str(run_dir), ".pt"))
        if not best_pt.is_file():
            materialized = materialize_canonical_run_model(str(run_dir), ext=".pt", move=True, normalize_metadata=True)
            if materialized is not None:
                best_pt = Path(materialized)
        return str(run_dir), str(best_pt), "runs", run_dir.name
    if args.model_name:
        model_path, model_key = _resolve_model_from_name(layout, str(args.model_name).strip())
        return str(model_path.parent), str(model_path), "models", model_key
    if args.weights:
        weights_path = Path(str(args.weights)).expanduser()
        if not weights_path.is_absolute():
            weights_path = (Path(layout.root) / weights_path).resolve()
        return resolve_root_dir_for_target(str(weights_path)), str(weights_path), "weights", weights_path.stem
    raise ValueError("Specify one of --run, --model-name or --weights.")


def _pick_interactive_target(layout: WorkspaceLayout) -> tuple[str, str, str, str | None]:
    options = ["runs", "models", "weights"]
    selected = prompt_choice("Test source", options, default="runs")
    if selected == "runs":
        from smartrain.core.runtime.run_discovery import find_run_directories

        runs = find_run_directories(layout.runs)
        if not runs:
            raise RuntimeError("No runs found.")
        printable = [os.path.relpath(x, layout.root) for x in runs]
        print_numbered_options("runs", printable)
        chosen = prompt_choice("Select run", runs, default=runs[0], show_options=False)
        run_dir = Path(chosen).resolve()
        ensure_run_layout(str(run_dir))
        run_pt = Path(canonical_run_model_path(str(run_dir), ".pt"))
        if not run_pt.is_file():
            materialized = materialize_canonical_run_model(str(run_dir), ext=".pt", move=True, normalize_metadata=True)
            if materialized is not None:
                run_pt = Path(materialized)
        return str(run_dir), str(run_pt), "runs", run_dir.name
    if selected == "models":
        entries = sorted(d.name for d in Path(layout.models).iterdir() if d.is_dir()) if os.path.isdir(layout.models) else []
        if not entries:
            raise RuntimeError("No promoted models found.")
        print_numbered_options("models", entries)
        chosen = prompt_choice("Select model", entries, default=entries[0], show_options=False)
        model_path, model_key = _resolve_model_from_name(layout, chosen)
        return str(model_path.parent), str(model_path), "models", model_key
    raw = prompt_text("Weights path", default="models").strip() or "models"
    weights_path = Path(raw).expanduser()
    if not weights_path.is_absolute():
        weights_path = (Path(layout.root) / weights_path).resolve()
    return resolve_root_dir_for_target(str(weights_path)), str(weights_path), "weights", weights_path.stem


def _resolve_data_yaml_for_target(
    *,
    target_kind: str,
    root_dir: str,
    layout: WorkspaceLayout,
    data_cli: str | None,
) -> str:
    if data_cli:
        return _normalize_data_to_yaml(data_cli)
    if target_kind == "runs":
        dataset_dir = resolve_dataset_path_for_resume(root_dir, layout.root)
        if not dataset_dir:
            raise RuntimeError("Failed to resolve dataset path for run.")
        return _normalize_data_to_yaml(dataset_dir)
    manifest_path = os.path.join(root_dir, "model_manifest.json")
    if os.path.isfile(manifest_path):
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        source_run = payload.get("source_run")
        if isinstance(source_run, str) and source_run.strip():
            dataset_dir = resolve_dataset_path_for_resume(source_run, layout.root)
            if dataset_dir:
                return _normalize_data_to_yaml(dataset_dir)
    raise RuntimeError("Dataset path is required for models/weights targets. Pass --data.")


def _print_test_plan(
    *,
    target_kind: str,
    target_label: str | None,
    root_dir: str,
    data_yaml: str,
    formats: list[str],
    split_name: str = "test",
) -> None:
    print("[INFO] Test plan:")
    print(f"  target:  {target_kind} {target_label or root_dir}")
    print(f"  dataset: {data_yaml}")
    print(f"  split:   {split_name}")
    print(f"  formats: {', '.join(formats)}")


def _resolve_default_inference_params(root_dir: str) -> dict[str, int | float | None]:
    defaults: dict[str, int | float | None] = {"imgsz": None, "conf": None, "iou": None, "batch": None}
    metadata_path = Path(root_dir) / "training_metadata.json"
    if metadata_path.is_file():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            inf = payload.get("inference")
            if isinstance(inf, dict):
                for key in ("imgsz", "conf", "iou", "batch"):
                    if inf.get(key) is not None:
                        defaults[key] = inf.get(key)
        except Exception:
            pass
    train_args = Path(root_dir) / "train" / "args.yaml"
    if train_args.is_file():
        try:
            payload = json.loads(json.dumps(yaml.safe_load(train_args.read_text(encoding="utf-8")) or {}))
            if defaults["imgsz"] is None and payload.get("imgsz") is not None:
                defaults["imgsz"] = payload.get("imgsz")
            if defaults["iou"] is None and payload.get("iou") is not None:
                defaults["iou"] = payload.get("iou")
            if defaults["batch"] is None and payload.get("batch") is not None:
                defaults["batch"] = payload.get("batch")
        except Exception:
            pass
    if defaults["imgsz"] is None:
        defaults["imgsz"] = 640
    if defaults["conf"] is None:
        defaults["conf"] = 0.001
    if defaults["iou"] is None:
        defaults["iou"] = 0.7
    return defaults


def _normalize_task_for_backend(task: str | None) -> str:
    value = str(task or "").strip().lower()
    if value in {"detection", "detect", ""}:
        return "detect"
    if value in {"segmentation", "segment"}:
        return "segment"
    if value in {"classification", "classify"}:
        return "classify"
    return "detect"


def _infer_task_from_training_metadata(root_dir: str) -> str | None:
    emit_legacy_read_deprecation_warnings()
    from smartrain.orchestrators.canonical_gateway import resolve_task_context

    ctx = resolve_task_context(root_dir)
    return _normalize_task_for_backend(ctx.task_type)


def _has_deep_diagnostics_artifacts(root_dir: str, format_name: str) -> bool:
    deep_dir = os.path.join(format_test_dir(root_dir, format_name), "deep_diagnostics")
    if not os.path.isdir(deep_dir):
        return False
    for split in ("test", "val"):
        if not os.path.isfile(os.path.join(deep_dir, f"debug_{split}.jsonl")):
            return False
        if not os.path.isfile(os.path.join(deep_dir, f"debug_{split}_summary.json")):
            return False
    return os.path.isfile(os.path.join(deep_dir, "debug_params.json"))


def _should_rerun_existing_match(
    *,
    interactive: bool,
    force: bool,
    root_dir: str,
    format_name: str,
    target_path: str,
    dataset_yaml: str,
    imgsz: int | None,
    conf: float | None,
    iou: float | None,
    deep_diagnostics: bool = False,
) -> bool:
    if force:
        return True
    matches = has_matching_test_artifacts(
        root_dir,
        format_name=format_name,
        target_path=target_path,
        dataset_yaml=dataset_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )
    if not matches:
        if has_complete_test_artifacts(root_dir, format_name):
            if any(v is not None for v in (imgsz, conf, iou)):
                print(
                    f"[INFO] {format_name}: existing artifacts found, but model/dataset/params differ "
                    f"(imgsz={imgsz}, conf={conf}, iou={iou}). Recomputing."
                )
            else:
                print(
                    f"[INFO] {format_name}: existing artifacts found, but model or dataset differs. Recomputing."
                )
        return True

    if deep_diagnostics and not _has_deep_diagnostics_artifacts(root_dir, format_name):
        if interactive:
            return prompt_yes_no(
                f"{format_name}: deep-diagnostics artifacts are missing. Re-run",
                default=True,
            )
        print(f"[INFO] {format_name}: deep-diagnostics artifacts are missing. Recomputing.")
        return True
    if interactive:
        return prompt_yes_no(
            f"{format_name}: matching test artifacts already exist for this model and dataset. Re-run",
            default=False,
        )
    print(f"[INFO] {format_name}: matching test artifacts already exist for this model and dataset, skipping.")
    return False


def _run_native_backend_isolated(
    *,
    root_dir: str,
    weights_path: str,
    dataset_yaml_path: str,
    format_name: str,
    imgsz: int | None,
    val_conf: float | None,
    val_iou: float | None,
    val_batch: int | None,
    collect_performance: bool = False,
    perf_warmup_images: int = 5,
    onnx_provider_policy: str | None = None,
    runtime_device: str | None = None,
) -> tuple[bool, str | None]:
    with tempfile.NamedTemporaryFile(prefix=f"smartrain_test_{format_name}_", suffix=".json", delete=False) as tmp:
        result_path = tmp.name
    try:
        cmd = [
            sys.executable,
            "-m",
            "smartrain.workflows.testing.model_test_backend_runner",
            "--root-dir",
            root_dir,
            "--weights-path",
            weights_path,
            "--dataset-yaml-path",
            dataset_yaml_path,
            "--format-name",
            format_name,
            "--result-json",
            result_path,
        ]
        if imgsz is not None:
            cmd.extend(["--imgsz", str(imgsz)])
        if val_conf is not None:
            cmd.extend(["--conf", str(val_conf)])
        if val_iou is not None:
            cmd.extend(["--iou", str(val_iou)])
        if val_batch is not None:
            cmd.extend(["--batch", str(val_batch)])
        if collect_performance:
            cmd.append("--perf")
            cmd.extend(["--perf-warmup-images", str(max(0, int(perf_warmup_images)))])
        if onnx_provider_policy:
            cmd.extend(["--onnx-provider-policy", str(onnx_provider_policy)])
        if runtime_device:
            cmd.extend(["--device", str(runtime_device)])
        # Stream child process output directly to the current terminal so tqdm
        # progress bars from native backends (engine/trt) remain visible.
        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0:
            tail = "native backend crashed (see logs above)"
            if proc.returncode < 0:
                tail = f"native backend terminated by signal {-proc.returncode}: {tail}"
            else:
                tail = f"native backend exit_code={proc.returncode}: {tail}"
            return False, tail
        try:
            payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"native backend completed without valid result payload: {exc}"
        success = bool(payload.get("success"))
        error = payload.get("error")
        return success, (str(error) if error else None)
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass


def _collect_interactive_rerun_decisions(
    *,
    interactive: bool,
    force: bool,
    missing_only: bool,
    root_dir: str,
    target_kind: str,
    primary_path: str,
    formats: list[str],
    selected_artifacts: list[tuple[str, str]] | None,
    data_yaml: str,
    requested_imgsz: int | None,
    requested_conf: float | None,
    requested_iou: float | None,
    deep_diagnostics: bool,
) -> dict[str, bool]:
    decisions: dict[str, bool] = {}
    if not interactive or force:
        return decisions

    # Ask all overwrite/rerun questions upfront before any heavy work starts.
    if "pt" in formats:
        if target_kind == "runs" and (not missing_only or not has_complete_test_artifacts(root_dir, "pt")):
            decisions[_artifact_key("pt", primary_path)] = _should_rerun_existing_match(
                interactive=interactive,
                force=force,
                root_dir=root_dir,
                format_name="pt",
                target_path=primary_path,
                dataset_yaml=data_yaml,
                imgsz=requested_imgsz,
                conf=requested_conf,
                iou=requested_iou,
                deep_diagnostics=deep_diagnostics,
            )
        elif target_kind in {"models", "weights"} and (not missing_only or not has_complete_test_artifacts(root_dir, "pt")):
            decisions[_artifact_key("pt", primary_path)] = _should_rerun_existing_match(
                interactive=interactive,
                force=force,
                root_dir=root_dir,
                format_name="pt",
                target_path=primary_path,
                dataset_yaml=data_yaml,
                imgsz=requested_imgsz,
                conf=requested_conf,
                iou=requested_iou,
                deep_diagnostics=deep_diagnostics,
            )

    entries: list[tuple[str, str]] = []
    if selected_artifacts:
        entries.extend(selected_artifacts)
    else:
        for fmt in ("onnx", "engine", "trt"):
            if fmt not in formats:
                continue
            if missing_only and has_complete_test_artifacts(root_dir, fmt):
                continue
            try:
                artifact_path = _resolve_existing_artifact(
                    root_dir=root_dir,
                    primary_path=primary_path,
                    format_name=fmt,
                    target_kind=target_kind,
                )
            except Exception:
                continue
            entries.append((fmt, artifact_path))

    for fmt, artifact_path in entries:
        decisions[_artifact_key(fmt, artifact_path)] = _should_rerun_existing_match(
            interactive=interactive,
            force=force,
            root_dir=root_dir,
            format_name=fmt,
            target_path=artifact_path,
            dataset_yaml=data_yaml,
            imgsz=requested_imgsz,
            conf=requested_conf,
            iou=requested_iou,
            deep_diagnostics=deep_diagnostics,
        )

    return decisions


def _check_native_format_preflight(format_name: str) -> tuple[bool, str | None]:
    fmt = str(format_name).strip().lower()
    if fmt not in {"engine", "trt"}:
        return True, None
    ready, reason = trt_checks.check_tensorrt_ready()
    if not ready:
        return False, reason
    cuda_ready, cuda_reason = trt_checks.check_python_cuda_runtime_ready()
    if not cuda_ready:
        return False, cuda_reason
    return True, None


def _check_onnx_format_preflight(policy: str) -> tuple[bool, str | None]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        return False, f"onnxruntime import failed: {exc}"
    available = list(ort.get_available_providers())
    if policy == "cpu_only":
        if "CPUExecutionProvider" in available:
            return True, "onnx policy=cpu_only: CUDA provider disabled by policy."
        return False, "CPUExecutionProvider is unavailable."
    if "CUDAExecutionProvider" in available:
        return True, None
    if policy == "gpu_strict":
        return False, f"CUDAExecutionProvider is unavailable. available={available}"
    if "CPUExecutionProvider" in available:
        return True, f"CUDAExecutionProvider is unavailable. available={available}; CPU fallback may be used."
    return False, f"No usable ONNX providers. available={available}"


def main(argv: list[str] | None = None) -> None:
    parser = build_model_test_arg_parser()
    args = parser.parse_args(argv)
    argv_list = list(argv) if argv is not None else []
    request = make_command_request(
        "test", argv_list, interactive_allowed=is_interactive_allowed(argv_list)
    )
    interactive = request.interactive_allowed
    ensure_matplotlib_training_runtime(non_interactive=not interactive)
    workspace_root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(workspace_root)
    atexit.register(lambda wr=workspace_root: best_effort_prune_workspace_runs_detect(wr))
    user_onnx_policy = getattr(args, "onnx_provider_policy", None)
    args.device = resolve_device_request(getattr(args, "device", None) or default_device_value())
    onnx_provider_policy = str(
        user_onnx_policy or os.getenv("SMARTTRAIN_ONNX_PROVIDER_POLICY", "gpu_preferred")
    ).strip()
    if onnx_provider_policy not in {"gpu_strict", "gpu_preferred", "cpu_only"}:
        parser.error(f"Unsupported --onnx-provider-policy: {onnx_provider_policy}")

    if not any((args.run, args.model_name, args.weights)):
        if not interactive:
            parser.error("Specify one of --run, --model-name or --weights in non-interactive mode.")
        root_dir, primary_path, target_kind, target_label = _pick_interactive_target(layout)
        formats = _parse_formats("pt,onnx,engine,trt")
        default_data_yaml: str | None = None
        try:
            default_data_yaml = _resolve_data_yaml_for_target(
                target_kind=target_kind,
                root_dir=root_dir,
                layout=layout,
                data_cli=None,
            )
        except Exception:
            default_data_yaml = None
        raw_data = prompt_text("Dataset path or data.yaml", default=(default_data_yaml or "")).strip()
        try:
            data_yaml = _resolve_data_yaml_for_target(
                target_kind=target_kind,
                root_dir=root_dir,
                layout=layout,
                data_cli=(raw_data or default_data_yaml),
            )
        except FileNotFoundError as exc:
            parser.error(str(exc))
        defaults = _resolve_default_inference_params(root_dir)
        args.imgsz = int(defaults["imgsz"]) if defaults["imgsz"] is not None else None
        args.conf = float(defaults["conf"]) if defaults["conf"] is not None else None
        args.iou = float(defaults["iou"]) if defaults["iou"] is not None else None
        # Keep test runs memory-stable by default across backends.
        args.batch = 1 if args.batch is None else int(args.batch)
        args.run = None
        args.model_name = None
        args.weights = None
        if target_kind == "runs":
            args.run = root_dir
        elif target_kind == "models":
            args.model_name = target_label
        else:
            args.weights = primary_path
        args.data = data_yaml
        args.formats = ",".join(formats)
        if sys.stdin.isatty():
            args.device = prompt_device_selection(title="test devices", default_device=str(args.device or default_device_value()))
    else:
        try:
            root_dir, primary_path, target_kind, target_label = _resolve_target(args, layout)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        if args.formats is None:
            if interactive and target_kind == "runs":
                args.formats = "pt,onnx,engine,trt"
            else:
                args.formats = "pt"
        formats = _parse_formats(args.formats)
        try:
            data_yaml = _resolve_data_yaml_for_target(target_kind=target_kind, root_dir=root_dir, layout=layout, data_cli=args.data)
        except FileNotFoundError as exc:
            parser.error(str(exc))
        defaults = _resolve_default_inference_params(root_dir)
        if args.imgsz is None:
            args.imgsz = int(defaults["imgsz"]) if defaults["imgsz"] is not None else None
        if args.conf is None:
            args.conf = float(defaults["conf"]) if defaults["conf"] is not None else None
        if args.iou is None:
            args.iou = float(defaults["iou"]) if defaults["iou"] is not None else None
        if args.batch is None:
            args.batch = 1
    if args.task is None:
        args.task = _infer_task_from_training_metadata(root_dir) or "detect"
    else:
        args.task = _normalize_task_for_backend(args.task)
    # Effective params after defaults — must match args.* so pt_uni / skip logic compares real imgsz.
    requested_imgsz = args.imgsz
    requested_conf = args.conf
    requested_iou = args.iou
    args.device = resolve_device_request(args.device or default_device_value())
    if user_onnx_policy is None and str(args.device).strip().lower() == "cpu":
        onnx_provider_policy = "cpu_only"
    try:
        validate_device_available(args.device)
    except Exception as exc:
        parser.error(str(exc))
    print(f"[INFO] Test device: {device_display_name(args.device)}")

    run_model_test_after_setup(
        parser=parser,
        args=args,
        request=request,
        workspace_root=workspace_root,
        interactive=interactive,
        root_dir=root_dir,
        primary_path=primary_path,
        target_kind=target_kind,
        target_label=target_label,
        data_yaml=data_yaml,
        formats=formats,
        onnx_provider_policy=onnx_provider_policy,
        requested_imgsz=requested_imgsz,
        requested_conf=requested_conf,
        requested_iou=requested_iou,
    )


if __name__ == "__main__":
    main()
