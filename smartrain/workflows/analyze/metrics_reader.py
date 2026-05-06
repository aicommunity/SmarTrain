from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from smartrain.workflows.analyze.analyze_models import RunRecord
from smartrain.workflows.testing.model_test_service import (
    INTERNAL_TEST_FORMATS,
    SUPPORTED_TEST_FORMATS,
    format_test_dir,
    format_metrics_path,
    format_metrics_path_for_split,
    load_test_artifacts_manifest,
)

DEFAULT_MAP_COL = "metrics/mAP50-95(B)"
METRIC_AGG_COLUMNS = ("mAP50-95", "mAP50", "Box-F1", "Box-P", "Box-R")


def _infer_model_from_run_dir_name(run_dir: str) -> str | None:
    run_name = os.path.basename(os.path.abspath(run_dir))
    m = re.search(r"(yolo[a-z0-9]*[nslmx](?:-(?:seg|cls|pose|obb))?)", run_name, flags=re.IGNORECASE)
    return m.group(1).lower() if m else None


def _infer_model_from_args_yaml(run_dir: str) -> str | None:
    args_yaml = os.path.join(run_dir, "train-ultralytics", "args.yaml")
    legacy_args = os.path.join(run_dir, "train", "args.yaml")
    if not os.path.isfile(args_yaml) and os.path.isfile(legacy_args):
        args_yaml = legacy_args
    if not os.path.isfile(args_yaml):
        return None
    try:
        with open(args_yaml, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    raw_model = payload.get("model")
    if not isinstance(raw_model, str) or not raw_model.strip():
        return None
    token = os.path.basename(raw_model.strip())
    if token.endswith(".pt"):
        token = token[:-3]
    if token.endswith(".yaml"):
        token = token[:-5]
    token = token.strip().lower()
    if token in {"last", "best"}:
        return _infer_model_from_run_dir_name(run_dir)
    return token or None


def _infer_dataset_name_from_run_dir(run_dir: str) -> str | None:
    parent = os.path.basename(os.path.dirname(os.path.abspath(run_dir)))
    return parent or None


def load_metadata(run_dir: str) -> dict[str, Any]:
    path = os.path.join(run_dir, "training_metadata.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def latest_test_metrics_path(run_dir: str, format_name: str | None = "pt") -> str | None:
    if format_name not in (None, "", "pt"):
        p = format_metrics_path(run_dir, format_name)
        return p if os.path.isfile(p) else None
    # Prefer canonical layout produced by current test pipeline.
    canonical_pt_metrics = format_metrics_path(run_dir, "pt")
    if os.path.isfile(canonical_pt_metrics):
        return canonical_pt_metrics
    # Fallback for legacy runs that still store metrics in run root.
    legacy_pt_metrics = os.path.join(run_dir, "test_metrics.csv")
    return legacy_pt_metrics if os.path.isfile(legacy_pt_metrics) else None


def _iter_test_formats(include_internal: bool = False) -> tuple[str, ...]:
    return SUPPORTED_TEST_FORMATS + INTERNAL_TEST_FORMATS if include_internal else SUPPORTED_TEST_FORMATS


def _resolve_manifest_metrics_path(run_dir: str, rel_path: str) -> str | None:
    rel = str(rel_path or "").strip()
    if not rel:
        return None
    candidate = os.path.abspath(os.path.join(run_dir, rel))
    if os.path.isfile(candidate):
        return candidate
    # Legacy manifests may keep root-relative names while files already moved under tests/.
    basename = os.path.basename(rel)
    if basename:
        tests_candidate = os.path.join(run_dir, "tests", basename)
        if os.path.isfile(tests_candidate):
            return os.path.abspath(tests_candidate)
    return None


def read_test_metrics_by_format(run_dir: str, *, include_internal: bool = False) -> dict[str, str]:
    out: dict[str, str] = {}
    manifest = load_test_artifacts_manifest(run_dir)
    formats = manifest.get("formats")
    if isinstance(formats, dict):
        for fmt in _iter_test_formats(include_internal):
            entry = formats.get(fmt)
            if not isinstance(entry, dict):
                continue
            rel = entry.get("metrics_csv")
            selected: str | None = None
            if isinstance(rel, str) and rel.strip():
                p = _resolve_manifest_metrics_path(run_dir, rel)
                if p:
                    selected = p
            if selected is None:
                artifacts = entry.get("artifacts")
                if isinstance(artifacts, list):
                    for item in artifacts:
                        if not isinstance(item, dict):
                            continue
                        rel_item = item.get("metrics_csv")
                        if not isinstance(rel_item, str) or not rel_item.strip():
                            continue
                        p = _resolve_manifest_metrics_path(run_dir, rel_item)
                        if p:
                            selected = p
                            break
            if selected is not None:
                out[fmt] = selected
    for fmt in _iter_test_formats(include_internal):
        p = latest_test_metrics_path(run_dir, fmt)
        if p and os.path.isfile(p):
            out.setdefault(fmt, p)
    return out


def read_test_metrics_by_format_artifacts(
    run_dir: str,
    *,
    include_internal: bool = False,
) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    manifest = load_test_artifacts_manifest(run_dir)
    formats = manifest.get("formats")
    if not isinstance(formats, dict):
        return out
    for fmt in _iter_test_formats(include_internal):
        entry = formats.get(fmt)
        if not isinstance(entry, dict):
            continue
        records: list[dict[str, str]] = []
        artifacts = entry.get("artifacts")
        if isinstance(artifacts, list):
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                rel_metrics = item.get("metrics_csv")
                if not isinstance(rel_metrics, str) or not rel_metrics.strip():
                    continue
                metrics_path = _resolve_manifest_metrics_path(run_dir, rel_metrics)
                if not metrics_path:
                    continue
                rel_target = item.get("target_path")
                target_path = os.path.abspath(os.path.join(run_dir, rel_target)) if isinstance(rel_target, str) and rel_target else ""
                records.append({"metrics_path": metrics_path, "target_path": target_path})
        if not records:
            rel = entry.get("metrics_csv")
            if isinstance(rel, str) and rel.strip():
                metrics_path = _resolve_manifest_metrics_path(run_dir, rel)
                if metrics_path:
                    rel_target = entry.get("target_path")
                    target_path = (
                        os.path.abspath(os.path.join(run_dir, rel_target))
                        if isinstance(rel_target, str) and rel_target
                        else ""
                    )
                    records.append({"metrics_path": metrics_path, "target_path": target_path})
        if records:
            out[fmt] = records
    return out


def read_test_performance_by_format_artifacts(
    run_dir: str,
    *,
    include_internal: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    def _perf_target_guess(run_dir_local: str, fmt_local: str, stem: str) -> str:
        ext_by_fmt = {
            "pt": ".pt",
            "pt_uni": ".pt",
            "onnx": ".onnx",
            "engine": ".engine",
            "trt": ".trt",
        }
        ext = ext_by_fmt.get(fmt_local, "")
        if not ext:
            return stem
        candidate = os.path.abspath(os.path.join(run_dir_local, "models", f"{stem}{ext}"))
        return candidate if os.path.isfile(candidate) else stem

    out: dict[str, list[dict[str, Any]]] = {}
    manifest = load_test_artifacts_manifest(run_dir)
    formats = manifest.get("formats")
    if not isinstance(formats, dict):
        return out
    for fmt in _iter_test_formats(include_internal):
        entry = formats.get(fmt)
        if not isinstance(entry, dict):
            continue
        records: list[dict[str, Any]] = []
        artifacts = entry.get("artifacts")
        if isinstance(artifacts, list):
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                perf = item.get("performance")
                if not isinstance(perf, dict):
                    continue
                rel_target = item.get("target_path")
                target_path = os.path.abspath(os.path.join(run_dir, rel_target)) if isinstance(rel_target, str) and rel_target else ""
                records.append({"target_path": target_path, "performance": perf})
        # Secondary source for legacy runs: perf_*.json files in test directories.
        test_dir = format_test_dir(run_dir, fmt)
        if os.path.isdir(test_dir):
            for name in sorted(os.listdir(test_dir)):
                if not (name.startswith("perf_") and name.endswith(".json")):
                    continue
                perf_path = os.path.join(test_dir, name)
                try:
                    with open(perf_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                except Exception:
                    continue
                if not isinstance(payload, dict) or not payload:
                    continue
                stem = name[len("perf_") : -len(".json")]
                guessed_target = _perf_target_guess(run_dir, fmt, stem)
                # Avoid duplicate target entries from manifest and file fallback.
                if any(str(r.get("target_path") or "") == str(guessed_target) for r in records):
                    continue
                records.append({"target_path": guessed_target, "performance": payload})
        if records:
            out[fmt] = records
    return out


def read_test_system_profile_by_format_artifacts(
    run_dir: str,
    *,
    include_internal: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    manifest = load_test_artifacts_manifest(run_dir)
    formats = manifest.get("formats")
    if not isinstance(formats, dict):
        return out
    for fmt in _iter_test_formats(include_internal):
        entry = formats.get(fmt)
        if not isinstance(entry, dict):
            continue
        records: list[dict[str, Any]] = []
        artifacts = entry.get("artifacts")
        if isinstance(artifacts, list):
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                profile = item.get("test_system_profile")
                if not isinstance(profile, dict):
                    continue
                rel_target = item.get("target_path")
                target_path = os.path.abspath(os.path.join(run_dir, rel_target)) if isinstance(rel_target, str) and rel_target else ""
                records.append({"target_path": target_path, "test_system_profile": profile})
        if records:
            out[fmt] = records
    return out


def read_metrics_by_format_for_split(
    run_dir: str,
    split: str,
    *,
    include_internal: bool = False,
) -> dict[str, str]:
    split_name = str(split).strip().lower()
    if split_name == "test":
        return read_test_metrics_by_format(run_dir, include_internal=include_internal)
    out: dict[str, str] = {}
    for fmt in _iter_test_formats(include_internal):
        p = format_metrics_path_for_split(run_dir, split_name, fmt)
        if os.path.isfile(p):
            out[fmt] = p
    return out


def read_metrics_by_format_for_split_artifacts(
    run_dir: str,
    split: str,
    *,
    include_internal: bool = False,
) -> dict[str, list[dict[str, str]]]:
    split_name = str(split).strip().lower()
    if split_name == "test":
        return read_test_metrics_by_format_artifacts(run_dir, include_internal=include_internal)
    out: dict[str, list[dict[str, str]]] = {}
    for fmt in _iter_test_formats(include_internal):
        p = format_metrics_path_for_split(run_dir, split_name, fmt)
        if os.path.isfile(p):
            out[fmt] = [{"metrics_path": p, "target_path": ""}]
    return out


def results_csv_path(run_dir: str) -> str | None:
    """Resolve ``results.csv`` for a run, including sibling ``train-ultralytics*`` save dirs.

    Ultralytics increments the train folder name (e.g. ``train-ultralytics-2``) when
    ``exist_ok=False`` and the default name already exists; metrics then live under that
    folder, not only ``train-ultralytics/results.csv``.
    """
    root = Path(run_dir).expanduser().resolve()
    candidates: list[Path] = []
    for rel in ("train-ultralytics/results.csv", "train/results.csv"):
        p = root.joinpath(*rel.split("/"))
        if p.is_file():
            candidates.append(p)
    try:
        for p in root.glob("train-ultralytics*/results.csv"):
            if p.is_file():
                candidates.append(p)
    except OSError:
        pass
    seen: set[str] = set()
    unique: list[Path] = []
    for p in candidates:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    if not unique:
        return None
    return str(max(unique, key=lambda x: x.stat().st_mtime))


def training_args_yaml_path(run_dir: str, results_csv: str | None = None) -> str:
    """Prefer ``args.yaml`` next to ``results.csv`` (same Ultralytics ``save_dir``), then canonical paths."""
    root = Path(run_dir).expanduser().resolve()
    if results_csv:
        paired = Path(results_csv).expanduser().resolve().parent / "args.yaml"
        if paired.is_file():
            return str(paired)
    for rel in ("train-ultralytics/args.yaml", "train/args.yaml"):
        p = root.joinpath(*rel.split("/"))
        if p.is_file():
            return str(p)
    try:
        globs = sorted(
            (p for p in root.glob("train-ultralytics*/args.yaml") if p.is_file()),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if globs:
            return str(globs[0])
    except OSError:
        pass
    return str(root / "train-ultralytics" / "args.yaml")


def pick_map_column(df: pd.DataFrame) -> str | None:
    df.columns = [str(c).strip() for c in df.columns]
    for c in (DEFAULT_MAP_COL, "metrics/mAP50(B)"):
        if c in df.columns:
            return c
    for c in df.columns:
        if "mAP50" in c and "B" in c:
            return c
    return None


def flatten_metadata(md: dict[str, Any], run_dir: str) -> dict[str, Any]:
    row: dict[str, Any] = {"run_dir": run_dir}
    ti = md.get("training_info") or {}
    model = ti.get("model")
    if not model:
        model = _infer_model_from_args_yaml(run_dir)
    row["model"] = model
    dataset_name = (ti.get("dataset") or {}).get("name")
    if not dataset_name:
        dataset_name = _infer_dataset_name_from_run_dir(run_dir)
    row["dataset_name"] = dataset_name
    row["dataset_hash"] = (ti.get("dataset") or {}).get("hash")
    hp = ti.get("hyperparameters") or {}
    row["epochs"] = hp.get("epochs")
    row["batch_size"] = hp.get("batch_size")
    row["train_image_size"] = hp.get("image_size")
    inf = md.get("inference") or {}
    row["val_imgsz"] = inf.get("imgsz")
    row["val_conf"] = inf.get("conf")
    row["val_iou"] = inf.get("iou")
    st = md.get("status") or {}
    row["training_ok"] = (st.get("training") or {}).get("success")
    row["testing_ok"] = (st.get("testing") or {}).get("success")
    ts = md.get("timestamps") or {}
    row["training_duration_s"] = (ts.get("training") or {}).get("duration_seconds")
    sp = md.get("system_profile") or {}
    cpu = sp.get("cpu") or {}
    ram = sp.get("ram") or {}
    gpu = sp.get("gpu") or {}
    disk = sp.get("disk") or {}
    platform = sp.get("platform") or {}
    gpus = gpu.get("devices") if isinstance(gpu.get("devices"), list) else []
    row["sys_cpu_model"] = cpu.get("model")
    row["sys_cpu_arch"] = cpu.get("architecture")
    row["sys_cpu_logical_cores"] = cpu.get("logical_cores")
    row["sys_cpu_physical_cores"] = cpu.get("physical_cores")
    row["sys_ram_total_gb"] = ram.get("total_gb")
    row["sys_gpu_cuda_available"] = gpu.get("cuda_available")
    row["sys_gpu_count"] = len(gpus)
    row["sys_gpu_total_vram_gb"] = gpu.get("total_vram_gb")
    row["sys_gpu_0_name"] = gpus[0].get("name") if len(gpus) >= 1 and isinstance(gpus[0], dict) else None
    row["sys_gpu_0_vram_gb"] = (
        gpus[0].get("total_vram_gb") if len(gpus) >= 1 and isinstance(gpus[0], dict) else None
    )
    row["sys_disk_mount_point"] = disk.get("mount_point")
    row["sys_disk_fs"] = disk.get("filesystem")
    row["sys_disk_total_gb"] = disk.get("total_gb")
    row["sys_disk_free_gb"] = disk.get("free_gb")
    row["sys_os"] = platform.get("os")
    row["sys_os_release"] = platform.get("os_release")
    row["sys_python_version"] = platform.get("python_version")
    row["sys_hostname"] = platform.get("hostname")
    return row


def read_test_metrics_row(run_dir: str, format_name: str | None = "pt") -> dict[str, Any]:
    fmt = str(format_name or "pt").strip().lower()
    by_format = read_test_metrics_by_format(run_dir)
    tm = by_format.get(fmt) if by_format else None
    if not tm:
        tm = latest_test_metrics_path(run_dir, format_name)
    if not tm:
        return {}
    df = pd.read_csv(tm)
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) == 0:
        return {}
    # Prefer explicit aggregate row if present.
    if "Class" in df.columns:
        cls = df["Class"].astype(str).str.strip().str.lower()
        all_mask = cls.eq("all")
        if bool(all_mask.any()):
            return df.loc[all_mask].iloc[0].to_dict()
    # If metrics are per-class without an "all" row, build macro-average.
    if "Class" in df.columns and len(df) > 1:
        out: dict[str, Any] = {}
        for col in METRIC_AGG_COLUMNS:
            if col in df.columns:
                out[col] = pd.to_numeric(df[col], errors="coerce").mean()
        if out:
            out["Class"] = "all"
            return out
    return df.iloc[0].to_dict()


def read_train_last_row(run_dir: str, metric_column: str | None = None) -> dict[str, Any]:
    rc = results_csv_path(run_dir)
    if not rc:
        return {}
    df = pd.read_csv(rc)
    df.columns = [str(c).strip() for c in df.columns]
    if "epoch" not in df.columns or len(df) == 0:
        return {}
    mcol = metric_column if metric_column and metric_column in df.columns else pick_map_column(df)
    if mcol is None:
        return {}
    last = df.iloc[-1]
    return {"epoch": last.get("epoch"), mcol: last.get(mcol)}


def build_run_record(run_dir: str) -> RunRecord:
    md = load_metadata(run_dir)
    flat = flatten_metadata(md, run_dir)
    return RunRecord(
        run_dir=run_dir,
        model=flat.get("model"),
        dataset_name=flat.get("dataset_name"),
        training_ok=flat.get("training_ok"),
        testing_ok=flat.get("testing_ok"),
        training_duration_s=flat.get("training_duration_s"),
        test_metrics=read_test_metrics_row(run_dir),
        train_last_metrics=read_train_last_row(run_dir),
    )

