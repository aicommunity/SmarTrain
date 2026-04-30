from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd
import yaml

from smartrain.analyze_models import RunRecord
from smartrain.model_test_service import (
    SUPPORTED_TEST_FORMATS,
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
    pt_metrics = os.path.join(run_dir, "test_metrics.csv")
    return pt_metrics if os.path.isfile(pt_metrics) else None


def read_test_metrics_by_format(run_dir: str) -> dict[str, str]:
    out: dict[str, str] = {}
    manifest = load_test_artifacts_manifest(run_dir)
    formats = manifest.get("formats")
    if isinstance(formats, dict):
        for fmt in SUPPORTED_TEST_FORMATS:
            entry = formats.get(fmt)
            if not isinstance(entry, dict):
                continue
            rel = entry.get("metrics_csv")
            selected: str | None = None
            if isinstance(rel, str) and rel.strip():
                p = os.path.abspath(os.path.join(run_dir, rel))
                if os.path.isfile(p):
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
                        p = os.path.abspath(os.path.join(run_dir, rel_item))
                        if os.path.isfile(p):
                            selected = p
                            break
            if selected is not None:
                out[fmt] = selected
    for fmt in SUPPORTED_TEST_FORMATS:
        p = latest_test_metrics_path(run_dir, fmt)
        if p and os.path.isfile(p):
            out.setdefault(fmt, p)
    return out


def read_test_metrics_by_format_artifacts(run_dir: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    manifest = load_test_artifacts_manifest(run_dir)
    formats = manifest.get("formats")
    if not isinstance(formats, dict):
        return out
    for fmt in SUPPORTED_TEST_FORMATS:
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
                metrics_path = os.path.abspath(os.path.join(run_dir, rel_metrics))
                if not os.path.isfile(metrics_path):
                    continue
                rel_target = item.get("target_path")
                target_path = os.path.abspath(os.path.join(run_dir, rel_target)) if isinstance(rel_target, str) and rel_target else ""
                records.append({"metrics_path": metrics_path, "target_path": target_path})
        if not records:
            rel = entry.get("metrics_csv")
            if isinstance(rel, str) and rel.strip():
                metrics_path = os.path.abspath(os.path.join(run_dir, rel))
                if os.path.isfile(metrics_path):
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


def read_test_performance_by_format_artifacts(run_dir: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    manifest = load_test_artifacts_manifest(run_dir)
    formats = manifest.get("formats")
    if not isinstance(formats, dict):
        return out
    for fmt in SUPPORTED_TEST_FORMATS:
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
        if records:
            out[fmt] = records
    return out


def read_test_system_profile_by_format_artifacts(run_dir: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    manifest = load_test_artifacts_manifest(run_dir)
    formats = manifest.get("formats")
    if not isinstance(formats, dict):
        return out
    for fmt in SUPPORTED_TEST_FORMATS:
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


def read_metrics_by_format_for_split(run_dir: str, split: str) -> dict[str, str]:
    split_name = str(split).strip().lower()
    if split_name == "test":
        return read_test_metrics_by_format(run_dir)
    out: dict[str, str] = {}
    for fmt in SUPPORTED_TEST_FORMATS:
        p = format_metrics_path_for_split(run_dir, split_name, fmt)
        if os.path.isfile(p):
            out[fmt] = p
    return out


def read_metrics_by_format_for_split_artifacts(run_dir: str, split: str) -> dict[str, list[dict[str, str]]]:
    split_name = str(split).strip().lower()
    if split_name == "test":
        return read_test_metrics_by_format_artifacts(run_dir)
    out: dict[str, list[dict[str, str]]] = {}
    for fmt in SUPPORTED_TEST_FORMATS:
        p = format_metrics_path_for_split(run_dir, split_name, fmt)
        if os.path.isfile(p):
            out[fmt] = [{"metrics_path": p, "target_path": ""}]
    return out


def results_csv_path(run_dir: str) -> str | None:
    p = os.path.join(run_dir, "train-ultralytics", "results.csv")
    if os.path.exists(p):
        return p
    legacy = os.path.join(run_dir, "train", "results.csv")
    return legacy if os.path.exists(legacy) else None


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

