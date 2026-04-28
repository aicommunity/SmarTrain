from __future__ import annotations

import json
import os
import re
from glob import glob
from typing import Any

import pandas as pd
import yaml

from smartrain.analyze_models import RunRecord
from smartrain.model_test_service import (
    SUPPORTED_TEST_FORMATS,
    format_metrics_path,
    load_test_artifacts_manifest,
)

DEFAULT_MAP_COL = "metrics/mAP50-95(B)"


def _infer_model_from_run_dir_name(run_dir: str) -> str | None:
    run_name = os.path.basename(os.path.abspath(run_dir))
    m = re.search(r"(yolo[a-z0-9]*[nslmx](?:-(?:seg|cls|pose|obb))?)", run_name, flags=re.IGNORECASE)
    return m.group(1).lower() if m else None


def _infer_model_from_args_yaml(run_dir: str) -> str | None:
    args_yaml = os.path.join(run_dir, "train", "args.yaml")
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
    candidates = sorted(glob(os.path.join(run_dir, "test_metrics*.csv")))
    return candidates[-1] if candidates else None


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
            if not isinstance(rel, str) or not rel.strip():
                continue
            p = os.path.abspath(os.path.join(run_dir, rel))
            if os.path.isfile(p):
                out[fmt] = p
    for fmt in SUPPORTED_TEST_FORMATS:
        p = latest_test_metrics_path(run_dir, fmt)
        if p and os.path.isfile(p):
            out.setdefault(fmt, p)
    return out


def results_csv_path(run_dir: str) -> str | None:
    p = os.path.join(run_dir, "train", "results.csv")
    return p if os.path.exists(p) else None


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
    tm = latest_test_metrics_path(run_dir, format_name)
    if not tm:
        return {}
    df = pd.read_csv(tm)
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) == 0:
        return {}
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

