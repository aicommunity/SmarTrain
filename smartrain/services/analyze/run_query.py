from __future__ import annotations

import argparse
import json
import os
from typing import Any

from smartrain.core.runtime.run_discovery import find_run_directories
from smartrain.services.analyze.models import RunRecord


def matches_optional_bool(value: bool | None, expected: bool | None) -> bool:
    if expected is None:
        return True
    return value is expected


def build_run_record_unified(
    run_dir: str,
    *,
    read_test_metrics_for_run_cb: Any,
) -> RunRecord:
    from smartrain.orchestrators.unified_gateway import load_target

    payload = load_target(run_dir, source_kind="run")
    model_name: str | None = None
    dataset_name: str | None = None
    if payload.models:
        model_name = str(payload.models[0].model_id or "").strip() or None
    if payload.runs:
        dataset_name = str(payload.runs[0].dataset_ref or "").strip() or None
    metrics = read_test_metrics_for_run_cb(run_dir)
    return RunRecord(
        run_dir=run_dir,
        model=model_name,
        dataset_name=dataset_name,
        training_ok=None,
        testing_ok=None,
        training_duration_s=None,
        test_metrics=metrics,
        train_last_metrics={},
    )


def read_test_metrics_for_run(run_dir: str, *, format_name: str = "pt") -> dict[str, Any]:
    from smartrain.orchestrators.unified_gateway import load_metrics

    metric_refs = load_metrics(run_dir, source_kind="run", format_name=format_name)
    if metric_refs:
        out = dict(metric_refs[0].primary_metrics or {})
        out.update(dict(metric_refs[0].secondary_metrics or {}))
        return out
    return {}


def _system_profile_flat_from_training_metadata(run_dir: str) -> dict[str, Any]:
    """Map training_metadata.json system_profile (nested) to flat sys_* keys for system_profile_compare.csv."""
    path = os.path.join(run_dir, "training_metadata.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return {}
    sp = meta.get("system_profile")
    if not isinstance(sp, dict):
        return {}
    cpu = sp.get("cpu") if isinstance(sp.get("cpu"), dict) else {}
    ram = sp.get("ram") if isinstance(sp.get("ram"), dict) else {}
    gpu = sp.get("gpu") if isinstance(sp.get("gpu"), dict) else {}
    disk = sp.get("disk") if isinstance(sp.get("disk"), dict) else {}
    plat = sp.get("platform") if isinstance(sp.get("platform"), dict) else {}
    devices = gpu.get("devices") if isinstance(gpu.get("devices"), list) else []
    dev0 = devices[0] if len(devices) >= 1 and isinstance(devices[0], dict) else {}
    return {
        "sys_cpu_model": cpu.get("model"),
        "sys_cpu_arch": cpu.get("architecture"),
        "sys_cpu_logical_cores": cpu.get("logical_cores"),
        "sys_cpu_physical_cores": cpu.get("physical_cores"),
        "sys_ram_total_gb": ram.get("total_gb"),
        "sys_gpu_cuda_available": gpu.get("cuda_available"),
        "sys_gpu_count": len(devices) if devices else None,
        "sys_gpu_total_vram_gb": gpu.get("total_vram_gb"),
        "sys_gpu_0_name": dev0.get("name"),
        "sys_gpu_0_vram_gb": dev0.get("total_vram_gb"),
        "sys_disk_mount_point": disk.get("mount_point"),
        "sys_disk_fs": disk.get("filesystem"),
        "sys_disk_total_gb": disk.get("total_gb"),
        "sys_disk_free_gb": disk.get("free_gb"),
        "sys_os": plat.get("os"),
        "sys_os_release": plat.get("os_release"),
        "sys_python_version": plat.get("python_version"),
        "sys_hostname": plat.get("hostname"),
    }


def flat_row_unified(run_dir: str, *, build_run_record_cb: Any) -> dict[str, Any]:
    rec = build_run_record_cb(run_dir)
    out: dict[str, Any] = {
        "run_dir": run_dir,
        "run_name": os.path.basename(run_dir.rstrip(os.sep)),
        "model": rec.model,
        "dataset_name": rec.dataset_name,
    }
    out.update(_system_profile_flat_from_training_metadata(run_dir))
    return out


def filtered_run_records(
    args: argparse.Namespace,
    *,
    build_run_record_cb: Any,
) -> list[tuple[str, Any]]:
    runs = find_run_directories(args.models_root)
    recs: list[tuple[str, Any]] = []
    filter_dataset = getattr(args, "filter_dataset", None)
    filter_model = getattr(args, "filter_model", None)
    filter_training_ok = getattr(args, "filter_training_ok", None)
    filter_testing_ok = getattr(args, "filter_testing_ok", None)
    for run_dir in runs:
        try:
            rec = build_run_record_cb(run_dir)
        except Exception as e:
            print(f"[WARN] {run_dir}: failed to index run ({e})")
            continue
        if filter_dataset and rec.dataset_name != filter_dataset:
            continue
        if filter_model and rec.model != filter_model:
            continue
        if not matches_optional_bool(rec.training_ok, filter_training_ok):
            continue
        if not matches_optional_bool(rec.testing_ok, filter_testing_ok):
            continue
        recs.append((run_dir, rec))
    return recs
