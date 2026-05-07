from __future__ import annotations

import os
from typing import Any, Callable

import pandas as pd


def write_system_profile_compare_csv(
    run_dirs: list[str],
    out_csv: str,
    *,
    flat_row_for_run_cb: Callable[[str], dict[str, Any]],
) -> str | None:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        try:
            row = flat_row_for_run_cb(run_dir)
        except Exception:
            continue
        rows.append(
            {
                "run_dir": run_dir,
                "run_name": os.path.basename(run_dir.rstrip(os.sep)),
                "model": row.get("model"),
                "dataset_name": row.get("dataset_name"),
                "sys_cpu_model": row.get("sys_cpu_model"),
                "sys_cpu_arch": row.get("sys_cpu_arch"),
                "sys_cpu_logical_cores": row.get("sys_cpu_logical_cores"),
                "sys_cpu_physical_cores": row.get("sys_cpu_physical_cores"),
                "sys_ram_total_gb": row.get("sys_ram_total_gb"),
                "sys_gpu_cuda_available": row.get("sys_gpu_cuda_available"),
                "sys_gpu_count": row.get("sys_gpu_count"),
                "sys_gpu_total_vram_gb": row.get("sys_gpu_total_vram_gb"),
                "sys_gpu_0_name": row.get("sys_gpu_0_name"),
                "sys_gpu_0_vram_gb": row.get("sys_gpu_0_vram_gb"),
                "sys_disk_mount_point": row.get("sys_disk_mount_point"),
                "sys_disk_fs": row.get("sys_disk_fs"),
                "sys_disk_total_gb": row.get("sys_disk_total_gb"),
                "sys_disk_free_gb": row.get("sys_disk_free_gb"),
                "sys_os": row.get("sys_os"),
                "sys_os_release": row.get("sys_os_release"),
                "sys_python_version": row.get("sys_python_version"),
                "sys_hostname": row.get("sys_hostname"),
            }
        )
    if not rows:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    return out_csv


def write_test_system_profile_compare_csv(
    run_dirs: list[str],
    out_csv: str,
    *,
    flat_row_for_run_cb: Callable[[str], dict[str, Any]],
    read_test_system_profile_by_format_artifacts_cb: Callable[[str], dict[str, list[dict[str, Any]]]],
) -> str | None:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        try:
            flat = flat_row_for_run_cb(run_dir)
        except Exception:
            flat = {}
        by_fmt = read_test_system_profile_by_format_artifacts_cb(run_dir)
        run_name = os.path.basename(run_dir.rstrip(os.sep))
        for fmt, records in by_fmt.items():
            for rec in records:
                profile = rec.get("test_system_profile") if isinstance(rec, dict) else None
                if not isinstance(profile, dict):
                    continue
                runtime = profile.get("runtime") if isinstance(profile.get("runtime"), dict) else {}
                cpu = profile.get("cpu") if isinstance(profile.get("cpu"), dict) else {}
                ram = profile.get("ram") if isinstance(profile.get("ram"), dict) else {}
                gpu = profile.get("gpu") if isinstance(profile.get("gpu"), dict) else {}
                platform = profile.get("platform") if isinstance(profile.get("platform"), dict) else {}
                devices = gpu.get("devices") if isinstance(gpu.get("devices"), list) else []
                rows.append(
                    {
                        "run_dir": run_dir,
                        "run_name": run_name,
                        "model": flat.get("model"),
                        "dataset_name": flat.get("dataset_name"),
                        "format": fmt,
                        "target_path": rec.get("target_path"),
                        "test_backend": runtime.get("backend"),
                        "test_provider": runtime.get("provider") or runtime.get("backend"),
                        "test_device": runtime.get("device"),
                        "sys_cpu_model": cpu.get("model"),
                        "sys_cpu_arch": cpu.get("architecture"),
                        "sys_cpu_logical_cores": cpu.get("logical_cores"),
                        "sys_cpu_physical_cores": cpu.get("physical_cores"),
                        "sys_ram_total_gb": ram.get("total_gb"),
                        "sys_gpu_cuda_available": gpu.get("cuda_available"),
                        "sys_gpu_count": len(devices),
                        "sys_gpu_total_vram_gb": gpu.get("total_vram_gb"),
                        "sys_gpu_0_name": devices[0].get("name") if len(devices) >= 1 and isinstance(devices[0], dict) else None,
                        "sys_gpu_0_vram_gb": (
                            devices[0].get("total_vram_gb") if len(devices) >= 1 and isinstance(devices[0], dict) else None
                        ),
                        "sys_os": platform.get("os"),
                        "sys_os_release": platform.get("os_release"),
                        "sys_python_version": platform.get("python_version"),
                        "sys_hostname": platform.get("hostname"),
                    }
                )
    if not rows:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    return out_csv

