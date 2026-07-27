"""Host and filesystem profile capture for training/test runs."""

from __future__ import annotations

import os
import platform
import shutil
import socket
from typing import Any


def bytes_to_gb(value: int | float | None) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) / (1024.0**3), 3)
    except (TypeError, ValueError):
        return None


def linux_cpu_model_name() -> str | None:
    cpuinfo = "/proc/cpuinfo"
    if not os.path.isfile(cpuinfo):
        return None
    try:
        with open(cpuinfo, "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        value = parts[1].strip()
                        if value:
                            return value
    except Exception:
        return None
    return None


def linux_physical_core_count() -> int | None:
    cpuinfo = "/proc/cpuinfo"
    if not os.path.isfile(cpuinfo):
        return None
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    try:
        with open(cpuinfo, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    if current:
                        entries.append(current)
                        current = {}
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                current[key.strip().lower()] = value.strip()
        if current:
            entries.append(current)
    except Exception:
        return None
    cores: set[tuple[str, str]] = set()
    for item in entries:
        physical = item.get("physical id")
        core = item.get("core id")
        if physical is None or core is None:
            continue
        cores.add((physical, core))
    if cores:
        return len(cores)
    cpu_cores = entries[0].get("cpu cores") if entries else None
    if cpu_cores:
        try:
            n = int(cpu_cores)
            if n > 0:
                return n
        except ValueError:
            return None
    return None


def linux_mem_total_bytes() -> int | None:
    meminfo = "/proc/meminfo"
    if not os.path.isfile(meminfo):
        return None
    try:
        with open(meminfo, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        if kb > 0:
                            return kb * 1024
    except Exception:
        return None
    return None


def resolve_mount_point(path: str) -> str:
    cur = os.path.abspath(path)
    while not os.path.ismount(cur):
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return cur


def linux_fs_type_for_mount(mount_point: str) -> str | None:
    mounts_file = "/proc/mounts"
    if not os.path.isfile(mounts_file):
        return None
    normalized = os.path.abspath(mount_point)
    best_match = ""
    best_fs = None
    try:
        with open(mounts_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_raw = parts[1].replace("\\040", " ")
                fs_type = parts[2]
                if normalized == mount_raw or normalized.startswith(mount_raw + os.sep):
                    if len(mount_raw) > len(best_match):
                        best_match = mount_raw
                        best_fs = fs_type
    except Exception:
        return None
    return best_fs


def collect_system_profile(run_dir: str) -> dict[str, Any]:
    warnings: list[str] = []
    cpu_model = linux_cpu_model_name() or (platform.processor() or None)
    logical_cores = os.cpu_count()
    physical_cores = linux_physical_core_count()
    if cpu_model is None:
        warnings.append("cpu_model_unavailable")
    if physical_cores is None:
        warnings.append("cpu_physical_cores_unavailable")

    ram_total = linux_mem_total_bytes()
    if ram_total is None:
        warnings.append("ram_total_unavailable")

    gpu_devices: list[dict[str, Any]] = []
    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gpu_devices.append(
                    {
                        "index": idx,
                        "name": torch.cuda.get_device_name(idx),
                        "total_vram_bytes": int(getattr(props, "total_memory", 0) or 0),
                        "total_vram_gb": bytes_to_gb(getattr(props, "total_memory", 0) or 0),
                    }
                )
    except Exception:
        warnings.append("gpu_probe_failed")

    mount_point = resolve_mount_point(run_dir)
    disk_usage = shutil.disk_usage(run_dir)
    fs_type = linux_fs_type_for_mount(mount_point)
    if fs_type is None:
        warnings.append("filesystem_type_unavailable")

    gpu_total_bytes = sum(int(x.get("total_vram_bytes", 0) or 0) for x in gpu_devices) if gpu_devices else 0

    return {
        "cpu": {
            "model": cpu_model,
            "architecture": platform.machine() or None,
            "logical_cores": int(logical_cores) if logical_cores else None,
            "physical_cores": int(physical_cores) if physical_cores else None,
        },
        "ram": {
            "total_bytes": int(ram_total) if ram_total else None,
            "total_gb": bytes_to_gb(ram_total),
        },
        "gpu": {
            "cuda_available": cuda_available,
            "devices": gpu_devices,
            "total_vram_bytes": int(gpu_total_bytes) if gpu_total_bytes else 0,
            "total_vram_gb": bytes_to_gb(gpu_total_bytes),
        },
        "disk": {
            "mount_point": mount_point,
            "filesystem": fs_type,
            "total_bytes": int(disk_usage.total),
            "used_bytes": int(disk_usage.used),
            "free_bytes": int(disk_usage.free),
            "total_gb": bytes_to_gb(disk_usage.total),
            "used_gb": bytes_to_gb(disk_usage.used),
            "free_gb": bytes_to_gb(disk_usage.free),
        },
        "platform": {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
        },
        "capture_warnings": warnings,
    }
