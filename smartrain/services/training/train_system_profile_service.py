"""Compatibility re-export for system-profile helpers."""

from smartrain.core.runtime.system_profile_impl import (
    bytes_to_gb,
    collect_system_profile,
    linux_cpu_model_name,
    linux_fs_type_for_mount,
    linux_mem_total_bytes,
    linux_physical_core_count,
    resolve_mount_point,
)

__all__ = [
    "bytes_to_gb",
    "collect_system_profile",
    "linux_cpu_model_name",
    "linux_fs_type_for_mount",
    "linux_mem_total_bytes",
    "linux_physical_core_count",
    "resolve_mount_point",
]

