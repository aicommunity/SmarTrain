from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _safe_import_version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
    except Exception:
        return None
    return str(getattr(mod, "__version__", None) or "unknown")


def _read_command_output(command: list[str]) -> str | None:
    try:
        out = subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True, timeout=2)
        text = str(out or "").strip()
        return text or None
    except Exception:
        return None


def collect_environment_profile() -> dict[str, Any]:
    profile: dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "frameworks": {
            "torch": _safe_import_version("torch"),
            "ultralytics": _safe_import_version("ultralytics"),
            "onnxruntime": _safe_import_version("onnxruntime"),
            "tensorrt": _safe_import_version("tensorrt"),
            "numpy": _safe_import_version("numpy"),
            "pillow": _safe_import_version("PIL"),
        },
        "gpu": {
            "nvidia_smi": _read_command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        },
    }
    try:
        import torch  # type: ignore

        profile["python"]["torch_cuda_runtime"] = getattr(torch.version, "cuda", None)
        profile["gpu"]["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            profile["gpu"]["torch_cuda_device_count"] = int(torch.cuda.device_count())
            profile["gpu"]["torch_cuda_devices"] = [
                {"index": i, "name": torch.cuda.get_device_name(i)} for i in range(int(torch.cuda.device_count()))
            ]
    except Exception:
        pass
    return profile


def write_environment_profile(output_path: str, payload: dict[str, Any]) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)
