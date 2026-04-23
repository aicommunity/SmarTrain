from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from smartrain.external_providers.base import ExternalProviderSpec


def probe_provider_repo(repo_path: str, spec: ExternalProviderSpec, venv_path: str | None = None) -> dict[str, object]:
    root = Path(repo_path).expanduser().resolve()
    train_file = root / spec.train_entry
    infer_file = root / spec.infer_entry
    req_file = root / (spec.requirements_entry or "")
    py_bin = _venv_python_path(venv_path) if venv_path else None

    runtime = _runtime_probe(spec.id, py_bin, root)
    return {
        "repo_found": root.is_dir(),
        "entrypoints_ok": train_file.is_file() and infer_file.is_file(),
        "train_entry_ok": train_file.is_file(),
        "infer_entry_ok": infer_file.is_file(),
        "requirements_ok": req_file.is_file() if spec.requirements_entry else True,
        "venv_ready": bool(py_bin and py_bin.is_file()),
        "venv_python": str(py_bin) if py_bin else None,
        "runtime_ok": bool(runtime.get("ok", True)),
        "runtime_reason": str(runtime.get("reason", "") or ""),
    }


def _venv_python_path(venv_path: str) -> Path:
    vp = Path(venv_path).expanduser().resolve()
    if os.name == "nt":
        return vp / "Scripts" / "python.exe"
    return vp / "bin" / "python"


def external_python_in_env(venv_path: str) -> str:
    p = _venv_python_path(venv_path)
    return str(p if p.is_file() else Path(sys.executable))


def _runtime_probe(provider_id: str, py_bin: Path | None, repo_root: Path) -> dict[str, object]:
    if py_bin is None or not py_bin.is_file():
        return {"ok": False, "reason": "venv python not found"}
    if provider_id != "mfel-yolo":
        return {"ok": True, "reason": ""}
    cmd = [
        str(py_bin),
        "-c",
        "import sys;sys.path.insert(0, r'%s');import DCNv4" % str(repo_root),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except Exception as e:
        return {"ok": False, "reason": f"failed to validate DCNv4 import: {e}"}
    if proc.returncode == 0:
        return {"ok": True, "reason": ""}
    stderr = (proc.stderr or "").strip()
    if stderr:
        return {"ok": False, "reason": f"missing runtime dependency DCNv4 ({stderr})"}
    return {"ok": False, "reason": "missing runtime dependency DCNv4"}

