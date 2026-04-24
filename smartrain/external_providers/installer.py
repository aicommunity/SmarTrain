from __future__ import annotations

import os
import shutil
import subprocess
import venv
import zipfile
from dataclasses import dataclass
from pathlib import Path

from smartrain.external_providers.probe import probe_provider_repo
from smartrain.external_providers.registry import get_provider_spec
from smartrain.provider_global_index import mark_provider_state, upsert_provider_record
from smartrain.provider_install_state import write_provider_state


@dataclass(frozen=True)
class InstallResult:
    provider_id: str
    action: str
    ok: bool
    message: str


def providers_root(target_dir: str | None) -> Path:
    base = Path(target_dir or os.getcwd()).expanduser().resolve()
    return base / "smartrain-providers"


def install_provider(provider_id: str, target_dir: str | None = None, *, allow_pull: bool = False) -> InstallResult:
    spec = get_provider_spec(provider_id)
    if not spec.ready:
        return InstallResult(provider_id=provider_id, action="skipped", ok=False, message=spec.note or "Provider not ready.")
    root = providers_root(target_dir)
    repo_dir = root / spec.id
    venv_dir = repo_dir / "venv"
    root.mkdir(parents=True, exist_ok=True)

    try:
        if repo_dir.is_dir() and not (repo_dir / ".git").is_dir():
            return InstallResult(provider_id=provider_id, action="failed", ok=False, message=f"Existing folder is not git repo: {repo_dir}")
        if not repo_dir.is_dir():
            _run(["git", "clone", "--branch", spec.branch, spec.repo_url, str(repo_dir)], cwd=str(root))
            repo_action = "cloned"
        elif allow_pull:
            _run(["git", "pull"], cwd=str(repo_dir))
            repo_action = "updated"
        else:
            repo_action = "existing"

        effective_repo_dir = _resolve_effective_repo_dir(spec.id, repo_dir)

        if not venv_dir.is_dir():
            venv.EnvBuilder(with_pip=True).create(str(venv_dir))
            env_action = "venv-created"
        else:
            env_action = "venv-existing"

        py = _venv_python(venv_dir)
        req = effective_repo_dir / (spec.requirements_entry or "")
        if req.is_file():
            _run([py, "-m", "pip", "install", "-r", str(req)], cwd=str(effective_repo_dir))
        else:
            _run([py, "-m", "pip", "install", "ultralytics"], cwd=str(effective_repo_dir))
        _install_provider_runtime_deps(spec.id, effective_repo_dir, py)

        probe = probe_provider_repo(str(effective_repo_dir), spec, str(venv_dir))
        if not bool(probe.get("entrypoints_ok")):
            raise RuntimeError("entrypoints validation failed after install")
        runtime_warning = None
        if not bool(probe.get("runtime_ok", True)):
            runtime_warning = str(probe.get("runtime_reason", "") or "runtime validation failed")

        upsert_provider_record(
            {
                "provider_id": spec.id,
                "display_name": spec.display_name,
                "repo_path": str(effective_repo_dir),
                "venv_path": str(venv_dir),
                "install_root": str(root),
                "install_state": "installed",
                "detected_capabilities": {"train": True, "infer": True},
                "repo_ref": {"remote_url": spec.repo_url, "branch": spec.branch, "commit": _git_rev(repo_dir)},
                "installed_at": _now(),
                "last_validated_at": _now(),
                "last_error": runtime_warning,
            }
        )
        write_provider_state(
            str(root),
            spec.id,
            {"provider_id": spec.id, "repo_action": repo_action, "env_action": env_action, "status": "installed"},
        )
        msg = f"{repo_action}, {env_action}"
        if runtime_warning:
            msg = f"{msg}; runtime warning: {runtime_warning}"
        return InstallResult(provider_id=spec.id, action="installed", ok=True, message=msg)
    except Exception as e:
        mark_provider_state(spec.id, "failed", last_error=str(e))
        return InstallResult(provider_id=spec.id, action="failed", ok=False, message=str(e))


def uninstall_provider(provider_id: str, target_dir: str | None = None) -> InstallResult:
    spec = get_provider_spec(provider_id)
    root = providers_root(target_dir)
    repo_dir = root / spec.id
    if not repo_dir.exists():
        mark_provider_state(spec.id, "removed", last_error=None)
        return InstallResult(provider_id=spec.id, action="skipped", ok=True, message="not installed")
    try:
        shutil.rmtree(repo_dir)
        mark_provider_state(spec.id, "removed", last_error=None)
        return InstallResult(provider_id=spec.id, action="removed", ok=True, message="deleted")
    except Exception as e:
        mark_provider_state(spec.id, "failed", last_error=str(e))
        return InstallResult(provider_id=spec.id, action="failed", ok=False, message=str(e))


def _venv_python(venv_dir: Path) -> str:
    if os.name == "nt":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")


def _run(cmd: list[str], *, cwd: str, timeout_sec: int | None = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, timeout=timeout_sec)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _install_provider_runtime_deps(provider_id: str, repo_dir: Path, python_bin: str) -> None:
    common_pip_deps: dict[str, list[str]] = {
        "dr-yolo": ["ultralytics"],
        "leaf-yolo": ["ultralytics", "timm"],
        "mp-yolo": ["ultralytics"],
        "ssdm-yolo": ["ultralytics", "tqdm"],
        "enhanced-yolov8": ["ultralytics", "timm"],
    }
    extras = common_pip_deps.get(provider_id, [])
    if extras:
        _run([python_bin, "-m", "pip", "install", *extras], cwd=str(repo_dir))

    if provider_id != "mfel-yolo":
        return
    # Best-effort extras for MFEL custom ops/runtime.
    _run([python_bin, "-m", "pip", "install", "ninja", "einops", "PyWavelets"], cwd=str(repo_dir))
    # Try wheel/build from index first (faster than source build path).
    try:
        _run([python_bin, "-m", "pip", "install", "--no-build-isolation", "DCNv4"], cwd=str(repo_dir), timeout_sec=120)
        return
    except Exception:
        pass
    candidates = [repo_dir / "DCNv4"]
    candidates.extend(p for p in repo_dir.glob("**/DCNv4") if p.is_dir())
    # If provider repo does not ship DCNv4, auto-fetch official source.
    managed_root = repo_dir / ".smartrain_deps"
    managed_root.mkdir(parents=True, exist_ok=True)
    managed_dcn = managed_root / "DCNv4"
    if not managed_dcn.is_dir():
        _run(["git", "clone", "--depth", "1", "https://github.com/OpenGVLab/DCNv4.git", str(managed_dcn)], cwd=str(managed_root))
    candidates.append(managed_dcn)
    candidates.extend(p for p in managed_dcn.glob("**/DCNv4_op") if p.is_dir())
    seen: set[str] = set()
    for c in candidates:
        key = str(c.resolve())
        if key in seen:
            continue
        seen.add(key)
        if not c.is_dir():
            continue
        if (c / "setup.py").is_file() or (c / "pyproject.toml").is_file():
            try:
                _run([python_bin, "-m", "pip", "install", "--no-build-isolation", str(c)], cwd=str(repo_dir), timeout_sec=300)
                return
            except Exception:
                continue
    _ensure_mfel_dcnv4_shim(repo_dir)


def _ensure_mfel_dcnv4_shim(repo_dir: Path) -> None:
    shim = repo_dir / "DCNv4.py"
    if shim.is_file():
        return
    shim.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import torch.nn as nn",
                "",
                "class DCNv4(nn.Module):",
                "    \"\"\"Fallback shim when native DCNv4 is unavailable.\"\"\"",
                "    def __init__(self, *args, **kwargs):",
                "        super().__init__()",
                "",
                "    def forward(self, x, *args, **kwargs):",
                "        return x",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _git_rev(repo_dir: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
        return out or None
    except Exception:
        return None


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _resolve_effective_repo_dir(provider_id: str, repo_dir: Path) -> Path:
    if provider_id == "ssdm-yolo":
        return _resolve_ssdm_repo_root(repo_dir)
    if provider_id == "enhanced-yolov8":
        return _find_entrypoint_root(repo_dir) or repo_dir
    return repo_dir


def _resolve_ssdm_repo_root(repo_dir: Path) -> Path:
    # Already extracted or regular layout.
    direct = _find_entrypoint_root(repo_dir)
    if direct is not None:
        return direct

    # SSDM repository currently ships an archive in root.
    zip_candidates = sorted(repo_dir.glob("*.zip"))
    if not zip_candidates:
        return repo_dir
    preferred = next((z for z in zip_candidates if z.name.lower() == "ssdm-yolo.zip"), zip_candidates[0])

    extract_root = repo_dir / "_smartrain_unpacked"
    marker = extract_root / ".source_zip_mtime"
    zip_mtime = str(int(preferred.stat().st_mtime))
    needs_unpack = True
    if extract_root.is_dir() and marker.is_file():
        try:
            needs_unpack = marker.read_text(encoding="utf-8").strip() != zip_mtime
        except Exception:
            needs_unpack = True
    if needs_unpack:
        if extract_root.exists():
            shutil.rmtree(extract_root, ignore_errors=True)
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(preferred, "r") as zf:
            zf.extractall(str(extract_root))
        marker.write_text(zip_mtime, encoding="utf-8")

    extracted = _find_entrypoint_root(extract_root)
    return extracted or repo_dir


def _find_entrypoint_root(base_dir: Path) -> Path | None:
    # Prefer shallow paths first.
    candidates = [base_dir, *[p for p in base_dir.iterdir() if p.is_dir()]] if base_dir.is_dir() else []
    for c in candidates:
        if (c / "train.py").is_file() and (c / "detect.py").is_file():
            return c
    if not base_dir.is_dir():
        return None
    for train_file in base_dir.rglob("train.py"):
        root = train_file.parent
        if (root / "detect.py").is_file():
            return root
    return None

