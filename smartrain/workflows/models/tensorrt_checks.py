from __future__ import annotations

import ctypes
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TrtexecCapabilities:
    supports_build_only: bool
    supports_explicit_batch: bool
    workspace_mode: Literal["workspace", "memPoolSize", "none"]


_TRTEXEC_RUNTIME_CACHE: tuple[bool, str] | None = None
_TRTEXEC_CAPS_CACHE: tuple[str, TrtexecCapabilities] | None = None


def resolve_cuda_runtime_module():
    try:
        from cuda import cudart  # type: ignore

        return cudart, "cuda.cudart"
    except Exception as old_exc:
        try:
            from cuda.bindings import runtime as cudart  # type: ignore

            return cudart, "cuda.bindings.runtime"
        except Exception as new_exc:
            raise RuntimeError(
                "python CUDA runtime is unavailable. Tried `from cuda import cudart` "
                "and `from cuda.bindings import runtime`. "
                f"Legacy API error: {old_exc}; new API error: {new_exc}"
            ) from new_exc


def resolve_trtexec_bin() -> str | None:
    candidate = shutil.which("trtexec")
    if candidate:
        return candidate
    fallback = "/usr/src/tensorrt/bin/trtexec"
    if Path(fallback).exists():
        return fallback
    return None


def check_tensorrt_ready() -> tuple[bool, str]:
    reasons: list[str] = []
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            reasons.append("CUDA GPU is not available")
    except Exception:
        reasons.append("PyTorch CUDA check failed")
    try:
        import tensorrt  # type: ignore # noqa: F401
    except Exception:
        trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
        if not Path(trtexec).exists():
            reasons.append("python package 'tensorrt' is not installed and trtexec is not found")
    if reasons:
        return False, "; ".join(reasons)
    return True, ""


def check_python_cuda_runtime_ready() -> tuple[bool, str]:
    try:
        _cudart, import_path = resolve_cuda_runtime_module()
    except Exception as e:
        return (
            False,
            "python CUDA runtime is unavailable. "
            f"Install CUDA Python bindings and verify runtime libraries: {e}",
        )
    _ = _cudart
    _ = import_path
    return True, ""


def detect_trtexec_capabilities(trtexec_bin: str) -> TrtexecCapabilities:
    help_text = ""
    try:
        proc = subprocess.run([trtexec_bin, "--help"], check=False, text=True, capture_output=True)
        help_text = f"{proc.stdout}\n{proc.stderr}"
    except Exception:
        help_text = ""
    help_lower = help_text.lower()
    has_help = bool(help_lower.strip())
    supports_build_only = "--buildonly" in help_lower if has_help else True
    supports_explicit_batch = "--explicitbatch" in help_lower if has_help else True
    if "--mempoolsize" in help_lower:
        workspace_mode: Literal["workspace", "memPoolSize", "none"] = "memPoolSize"
    elif "--workspace" in help_lower:
        workspace_mode = "workspace"
    elif has_help:
        workspace_mode = "none"
    else:
        workspace_mode = "workspace"
    return TrtexecCapabilities(
        supports_build_only=supports_build_only,
        supports_explicit_batch=supports_explicit_batch,
        workspace_mode=workspace_mode,
    )


def get_trtexec_capabilities(trtexec_bin: str) -> TrtexecCapabilities:
    global _TRTEXEC_CAPS_CACHE
    if _TRTEXEC_CAPS_CACHE is not None and _TRTEXEC_CAPS_CACHE[0] == trtexec_bin:
        return _TRTEXEC_CAPS_CACHE[1]
    caps = detect_trtexec_capabilities(trtexec_bin)
    _TRTEXEC_CAPS_CACHE = (trtexec_bin, caps)
    return caps


def append_trtexec_workspace_arg(cmd: list[str], workspace_mib: int, caps: TrtexecCapabilities) -> None:
    if caps.workspace_mode == "workspace":
        cmd.append(f"--workspace={workspace_mib}")
    elif caps.workspace_mode == "memPoolSize":
        cmd.append(f"--memPoolSize=workspace:{workspace_mib}")


def check_trtexec_dependencies() -> tuple[bool, str]:
    trtexec_bin = resolve_trtexec_bin()
    if not trtexec_bin:
        return False, "trtexec binary is not found"
    system_name = platform.system().lower()
    if system_name == "linux":
        try:
            proc = subprocess.run(["ldd", trtexec_bin], check=False, text=True, capture_output=True)
            output = f"{proc.stdout}\n{proc.stderr}"
            missing = [line.strip() for line in output.splitlines() if "not found" in line]
            if missing:
                return False, f"missing shared libs: {'; '.join(missing[:3])}"
        except Exception as e:
            return False, f"ldd check failed: {e}"
        try:
            ctypes.CDLL("libcublasLt.so.12")
        except OSError:
            return False, "libcublasLt.so.12 is not found"
    elif system_name == "windows":
        try:
            ctypes.WinDLL("cublasLt64_12.dll")
        except Exception:
            return False, "cublasLt64_12.dll is not found"
    return True, ""


def check_trtexec_gpu_ready() -> tuple[bool, str]:
    trtexec_bin = resolve_trtexec_bin()
    if not trtexec_bin:
        return False, "trtexec binary is not found"
    if platform.system().lower() == "windows":
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                return False, "PyTorch CUDA check failed (cuda is unavailable)"
        except Exception as e:
            return False, f"PyTorch CUDA check failed: {e}"
        return True, ""
    try:
        probe = subprocess.run(["nvidia-smi", "-L"], check=False, text=True, capture_output=True)
    except Exception as e:
        return False, f"nvidia-smi probe failed: {e}"
    if probe.returncode != 0:
        details = (probe.stderr or probe.stdout or "").strip()
        return False, f"nvidia-smi probe failed: {details or f'exit={probe.returncode}'}"
    out = (probe.stdout or "").strip()
    if "GPU " not in out:
        return False, "nvidia-smi did not report any GPUs"
    return True, ""


def check_trtexec_runtime_ready() -> tuple[bool, str]:
    global _TRTEXEC_RUNTIME_CACHE
    if _TRTEXEC_RUNTIME_CACHE is not None:
        return _TRTEXEC_RUNTIME_CACHE
    trtexec_bin = resolve_trtexec_bin()
    if not trtexec_bin:
        _TRTEXEC_RUNTIME_CACHE = (False, "trtexec binary is not found")
        return _TRTEXEC_RUNTIME_CACHE
    caps = get_trtexec_capabilities(trtexec_bin)
    try:
        import onnx  # type: ignore
        from onnx import TensorProto, helper  # type: ignore
    except Exception as e:
        _TRTEXEC_RUNTIME_CACHE = (False, f"onnx probe dependencies are unavailable: {e}")
        return _TRTEXEC_RUNTIME_CACHE
    probe_dir = Path(tempfile.mkdtemp(prefix="smartrain_trt_probe_"))
    onnx_path = probe_dir / "probe.onnx"
    engine_path = probe_dir / "probe.trt"
    try:
        x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])
        y = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node("Identity", inputs=["images"], outputs=["out"])
        graph = helper.make_graph([node], "smartrain_probe", [x], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
        onnx.save(model, str(onnx_path))
        base_cmd = [
            trtexec_bin,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
        ]
        attempts: list[tuple[str, list[str]]] = []
        cmd_primary = list(base_cmd)
        if caps.supports_build_only:
            cmd_primary.append("--buildOnly")
        append_trtexec_workspace_arg(cmd_primary, 64, caps)
        attempts.append(("primary", cmd_primary))
        attempts.append(("fallback_no_optional_flags", list(base_cmd)))

        unique_attempts: list[tuple[str, list[str]]] = []
        seen: set[tuple[str, ...]] = set()
        for name, cmd in attempts:
            key = tuple(cmd)
            if key in seen:
                continue
            seen.add(key)
            unique_attempts.append((name, cmd))

        reasons: list[str] = []
        for attempt_name, cmd in unique_attempts:
            if engine_path.exists():
                engine_path.unlink(missing_ok=True)
            proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
            if proc.returncode == 0 and engine_path.exists():
                _TRTEXEC_RUNTIME_CACHE = (True, "")
                return _TRTEXEC_RUNTIME_CACHE
            err = (proc.stderr or proc.stdout or "").strip()
            short = err.splitlines()[-1] if err else f"exit={proc.returncode}"
            reasons.append(f"{attempt_name}: {short}")

        details = "; ".join(reasons[:2]) if reasons else "engine is not produced"
        _TRTEXEC_RUNTIME_CACHE = (False, f"trtexec runtime probe failed: {details}")
        return _TRTEXEC_RUNTIME_CACHE
    except Exception as e:
        _TRTEXEC_RUNTIME_CACHE = (False, f"trtexec runtime probe failed: {e}")
        return _TRTEXEC_RUNTIME_CACHE
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)

