from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from smartrain.external_providers.adapters import build_external_infer_spec, build_external_train_spec
from smartrain.external_providers.probe import external_python_in_env


def run_external_train(
    provider_id: str,
    repo_path: str,
    venv_path: str,
    *,
    dataset_path: str,
    model: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str | None = None,
    target_dir: str | None = None,
    run_name: str | None = None,
) -> int:
    spec = build_external_train_spec(
        provider_id,
        repo_path,
        dataset_path=dataset_path,
        model=model,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        target_dir=target_dir,
        run_name=run_name,
    )
    return _run_python_script(spec.script_path, repo_path, venv_path, spec.args, spec.env_overrides)


def run_external_infer(
    provider_id: str,
    repo_path: str,
    venv_path: str,
    *,
    model_path: str,
    source_path: str,
    conf: float,
    imgsz: int,
    device: str | None = None,
    target_dir: str | None = None,
    run_name: str | None = None,
    task_type: str | None = None,
) -> int | dict[str, object]:
    result_json_path = str(Path(tempfile.mkdtemp(prefix="smartrain-ext-infer-")) / "inference_result.json")
    spec = build_external_infer_spec(
        provider_id,
        repo_path,
        model_path=model_path,
        source_path=source_path,
        conf=conf,
        imgsz=imgsz,
        device=device,
        target_dir=target_dir,
        run_name=run_name,
        result_json=result_json_path,
        task_type=task_type,
    )
    rc = _run_python_script(spec.script_path, repo_path, venv_path, spec.args, spec.env_overrides)
    payload = _try_load_structured_infer_result(result_json_path, return_code=rc)
    return payload if payload is not None else rc


def _run_python_script(
    script_path: str,
    cwd: str,
    venv_path: str,
    args: list[str],
    env_overrides: dict[str, str] | None = None,
) -> int:
    python_bin = external_python_in_env(venv_path)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if env_overrides:
        env.update(env_overrides)
    cmd = [python_bin, script_path, *list(args or [])]
    proc = subprocess.run(cmd, cwd=str(Path(cwd).expanduser().resolve()), env=env)
    return int(proc.returncode)


def _try_load_structured_infer_result(path: str, *, return_code: int) -> dict[str, object] | None:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {
            "return_code": int(return_code),
            "images": [],
            "diagnostics": {"structured_result_status": "invalid_json"},
        }
    if not isinstance(raw, dict):
        return {
            "return_code": int(return_code),
            "images": [],
            "diagnostics": {"structured_result_status": "invalid_payload_type"},
        }
    images = raw.get("images")
    normalized_images = images if isinstance(images, list) else []
    rc_raw = raw.get("return_code", return_code)
    try:
        rc = int(rc_raw)
    except Exception:
        rc = int(return_code)
    out: dict[str, object] = {"return_code": rc, "images": normalized_images}
    diagnostics = raw.get("diagnostics")
    if isinstance(diagnostics, dict):
        out["diagnostics"] = diagnostics
    return out

