from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExternalRunSpec:
    script_path: str
    args: list[str]
    env_overrides: dict[str, str]


def _external_project_dir(target_dir: str | None, dataset_path: str) -> str | None:
    if not target_dir:
        return None
    ds_name = Path(dataset_path).resolve().name
    return str(Path(target_dir).expanduser().resolve() / ds_name)


def build_external_train_spec(
    provider_id: str,
    repo_path: str,
    *,
    dataset_path: str,
    model: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str | None = None,
    target_dir: str | None = None,
    run_name: str | None = None,
    task_type: str | None = None,
) -> ExternalRunSpec:
    repo = Path(repo_path).expanduser().resolve()
    pid = provider_id.strip().lower()
    project_dir = _external_project_dir(target_dir, dataset_path)
    task_args = ["--task", str(task_type or "detection")]
    if pid in ("dr-yolo", "leaf-yolo"):
        launcher = Path(__file__).resolve().parent / "launchers" / "mp_train_launcher.py"
        args = [
            "--repo",
            str(repo),
            "--data",
            str(Path(dataset_path).resolve() / "data.yaml"),
            "--model",
            model,
            "--epochs",
            str(epochs),
            "--batch",
            str(batch),
            "--imgsz",
            str(imgsz),
        ]
        if project_dir:
            args += ["--project", project_dir]
        if run_name:
            args += ["--name", str(run_name)]
        if device:
            args += ["--device", str(device)]
        args += task_args
        return ExternalRunSpec(script_path=str(launcher), args=args, env_overrides={})

    if pid == "ssdm-yolo":
        launcher = Path(__file__).resolve().parent / "launchers" / "mp_train_launcher.py"
        args = [
            "--repo",
            str(repo),
            "--data",
            str(Path(dataset_path).resolve() / "data.yaml"),
            "--model",
            model,
            "--epochs",
            str(epochs),
            "--batch",
            str(batch),
            "--imgsz",
            str(imgsz),
        ]
        if project_dir:
            args += ["--project", project_dir]
        if run_name:
            args += ["--name", str(run_name)]
        if device:
            args += ["--device", str(device)]
        args += task_args
        return ExternalRunSpec(script_path=str(launcher), args=args, env_overrides={})

    if pid == "enhanced-yolov8":
        launcher = Path(__file__).resolve().parent / "launchers" / "mp_train_launcher.py"
        args = [
            "--repo",
            str(repo),
            "--data",
            str(Path(dataset_path).resolve() / "data.yaml"),
            "--model",
            model,
            "--epochs",
            str(epochs),
            "--batch",
            str(batch),
            "--imgsz",
            str(imgsz),
        ]
        if project_dir:
            args += ["--project", project_dir]
        if run_name:
            args += ["--name", str(run_name)]
        if device:
            args += ["--device", str(device)]
        args += task_args
        return ExternalRunSpec(script_path=str(launcher), args=args, env_overrides={})

    if pid == "mfel-yolo":
        launcher = Path(__file__).resolve().parent / "launchers" / "mfel_train_launcher.py"
        args = [
            "--repo",
            str(repo),
            "--data",
            str(Path(dataset_path).resolve() / "data.yaml"),
            "--model",
            model,
            "--epochs",
            str(epochs),
            "--batch",
            str(batch),
            "--imgsz",
            str(imgsz),
        ]
        if project_dir:
            args += ["--project", project_dir]
        if run_name:
            args += ["--name", str(run_name)]
        if device:
            args += ["--device", str(device)]
        args += task_args
        return ExternalRunSpec(script_path=str(launcher), args=args, env_overrides={})

    if pid == "mp-yolo":
        launcher = Path(__file__).resolve().parent / "launchers" / "mp_train_launcher.py"
        args = [
            "--repo",
            str(repo),
            "--data",
            str(Path(dataset_path).resolve() / "data.yaml"),
            "--model",
            model,
            "--epochs",
            str(epochs),
            "--batch",
            str(batch),
            "--imgsz",
            str(imgsz),
        ]
        if project_dir:
            args += ["--project", project_dir]
        if run_name:
            args += ["--name", str(run_name)]
        if device:
            args += ["--device", str(device)]
        args += task_args
        return ExternalRunSpec(script_path=str(launcher), args=args, env_overrides={})

    raise ValueError(f"Unsupported provider for train adapter: {provider_id}")


def build_external_infer_spec(
    provider_id: str,
    repo_path: str,
    *,
    model_path: str,
    source_path: str,
    conf: float,
    imgsz: int,
    device: str | None = None,
    target_dir: str | None = None,
    run_name: str | None = None,
    result_json: str | None = None,
    task_type: str | None = None,
) -> ExternalRunSpec:
    repo = Path(repo_path).expanduser().resolve()
    pid = provider_id.strip().lower()
    if pid in ("dr-yolo", "leaf-yolo"):
        launcher = Path(__file__).resolve().parent / "launchers" / "mp_infer_launcher.py"
        args = [
            "--repo",
            str(repo),
            "--model",
            model_path,
            "--source",
            source_path,
            "--conf",
            str(conf),
            "--imgsz",
            str(imgsz),
        ]
        if device:
            args += ["--device", str(device)]
        if target_dir:
            args += ["--project", str(target_dir)]
        if run_name:
            args += ["--name", str(run_name)]
        if result_json:
            args += ["--result-json", str(result_json)]
        if task_type:
            args += ["--task", str(task_type)]
        return ExternalRunSpec(script_path=str(launcher), args=args, env_overrides={})

    if pid == "ssdm-yolo":
        launcher = Path(__file__).resolve().parent / "launchers" / "mp_infer_launcher.py"
        args = [
            "--repo",
            str(repo),
            "--model",
            model_path,
            "--source",
            source_path,
            "--conf",
            str(conf),
            "--imgsz",
            str(imgsz),
        ]
        if device:
            args += ["--device", str(device)]
        if target_dir:
            args += ["--project", str(target_dir)]
        if run_name:
            args += ["--name", str(run_name)]
        if result_json:
            args += ["--result-json", str(result_json)]
        if task_type:
            args += ["--task", str(task_type)]
        return ExternalRunSpec(script_path=str(launcher), args=args, env_overrides={})

    if pid == "enhanced-yolov8":
        launcher = Path(__file__).resolve().parent / "launchers" / "mp_infer_launcher.py"
        args = [
            "--repo",
            str(repo),
            "--model",
            model_path,
            "--source",
            source_path,
            "--conf",
            str(conf),
            "--imgsz",
            str(imgsz),
        ]
        if device:
            args += ["--device", str(device)]
        if target_dir:
            args += ["--project", str(target_dir)]
        if run_name:
            args += ["--name", str(run_name)]
        if result_json:
            args += ["--result-json", str(result_json)]
        if task_type:
            args += ["--task", str(task_type)]
        return ExternalRunSpec(script_path=str(launcher), args=args, env_overrides={})

    if pid == "mfel-yolo":
        launcher = Path(__file__).resolve().parent / "launchers" / "mfel_infer_launcher.py"
        args = [
            "--repo",
            str(repo),
            "--model",
            model_path,
            "--source",
            source_path,
            "--conf",
            str(conf),
            "--imgsz",
            str(imgsz),
        ]
        if device:
            args += ["--device", str(device)]
        if target_dir:
            args += ["--project", str(target_dir)]
        if run_name:
            args += ["--name", str(run_name)]
        if result_json:
            args += ["--result-json", str(result_json)]
        if task_type:
            args += ["--task", str(task_type)]
        return ExternalRunSpec(script_path=str(launcher), args=args, env_overrides={})

    if pid == "mp-yolo":
        launcher = Path(__file__).resolve().parent / "launchers" / "mp_infer_launcher.py"
        args = [
            "--repo",
            str(repo),
            "--model",
            model_path,
            "--source",
            source_path,
            "--conf",
            str(conf),
            "--imgsz",
            str(imgsz),
        ]
        if device:
            args += ["--device", str(device)]
        if target_dir:
            args += ["--project", str(target_dir)]
        if run_name:
            args += ["--name", str(run_name)]
        if result_json:
            args += ["--result-json", str(result_json)]
        if task_type:
            args += ["--task", str(task_type)]
        return ExternalRunSpec(script_path=str(launcher), args=args, env_overrides={})

    raise ValueError(f"Unsupported provider for infer adapter: {provider_id}")

