"""
Профиль обучения: YAML с ключами для Ultralytics YOLO.train() + переопределения CLI.
Ключи smartrain (export/clearml/weighted) не передаются в ultralytics YOLO.train().
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Ключи, обрабатываемые smart-train, не уходят в model.train()
SMARTRAIN_TRAIN_KEYS = frozenset(
    {
        "export_onnx",
        "export_onnx_half",
        "export_onnx_simplify",
        "export_onnx_opset",
        "export_onnx_dynamic",
        "weighted_sampling",
        "clearml",
        "clearml_project",
    }
)


def load_train_profile(path: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Файл профиля обучения не найден: {path}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: ожидается YAML-объект (словарь).")
    return raw


def task_to_metadata_task_type(task: str | None) -> str:
    t = (task or "detect").strip().lower()
    if t == "segment":
        return "segmentation"
    if t in ("classify", "classification"):
        return "classification"
    if t == "pose":
        return "pose"
    if t == "obb":
        return "oriented_detection"
    return "detection"


def extract_smartrain_options(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Удаляет из копии cfg ключи smartrain; возвращает (ultralytics_cfg, options)."""
    u = copy.deepcopy(cfg)
    opts: dict[str, Any] = {}
    for k in SMARTRAIN_TRAIN_KEYS:
        if k in u:
            opts[k] = u.pop(k)
    return u, opts


def merge_cli_into_ultralytics_cfg(
    u_cfg: dict[str, Any],
    *,
    model: str | None,
    epochs: int | None,
    batch: int | None,
    imgsz: int | None,
    task: str | None,
    defaults: dict[str, Any],
) -> None:
    """
    CLI переопределяет профиль только для явно переданных аргументов;
    иначе берётся YAML, иначе defaults (model/epochs/batch/imgsz/task).
    """
    if model is not None:
        u_cfg["model"] = model
    elif "model" not in u_cfg:
        u_cfg["model"] = defaults["model"]
    if epochs is not None:
        u_cfg["epochs"] = epochs
    elif "epochs" not in u_cfg:
        u_cfg["epochs"] = defaults["epochs"]
    if batch is not None:
        u_cfg["batch"] = batch
    elif "batch" not in u_cfg:
        u_cfg["batch"] = defaults["batch"]
    if imgsz is not None:
        u_cfg["imgsz"] = imgsz
    elif "imgsz" not in u_cfg:
        u_cfg["imgsz"] = defaults["imgsz"]
    if task is not None:
        u_cfg["task"] = task
    elif "task" not in u_cfg:
        u_cfg["task"] = defaults.get("task", "detect")


def apply_cli_smartrain_overrides(opts: dict[str, Any], args: Any) -> None:
    """Флаги CLI переопределяют опции из YAML (мутация opts)."""
    if getattr(args, "export_onnx", False):
        opts["export_onnx"] = True
    if getattr(args, "weighted_sampling", False):
        opts["weighted_sampling"] = True
    if getattr(args, "clearml", False):
        opts["clearml"] = True
    cp = getattr(args, "clearml_project", None)
    if cp:
        opts["clearml_project"] = cp


def resolve_profile_data_path(data_field: str) -> str:
    """
    Поле data из YAML: путь к data.yaml или к каталогу датасета.
    """
    expanded = str(Path(data_field).expanduser())
    if expanded.lower().endswith(".yaml") or expanded.lower().endswith(".yml"):
        p = Path(expanded).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"data из профиля: файл не найден: {p}")
        return str(p)
    root = Path(expanded).resolve()
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data из профиля: нет {yaml_path}")
    return str(yaml_path)


def dataset_root_from_data_yaml(data_yaml_path: str) -> str:
    """Каталог датасета (родитель data.yaml) для хеша и имени."""
    return str(Path(data_yaml_path).resolve().parent)
