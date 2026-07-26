from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from smartrain.core.runtime.path_portable import is_abs_like, resolve_stored_path_under_workspace
from smartrain.core.runtime.run_discovery import find_run_directories
from smartrain.core.runtime.workspace_paths import WorkspaceLayout
from smartrain.core.workflow_adapters.inference_runtime_api import resolve_dataset_root_for_entry
from smartrain.core.workflow_adapters.training_runtime_api import resolve_dataset_path_for_resume
from smartrain.services.inference_runtime_helpers import _resolve_run_ref, load_catalog, resolve_model
from smartrain.services.visualization.contracts import VisRequest

_DATA_YAML_META_KEYS = {"path", "nc", "names", "download", "yaml_file"}


def _normalize_source_run_ref(layout: WorkspaceLayout, run_ref: str) -> str:
    """Dual-read abs or workspace-relative ``source_run`` / ``source_run_relative``."""
    s = (run_ref or "").strip()
    if not s or s.isdigit():
        return s
    if not is_abs_like(s):
        return resolve_stored_path_under_workspace(layout.root, s)
    p = Path(s)
    if p.exists():
        return str(p.expanduser().resolve())
    try:
        return resolve_stored_path_under_workspace(layout.root, s)
    except ValueError:
        return str(p.expanduser().resolve())


def _pick_source_run_ref(layout: WorkspaceLayout, *candidates: Any) -> str | None:
    for val in candidates:
        if isinstance(val, str) and val.strip():
            return _normalize_source_run_ref(layout, val.strip())
    return None


def _load_data_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid data.yaml: {path}")
    return payload


def _data_yaml_split_dirs(data_yaml_path: Path, requested: tuple[str, ...] | None) -> dict[str, Path]:
    cfg = _load_data_yaml(data_yaml_path)
    root = data_yaml_path.parent
    out: dict[str, Path] = {}
    for key, raw in cfg.items():
        if key in _DATA_YAML_META_KEYS:
            continue
        if not isinstance(raw, str) or not raw.strip():
            continue
        split_dir = (root / raw.strip()).resolve()
        if split_dir.is_dir():
            out[str(key)] = split_dir
    if requested:
        filtered = {k: v for k, v in out.items() if k in set(requested)}
        return filtered
    return out


def _class_names_from_yaml(data_yaml_path: Path) -> dict[int, str]:
    cfg = _load_data_yaml(data_yaml_path)
    names = cfg.get("names")
    if isinstance(names, dict):
        out: dict[int, str] = {}
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def resolve_dataset_target(layout: WorkspaceLayout, req: VisRequest) -> dict[str, Any]:
    if not req.dataset:
        raise ValueError("Dataset is required for vis dataset.")
    candidate = Path(str(req.dataset)).expanduser()
    if candidate.is_dir():
        dataset_root = candidate.resolve()
        dataset_name = candidate.name
    else:
        catalog = load_catalog(layout)
        if req.dataset not in catalog:
            raise KeyError(f"Dataset {req.dataset!r} not found in catalog.")
        dataset_root = resolve_dataset_root_for_entry(
            dataset_name=str(req.dataset),
            info=catalog[str(req.dataset)],
            workspace_root=layout.root,
            source_catalog_dir=layout.datasets,
            legacy_source_parent=layout.datasets,
        )
        dataset_name = str(req.dataset)
    data_yaml_path = (Path(dataset_root) / "data.yaml").resolve()
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml not found in dataset root: {dataset_root}")
    split_dirs = _data_yaml_split_dirs(data_yaml_path, req.splits)
    if not split_dirs:
        raise RuntimeError(f"No split directories found in {data_yaml_path}")
    class_names = _class_names_from_yaml(data_yaml_path)
    return {
        "dataset_name": dataset_name,
        "dataset_root": Path(dataset_root).resolve(),
        "data_yaml": data_yaml_path,
        "split_dirs": split_dirs,
        "class_names": class_names,
    }


def resolve_run_target(layout: WorkspaceLayout, req: VisRequest) -> dict[str, Any]:
    if not req.run_ref:
        raise ValueError("Run reference is required for vis run.")
    run_dir = _resolve_run_ref(layout, req.run_ref)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    dataset_dir = resolve_dataset_path_for_resume(str(run_dir), layout.root)
    if not dataset_dir:
        raise RuntimeError(f"Cannot resolve dataset path from run: {run_dir}")
    data_yaml_path = (Path(dataset_dir) / "data.yaml").resolve()
    split_dirs = _data_yaml_split_dirs(data_yaml_path, req.splits)
    if not split_dirs:
        raise RuntimeError(f"No split directories found for run dataset: {data_yaml_path}")
    return {
        "run_dir": run_dir.resolve(),
        "dataset_root": Path(dataset_dir).resolve(),
        "data_yaml": data_yaml_path,
        "split_dirs": split_dirs,
        "class_names": _class_names_from_yaml(data_yaml_path),
    }


def resolve_model_target(layout: WorkspaceLayout, req: VisRequest) -> dict[str, Any]:
    if not req.model_name and not req.weights:
        raise ValueError("Model target requires --model-name or --weights.")
    args = type(
        "ModelArgs",
        (),
        {"model_name": req.model_name, "run": None, "weights": req.weights},
    )()
    model_path, model_id, model_source = resolve_model(args, layout)
    model_dir = model_path.parent.resolve()
    run_ref = req.run_ref
    if not run_ref:
        manifest_path = model_dir / "model_manifest.json"
        if manifest_path.is_file():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            run_ref = _pick_source_run_ref(
                layout,
                payload.get("source_run_relative"),
                payload.get("source_run"),
            )
    if not run_ref:
        # Release sidecar next to weights: <stem>.json with source.source_run.
        sidecar = model_path.with_suffix(".json")
        if not sidecar.is_file() and model_path.suffix.lower() != ".pt":
            # e.g. .onnx next to detect_*.json
            for cand in model_dir.glob(f"{model_path.stem.split('.')[0]}*.json"):
                if cand.name.endswith(".meta.json"):
                    continue
                sidecar = cand
                break
        if sidecar.is_file():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                meta = None
            if isinstance(meta, dict):
                source = meta.get("source") if isinstance(meta.get("source"), dict) else {}
                run_ref = _pick_source_run_ref(
                    layout,
                    source.get("source_run_relative") if isinstance(source, dict) else None,
                    source.get("source_run") if isinstance(source, dict) else None,
                )
    if not run_ref:
        runs = find_run_directories(layout.runs)
        if runs:
            run_ref = runs[-1]
    if not run_ref:
        raise RuntimeError("Cannot resolve source run for model target. Pass --run explicitly.")
    run_req = VisRequest(
        mode="run",
        workspace_root=req.workspace_root,
        dataset=None,
        model_name=None,
        run_ref=str(run_ref),
        weights=None,
        splits=req.splits,
        limit=req.limit,
        conf=req.conf,
        device=req.device,
        overwrite=req.overwrite,
        non_interactive=req.non_interactive,
    )
    run_target = resolve_run_target(layout, run_req)
    run_target.update(
        {
            "model_path": model_path.resolve(),
            "model_id": model_id,
            "model_source": model_source,
            "model_dir": model_dir,
        }
    )
    return run_target

