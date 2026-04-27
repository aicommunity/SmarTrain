from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _extract_img_size_from_obj(obj: Any) -> int | None:
    if isinstance(obj, dict):
        training_info = obj.get("training_info")
        if isinstance(training_info, dict):
            hyperparameters = training_info.get("hyperparameters")
            if isinstance(hyperparameters, dict):
                for key in ("image_size", "imgsz", "img_size"):
                    value = hyperparameters.get(key)
                    if isinstance(value, (int, float)) and int(value) > 0:
                        return int(value)
        inference = obj.get("inference")
        if isinstance(inference, dict):
            value = inference.get("imgsz")
            if isinstance(value, (int, float)) and int(value) > 0:
                return int(value)
        train_summary = obj.get("ultralytics_train_summary")
        if isinstance(train_summary, dict):
            for key in ("imgsz", "img_size", "image_size"):
                value = train_summary.get(key)
                if isinstance(value, (int, float)) and int(value) > 0:
                    return int(value)
        for key in ("imgsz", "img_size", "image_size"):
            value = obj.get(key)
            if isinstance(value, (int, float)) and int(value) > 0:
                return int(value)
        for value in obj.values():
            found = _extract_img_size_from_obj(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _extract_img_size_from_obj(value)
            if found is not None:
                return found
    return None


def _read_img_size_from_meta_file(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except Exception:
        return None
    return _extract_img_size_from_obj(payload)


def _collect_context_candidates(model_path: Path) -> list[tuple[str, Path]]:
    mp = model_path.resolve()
    candidates: list[tuple[str, Path]] = []

    parts = [part.lower() for part in mp.parts]
    if "weights" in parts:
        try:
            idx = parts.index("weights")
            run_dir = Path(*mp.parts[:idx]).resolve().parent if idx >= 1 else None
        except Exception:
            run_dir = None
        if run_dir and run_dir.is_dir():
            candidates.extend(
                [
                    ("training_metadata", run_dir / "training_metadata.json"),
                    ("args_yaml", run_dir / "train" / "args.yaml"),
                    ("args_yaml", run_dir / "train" / "args.yml"),
                    ("runtime_yaml", run_dir / "_runtime_data_test.yaml"),
                ]
            )

    model_dir = mp.parent
    if model_dir.is_dir():
        for p in sorted(model_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in {".json", ".yaml", ".yml"}:
                source = "args_yaml" if p.name in {"args.yaml", "args.yml"} else p.suffix.lower().lstrip(".")
                candidates.append((source, p))
        for p in sorted(model_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".json", ".yaml", ".yml"}:
                rel_parts = p.relative_to(model_dir).parts
                if len(rel_parts) <= 2:
                    source = "args_yaml" if p.name in {"args.yaml", "args.yml"} else p.suffix.lower().lstrip(".")
                    candidates.append((source, p))
    return candidates


def infer_img_size_from_model_context(model_path: Path) -> int | None:
    value, _ = infer_img_size_with_source(model_path)
    return value


def infer_img_size_with_source(model_path: Path) -> tuple[int | None, str]:
    seen: set[Path] = set()
    for source, candidate in _collect_context_candidates(model_path):
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        value = _read_img_size_from_meta_file(resolved)
        if value is not None:
            if source in {"json", "yml", "yaml"} and "training_metadata" in resolved.name:
                source = "training_metadata"
            return value, source
    return None, "fallback_640"
