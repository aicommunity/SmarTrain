from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any, Callable

import yaml

from smartrain.core.testing.ultralytics_artifact_resolver import (
    PROVENANCE_TEST,
    resolve_ultralytics_artifacts,
)
from smartrain.core.testing.ultralytics_test_contract import ULTRALYTICS_TEST_COLLECT_IMAGE_NAMES
from smartrain.services.analyze.metrics_reader import training_args_yaml_path
from smartrain.services.analyze.run_query import _system_profile_flat_from_training_metadata


def _load_training_metadata(run_dir: str) -> dict[str, Any]:
    path = os.path.join(run_dir, "training_metadata.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_args_yaml(run_dir: str) -> dict[str, Any]:
    path = training_args_yaml_path(run_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def build_ultralytics_run_info(run_dir: str, *, model_fallback: str | None = None) -> dict[str, Any]:
    md = _load_training_metadata(run_dir)
    ti = md.get("training_info") if isinstance(md.get("training_info"), dict) else {}
    hp = ti.get("hyperparameters") if isinstance(ti.get("hyperparameters"), dict) else {}
    inf = md.get("inference") if isinstance(md.get("inference"), dict) else {}
    args_yaml = _read_args_yaml(run_dir)
    model = str(ti.get("model") or args_yaml.get("model") or model_fallback or "").strip() or None
    dataset = ti.get("dataset") if isinstance(ti.get("dataset"), dict) else {}
    dataset_name = str(dataset.get("name") or "").strip() or None
    epochs = hp.get("epochs")
    if epochs is None:
        epochs = args_yaml.get("epochs")
    batch_size = hp.get("batch_size")
    if batch_size is None:
        batch_size = args_yaml.get("batch")
    train_image_size = hp.get("image_size")
    if train_image_size is None:
        train_image_size = args_yaml.get("imgsz")
    val_imgsz = inf.get("imgsz")
    if val_imgsz is None:
        val_imgsz = args_yaml.get("imgsz")
    return {
        "model": model,
        "dataset_name": dataset_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_image_size": train_image_size,
        "val_imgsz": val_imgsz,
    }


def collect_ultralytics_test_artifacts(
    session_root: str,
    run_dirs: list[str],
    abbreviations: dict[str, str],
    *,
    run_test_backend_dir_cb: Callable[[str, str], str],
    build_run_record_unified_cb: Callable[[str], Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    _ = run_test_backend_dir_cb
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, str]] = []
    out_root = os.path.join(session_root, "artifacts", "ultralytics-test")
    csv_names = ("pr.csv", "pr_per_class.csv")
    image_patterns = ULTRALYTICS_TEST_COLLECT_IMAGE_NAMES

    for rd in run_dirs:
        run_name = os.path.basename(rd.rstrip(os.sep))
        run_code = abbreviations.get(run_name, run_name)
        resolved = resolve_ultralytics_artifacts(rd)
        rec = build_run_record_unified_cb(rd)
        run_info = build_ultralytics_run_info(rd, model_fallback=rec.model)
        if not run_info.get("dataset_name") and rec.dataset_name:
            run_info["dataset_name"] = rec.dataset_name
        preferred_test_dir = resolved.primary_test_dir or (
            resolved.source_dirs[0].path if resolved.source_dirs else os.path.join(rd, "tests", "test-ultralytics")
        )
        artifact_sources = {name: prov for name, (_path, prov) in resolved.resolved.items()}
        row: dict[str, Any] = {
            "run_dir": rd,
            "run_name": run_name,
            "run_code": run_code,
            "test_dir": preferred_test_dir,
            "exists": bool(resolved.resolved),
            "completeness": resolved.completeness,
            "missing_files": list(resolved.missing_required),
            "artifact_sources": artifact_sources,
            "run_info": run_info,
            "machine_info": _system_profile_flat_from_training_metadata(rd),
            "files": sorted(resolved.resolved.keys()),
            "csv": {},
            "images": [],
        }
        if not resolved.resolved:
            rows.append(row)
            continue

        safe_code = re.sub(r"[^\w.\-+]+", "_", str(run_code), flags=re.UNICODE).strip("._") or "run"
        dst_dir = os.path.join(out_root, safe_code)
        os.makedirs(dst_dir, exist_ok=True)

        for name in csv_names + image_patterns:
            hit = resolved.resolved.get(name)
            if hit is None:
                continue
            src, _prov = hit
            dst = os.path.join(dst_dir, name)
            try:
                shutil.copy2(src, dst)
                rel = os.path.relpath(dst, session_root)
                if name in csv_names:
                    row["csv"][name] = rel
                    artifacts.append({"role": f"ultralytics_test_{name.replace('.', '_')}", "path": rel})
                else:
                    row["images"].append(rel)
                    artifacts.append({"role": "ultralytics_test_image", "path": rel})
            except Exception:
                pass

        if resolved.completeness == "train_val_fallback":
            row["completeness_note"] = (
                "Some plots were taken from train-ultralytics (validation during training), not test-split."
            )
        elif resolved.completeness == "partial_csv_only":
            has_test_csv = all(
                resolved.resolved.get(n, (None, None))[1] == PROVENANCE_TEST for n in csv_names if n in resolved.resolved
            )
            if not has_test_csv:
                row["completeness_note"] = "Only PR CSV files are available; rich Ultralytics plots are missing."

        rows.append(row)
    return rows, artifacts
