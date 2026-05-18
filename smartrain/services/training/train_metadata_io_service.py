from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from smartrain.core.runtime.path_portable import relativize_if_under
from smartrain.core.runtime.run_artifacts import preferred_run_model_path, run_tests_dir
from smartrain.core.training.confidence_recommendation import (
    read_recommendation_file,
    recommendation_file_path,
)


def relative_to_workspace(path: str, workspace_root: str) -> str:
    ap = os.path.abspath(path)
    wr = os.path.abspath(workspace_root)
    try:
        return os.path.relpath(ap, wr)
    except ValueError:
        return ap


def write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass


def get_relative_path(target_path: str, base_path: str) -> str:
    try:
        target = Path(os.path.abspath(target_path))
        base = Path(os.path.abspath(base_path))
        try:
            relative = os.path.relpath(target, base)
            return relative
        except ValueError:
            return target.as_posix()
    except Exception:
        return os.path.abspath(target_path)


def recommendation_summary_for_metadata(model_dir: str) -> dict[str, Any] | None:
    out: dict[str, Any] = {"files": {}, "status": {}}
    found = False
    for split in ("val", "test"):
        p = recommendation_file_path(model_dir, split)
        payload = read_recommendation_file(p)
        if not isinstance(payload, dict):
            continue
        found = True
        out["files"][split] = os.path.basename(p)
        out["status"][split] = payload.get("status")
    return out if found else None


def ensure_initial_training_metadata(
    *,
    model_dir: str,
    dataset_path: str,
    model_version: str,
    epochs: int,
    batch: int,
    img_size: int,
    training_start_time: datetime,
    dataset_hash: str | None,
    workspace_root: str | None,
    task_type: str,
) -> None:
    metadata_file = os.path.join(model_dir, "training_metadata.json")
    payload: dict[str, Any] = {}
    if os.path.isfile(metadata_file):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                payload = existing
        except Exception:
            payload = {}

    ti = payload.setdefault("training_info", {})
    if not isinstance(ti, dict):
        ti = {}
        payload["training_info"] = ti
    ti.setdefault("framework", "ultralytics")
    provider = ti.setdefault("provider", {})
    if not isinstance(provider, dict):
        provider = {}
        ti["provider"] = provider
    provider.setdefault("type", "builtin")
    provider.setdefault("id", "ultralytics")
    ti.setdefault("task_type", task_type or "detection")
    ti.setdefault("model", model_version)
    ds = ti.setdefault("dataset", {})
    if not isinstance(ds, dict):
        ds = {}
        ti["dataset"] = ds
    ds.setdefault("name", os.path.basename(os.path.normpath(dataset_path)))
    ds.setdefault("path_relative", get_relative_path(dataset_path, model_dir))
    ds.setdefault("hash", dataset_hash)
    hp = ti.setdefault("hyperparameters", {})
    if not isinstance(hp, dict):
        hp = {}
        ti["hyperparameters"] = hp
    hp.setdefault("epochs", epochs)
    hp.setdefault("batch_size", batch)
    hp.setdefault("image_size", img_size)

    ts = payload.setdefault("timestamps", {})
    if not isinstance(ts, dict):
        ts = {}
        payload["timestamps"] = ts
    tr_ts = ts.setdefault("training", {})
    if not isinstance(tr_ts, dict):
        tr_ts = {}
        ts["training"] = tr_ts
    tr_ts.setdefault("start", training_start_time.isoformat())
    tr_ts.setdefault("end", None)
    tr_ts.setdefault("duration_seconds", None)
    te_ts = ts.setdefault("testing", {})
    if not isinstance(te_ts, dict):
        te_ts = {}
        ts["testing"] = te_ts
    te_ts.setdefault("start", None)
    te_ts.setdefault("end", None)
    te_ts.setdefault("duration_seconds", None)

    status = payload.setdefault("status", {})
    if not isinstance(status, dict):
        status = {}
        payload["status"] = status
    tr_status = status.setdefault("training", {})
    if not isinstance(tr_status, dict):
        tr_status = {}
        status["training"] = tr_status
    tr_status.setdefault("success", None)
    tr_status.setdefault("error", None)
    te_status = status.setdefault("testing", {})
    if not isinstance(te_status, dict):
        te_status = {}
        status["testing"] = te_status
    te_status.setdefault("success", None)
    te_status.setdefault("error", None)

    paths = payload.setdefault("paths", {})
    if not isinstance(paths, dict):
        paths = {}
        payload["paths"] = paths
    paths.setdefault("model_directory", ".")
    paths.setdefault("best_model", None)

    if workspace_root is not None:
        wb = payload.setdefault("workspace", {})
        if not isinstance(wb, dict):
            wb = {}
            payload["workspace"] = wb
        wb.setdefault("root", ".")
        wb.setdefault("dataset_path_relative", relative_to_workspace(dataset_path, workspace_root))
        wb.setdefault("run_directory_relative", relative_to_workspace(model_dir, workspace_root))

    write_json_atomic(metadata_file, payload)


def save_training_metadata(
    model_dir: str,
    dataset_path: str,
    *,
    model_version: str | None = None,
    training_start_time: datetime | None = None,
    training_end_time: datetime | None = None,
    test_start_time: datetime | None = None,
    test_end_time: datetime | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    img_size: int | None = None,
    training_success: bool = True,
    training_error: str | None = None,
    test_success: bool = True,
    test_error: str | None = None,
    dataset_hash: str | None = None,
    inference: dict[str, Any] | None = None,
    workspace_root: str | None = None,
    task_type: str | None = None,
    ultralytics_train_summary: dict[str, Any] | None = None,
    training_provider: str = "ultralytics",
    external_provider_id: str | None = None,
    system_profile: dict[str, Any] | None = None,
    matplotlib_runtime: dict[str, Any] | None = None,
    confidence_recommendation_config: dict[str, Any] | None = None,
    sync_test_artifacts_manifest_cb: Callable[..., dict[str, Any]],
) -> None:
    ds_abs = os.path.abspath(dataset_path)
    dataset_block: dict[str, Any] = {
        "name": os.path.basename(os.path.normpath(dataset_path)),
        "path_relative": get_relative_path(dataset_path, model_dir),
        "hash": dataset_hash,
    }
    if workspace_root is not None:
        wr_abs = os.path.abspath(workspace_root)
        if ds_abs == wr_abs or ds_abs.startswith(wr_abs + os.sep):
            rel_uw = relativize_if_under(workspace_root, ds_abs)
            if rel_uw is not None:
                dataset_block["path_under_workspace"] = rel_uw
        else:
            dataset_block["path_absolute"] = ds_abs
    else:
        dataset_block["path_absolute"] = ds_abs

    metadata = {
        "training_info": {
            "framework": "ultralytics" if training_provider == "ultralytics" else "external",
            "provider": {
                "type": "builtin" if training_provider == "ultralytics" else "external",
                "id": external_provider_id if training_provider != "ultralytics" else "ultralytics",
            },
            "task_type": task_type or "detection",
            "model": model_version,
            "dataset": dataset_block,
            "hyperparameters": {
                "epochs": epochs,
                "batch_size": batch,
                "image_size": img_size,
            },
        },
        "timestamps": {
            "training": {
                "start": training_start_time.isoformat() if training_start_time else None,
                "end": training_end_time.isoformat() if training_end_time else None,
                "duration_seconds": (training_end_time - training_start_time).total_seconds()
                if training_start_time and training_end_time
                else None,
            },
            "testing": {
                "start": test_start_time.isoformat() if test_start_time else None,
                "end": test_end_time.isoformat() if test_end_time else None,
                "duration_seconds": (test_end_time - test_start_time).total_seconds()
                if test_start_time and test_end_time
                else None,
            },
        },
        "status": {
            "training": {
                "success": training_success,
                "error": training_error,
            },
            "testing": {
                "success": test_success,
                "error": test_error,
            },
        },
        "paths": {
            "model_directory": ".",
            "best_model": os.path.basename(preferred_run_model_path(model_dir, ".pt"))
            if os.path.exists(preferred_run_model_path(model_dir, ".pt"))
            else None,
        },
    }

    if ultralytics_train_summary:
        metadata["training_info"]["ultralytics_train"] = ultralytics_train_summary

    if workspace_root is not None:
        metadata["workspace"] = {
            "root": ".",
            "dataset_path_relative": relative_to_workspace(dataset_path, workspace_root),
            "run_directory_relative": relative_to_workspace(model_dir, workspace_root),
        }

    if inference:
        metadata["inference"] = {k: v for k, v in inference.items() if v is not None}
    sp_out: dict[str, Any] = dict(system_profile) if system_profile else {}
    if matplotlib_runtime:
        sp_out["matplotlib_runtime"] = matplotlib_runtime
    if sp_out:
        metadata["system_profile"] = sp_out
    rec_summary = recommendation_summary_for_metadata(model_dir)
    if rec_summary:
        if confidence_recommendation_config:
            rec_summary["config"] = confidence_recommendation_config
        metadata["recommendations"] = {"confidence": rec_summary}
    test_manifest = sync_test_artifacts_manifest_cb(
        model_dir,
        target_by_format={"pt": metadata["paths"].get("best_model")},
        backend_by_format={"pt": "ultralytics"},
    )
    formats_payload = test_manifest.get("formats")
    if isinstance(formats_payload, dict) and formats_payload:
        metadata["test_artifacts_by_format"] = formats_payload

    metadata_file = os.path.join(model_dir, "training_metadata.json")

    try:
        write_json_atomic(metadata_file, metadata)
        print(f"[INFO] Training metadata saved: {metadata_file}")
    except Exception as e:
        print(f"[WARNING] Failed to save metadata: {e}")


def save_metrics_csv(test_result: Any, model_dir: str) -> str:
    csv_file = os.path.join(str(run_tests_dir(model_dir)), "test_metrics.csv")
    csv_data = test_result.to_csv()
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(csv_data)
    return csv_file

