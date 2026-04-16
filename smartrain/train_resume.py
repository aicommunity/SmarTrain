from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from smartrain.workspace_paths import WorkspaceLayout

RUN_STATUS_RESUMABLE_INCOMPLETE = "resumable_incomplete"
RUN_STATUS_INCOMPLETE_NON_RESUMABLE = "incomplete_non_resumable"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_BROKEN = "broken"


@dataclass
class RunDiagnosis:
    run_dir: str
    status: str
    reasons: list[str]
    has_args_yaml: bool
    has_results_csv: bool
    has_last_pt: bool
    has_best_pt: bool
    has_metadata: bool


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _training_success_from_metadata(meta: dict[str, Any] | None) -> bool | None:
    if not meta:
        return None
    st = meta.get("status")
    if not isinstance(st, dict):
        return None
    tr = st.get("training")
    if not isinstance(tr, dict):
        return None
    val = tr.get("success")
    return val if isinstance(val, bool) else None


def _is_results_csv_readable(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline()
        return bool(line.strip())
    except Exception:
        return False


def diagnose_run(run_dir: str) -> RunDiagnosis:
    rd = os.path.abspath(run_dir)
    args_yaml = os.path.join(rd, "train", "args.yaml")
    results_csv = os.path.join(rd, "train", "results.csv")
    last_pt = os.path.join(rd, "train", "weights", "last.pt")
    best_pt = os.path.join(rd, "train", "weights", "best.pt")
    metadata_path = os.path.join(rd, "training_metadata.json")

    has_args_yaml = os.path.isfile(args_yaml)
    has_results_csv = os.path.isfile(results_csv)
    has_last_pt = os.path.isfile(last_pt)
    has_best_pt = os.path.isfile(best_pt)
    has_metadata = os.path.isfile(metadata_path)

    reasons: list[str] = []
    if has_args_yaml:
        reasons.append("args_yaml_present")
    if has_results_csv:
        reasons.append("results_csv_present")
        if not _is_results_csv_readable(results_csv):
            reasons.append("results_csv_unreadable")
    if has_last_pt:
        reasons.append("last_checkpoint_present")
    if has_best_pt:
        reasons.append("best_checkpoint_present")
    if has_metadata:
        reasons.append("training_metadata_present")

    meta = _read_json(metadata_path) if has_metadata else None
    training_success = _training_success_from_metadata(meta)
    if has_metadata and training_success is True:
        reasons.append("metadata_training_success_true")
        return RunDiagnosis(
            run_dir=rd,
            status=RUN_STATUS_COMPLETED,
            reasons=reasons,
            has_args_yaml=has_args_yaml,
            has_results_csv=has_results_csv,
            has_last_pt=has_last_pt,
            has_best_pt=has_best_pt,
            has_metadata=has_metadata,
        )

    if has_last_pt:
        reasons.append("resume_checkpoint_available")
        status = RUN_STATUS_RESUMABLE_INCOMPLETE
    elif has_args_yaml or has_results_csv or (has_metadata and training_success is not True):
        reasons.append("missing_last_checkpoint")
        status = RUN_STATUS_INCOMPLETE_NON_RESUMABLE
    else:
        reasons.append("insufficient_run_artifacts")
        status = RUN_STATUS_BROKEN

    return RunDiagnosis(
        run_dir=rd,
        status=status,
        reasons=reasons,
        has_args_yaml=has_args_yaml,
        has_results_csv=has_results_csv,
        has_last_pt=has_last_pt,
        has_best_pt=has_best_pt,
        has_metadata=has_metadata,
    )


def discover_runs(workspace_root: str) -> list[str]:
    runs_root = WorkspaceLayout(workspace_root).runs
    out: list[str] = []
    if not os.path.isdir(runs_root):
        return out
    for dirpath, _, filenames in os.walk(runs_root):
        fset = set(filenames)
        if "training_metadata.json" in fset:
            out.append(dirpath)
            continue
        if "args.yaml" in fset and os.path.basename(dirpath) == "train":
            out.append(os.path.dirname(dirpath))
    return sorted(set(os.path.abspath(x) for x in out))


def diagnose_workspace_runs(workspace_root: str) -> list[RunDiagnosis]:
    return [diagnose_run(rd) for rd in discover_runs(workspace_root)]


def list_incomplete_runs(workspace_root: str) -> list[RunDiagnosis]:
    all_runs = diagnose_workspace_runs(workspace_root)
    return [r for r in all_runs if r.status in (RUN_STATUS_RESUMABLE_INCOMPLETE, RUN_STATUS_INCOMPLETE_NON_RESUMABLE)]


def _load_dataset_from_runtime_yaml(run_dir: str) -> str | None:
    data_yaml = os.path.join(run_dir, "_runtime_data_train.yaml")
    if not os.path.isfile(data_yaml):
        return None
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
        path_val = obj.get("path")
        return str(path_val) if isinstance(path_val, str) and path_val.strip() else None
    except Exception:
        return None


def update_resume_metadata(
    run_dir: str,
    *,
    success: bool,
    error: str | None = None,
    diagnosis: RunDiagnosis | None = None,
) -> None:
    metadata_path = os.path.join(run_dir, "training_metadata.json")
    payload: dict[str, Any] = {}
    existing = _read_json(metadata_path)
    if isinstance(existing, dict):
        payload = existing

    status = payload.setdefault("status", {})
    if not isinstance(status, dict):
        status = {}
        payload["status"] = status
    tr = status.setdefault("training", {})
    if not isinstance(tr, dict):
        tr = {}
        status["training"] = tr
    tr["success"] = bool(success)
    tr["error"] = error

    ts = payload.setdefault("timestamps", {})
    if not isinstance(ts, dict):
        ts = {}
        payload["timestamps"] = ts
    training_ts = ts.setdefault("training", {})
    if not isinstance(training_ts, dict):
        training_ts = {}
        ts["training"] = training_ts
    training_ts["end"] = datetime.now().isoformat()
    if not training_ts.get("start"):
        training_ts["start"] = training_ts["end"]

    if diagnosis is not None:
        payload["diagnostics"] = {
            "run_state": diagnosis.status,
            "reasons": list(diagnosis.reasons),
            "artifacts": {
                "has_args_yaml": diagnosis.has_args_yaml,
                "has_results_csv": diagnosis.has_results_csv,
                "has_last_pt": diagnosis.has_last_pt,
                "has_best_pt": diagnosis.has_best_pt,
                "has_metadata": diagnosis.has_metadata,
            },
        }

    dataset_path = _load_dataset_from_runtime_yaml(run_dir)
    if dataset_path:
        ti = payload.setdefault("training_info", {})
        if isinstance(ti, dict):
            ds = ti.setdefault("dataset", {})
            if isinstance(ds, dict):
                ds.setdefault("path_absolute", dataset_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resume_training_in_run(run_dir: str) -> None:
    last_pt = os.path.join(run_dir, "train", "weights", "last.pt")
    if not os.path.isfile(last_pt):
        raise FileNotFoundError(f"Resume checkpoint not found: {last_pt}")
    model = YOLO(last_pt)
    model.train(resume=True)


def format_resume_option(d: RunDiagnosis) -> str:
    base = Path(d.run_dir).name
    status = d.status
    return f"{base} | {status}"
