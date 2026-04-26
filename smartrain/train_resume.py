from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from smartrain.run_discovery import find_run_directories
from smartrain.workspace_paths import WorkspaceLayout

RUN_STATUS_RESUMABLE_INCOMPLETE = "resumable_incomplete"
RUN_STATUS_INCOMPLETE_NON_RESUMABLE = "incomplete_non_resumable"
RUN_STATUS_TRAINING_COMPLETE_TEST_PENDING = "training_complete_test_pending"
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
    has_test_dir: bool


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
    test_dir = os.path.join(rd, "test")

    has_args_yaml = os.path.isfile(args_yaml)
    has_results_csv = os.path.isfile(results_csv)
    has_last_pt = os.path.isfile(last_pt)
    has_best_pt = os.path.isfile(best_pt)
    has_metadata = os.path.isfile(metadata_path)
    has_test_dir = os.path.isdir(test_dir)

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
    if has_test_dir:
        reasons.append("test_dir_present")

    meta = _read_json(metadata_path) if has_metadata else None
    training_success = _training_success_from_metadata(meta)
    if has_metadata and training_success is True:
        reasons.append("metadata_training_success_true")
        if has_test_dir:
            return RunDiagnosis(
                run_dir=rd,
                status=RUN_STATUS_COMPLETED,
                reasons=reasons,
                has_args_yaml=has_args_yaml,
                has_results_csv=has_results_csv,
                has_last_pt=has_last_pt,
                has_best_pt=has_best_pt,
                has_metadata=has_metadata,
                has_test_dir=has_test_dir,
            )
        reasons.append("missing_test_dir")
        return RunDiagnosis(
            run_dir=rd,
            status=RUN_STATUS_TRAINING_COMPLETE_TEST_PENDING,
            reasons=reasons,
            has_args_yaml=has_args_yaml,
            has_results_csv=has_results_csv,
            has_last_pt=has_last_pt,
            has_best_pt=has_best_pt,
            has_metadata=has_metadata,
            has_test_dir=has_test_dir,
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
        has_test_dir=has_test_dir,
    )


def discover_runs(workspace_root: str) -> list[str]:
    runs_root = WorkspaceLayout(workspace_root).runs
    return sorted(set(os.path.abspath(x) for x in find_run_directories(runs_root)))


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_training_metadata_", suffix=".json", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
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


def diagnose_workspace_runs(workspace_root: str) -> list[RunDiagnosis]:
    return [diagnose_run(rd) for rd in discover_runs(workspace_root)]


def list_incomplete_runs(workspace_root: str) -> list[RunDiagnosis]:
    all_runs = diagnose_workspace_runs(workspace_root)
    return [
        r
        for r in all_runs
        if r.status
        in (
            RUN_STATUS_RESUMABLE_INCOMPLETE,
            RUN_STATUS_INCOMPLETE_NON_RESUMABLE,
            RUN_STATUS_TRAINING_COMPLETE_TEST_PENDING,
        )
    ]


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


def _load_train_args_yaml(run_dir: str) -> dict[str, Any]:
    args_yaml = os.path.join(run_dir, "train", "args.yaml")
    if not os.path.isfile(args_yaml):
        return {}
    try:
        with open(args_yaml, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _infer_model_from_args(run_dir: str) -> str | None:
    payload = _load_train_args_yaml(run_dir)
    raw_model = payload.get("model")
    if not isinstance(raw_model, str) or not raw_model.strip():
        return None
    token = os.path.basename(raw_model.strip())
    if token.endswith(".pt"):
        token = token[:-3]
    if token.endswith(".yaml"):
        token = token[:-5]
    return token or None


def _infer_dataset_name_from_run_dir(run_dir: str) -> str | None:
    name = os.path.basename(os.path.dirname(os.path.abspath(run_dir)))
    return name or None


def _infer_workspace_root_from_run_dir(run_dir: str) -> str | None:
    cur = os.path.abspath(run_dir)
    while True:
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        if os.path.basename(parent) == "runs":
            return os.path.dirname(parent)
        cur = parent


def _hydrate_training_info(payload: dict[str, Any], run_dir: str) -> None:
    ti = payload.setdefault("training_info", {})
    if not isinstance(ti, dict):
        ti = {}
        payload["training_info"] = ti

    model = _infer_model_from_args(run_dir)
    if model:
        ti.setdefault("model", model)

    ds = ti.setdefault("dataset", {})
    if not isinstance(ds, dict):
        ds = {}
        ti["dataset"] = ds

    dataset_name = _infer_dataset_name_from_run_dir(run_dir)
    if dataset_name:
        ds.setdefault("name", dataset_name)

    dataset_path = _load_dataset_from_runtime_yaml(run_dir)
    if dataset_path:
        ds.setdefault("path_absolute", dataset_path)

    workspace_root = _infer_workspace_root_from_run_dir(run_dir)
    if workspace_root:
        workspace_block = payload.setdefault("workspace", {})
        if not isinstance(workspace_block, dict):
            workspace_block = {}
            payload["workspace"] = workspace_block
        workspace_block.setdefault("root", ".")
        try:
            workspace_block.setdefault("run_directory_relative", os.path.relpath(run_dir, workspace_root))
            if dataset_path and os.path.abspath(dataset_path).startswith(os.path.abspath(workspace_root) + os.sep):
                workspace_block.setdefault("dataset_path_relative", os.path.relpath(dataset_path, workspace_root))
                ds.setdefault("path_under_workspace", os.path.relpath(dataset_path, workspace_root))
        except Exception:
            pass

def _is_dataset_dir(path: str | None) -> bool:
    if not path:
        return False
    p = os.path.abspath(os.path.expanduser(path))
    return os.path.isdir(p) and os.path.isfile(os.path.join(p, "data.yaml"))


def resolve_dataset_path_for_resume(run_dir: str, workspace_root: str) -> str | None:
    rd = os.path.abspath(run_dir)
    ws = os.path.abspath(workspace_root)

    runtime_path = _load_dataset_from_runtime_yaml(rd)
    if _is_dataset_dir(runtime_path):
        return os.path.abspath(os.path.expanduser(str(runtime_path)))

    metadata_path = os.path.join(rd, "training_metadata.json")
    meta = _read_json(metadata_path)
    if isinstance(meta, dict):
        ds = ((meta.get("training_info") or {}).get("dataset") or {})
        if isinstance(ds, dict):
            path_under_workspace = ds.get("path_under_workspace")
            if isinstance(path_under_workspace, str) and path_under_workspace.strip():
                candidate = os.path.join(ws, path_under_workspace)
                if _is_dataset_dir(candidate):
                    return os.path.abspath(candidate)

            path_relative = ds.get("path_relative")
            if isinstance(path_relative, str) and path_relative.strip():
                candidate = os.path.join(rd, path_relative)
                if _is_dataset_dir(candidate):
                    return os.path.abspath(candidate)

            path_absolute = ds.get("path_absolute")
            if _is_dataset_dir(path_absolute if isinstance(path_absolute, str) else None):
                return os.path.abspath(os.path.expanduser(str(path_absolute)))

        workspace_block = meta.get("workspace")
        if isinstance(workspace_block, dict):
            ds_rel = workspace_block.get("dataset_path_relative")
            if isinstance(ds_rel, str) and ds_rel.strip():
                candidate = os.path.join(ws, ds_rel)
                if _is_dataset_dir(candidate):
                    return os.path.abspath(candidate)

    dataset_name = os.path.basename(os.path.dirname(rd.rstrip(os.sep)))
    if dataset_name:
        direct_candidate = os.path.join(WorkspaceLayout(ws).work_datasets, dataset_name)
        if _is_dataset_dir(direct_candidate):
            return os.path.abspath(direct_candidate)

        info_path = WorkspaceLayout(ws).work_datasets_info_path()
        info = _read_json(info_path)
        if isinstance(info, dict):
            entry = info.get(dataset_name)
            if isinstance(entry, dict):
                data_path = entry.get("data_path")
                if isinstance(data_path, str) and data_path.strip():
                    candidate = os.path.join(ws, data_path)
                    if _is_dataset_dir(candidate):
                        return os.path.abspath(candidate)
            # Legacy fallback: scan entries by data_path suffix
            for _, entry_val in info.items():
                if not isinstance(entry_val, dict):
                    continue
                data_path = entry_val.get("data_path")
                if not isinstance(data_path, str):
                    continue
                if os.path.basename(os.path.normpath(data_path)) != dataset_name:
                    continue
                candidate = os.path.join(ws, data_path)
                if _is_dataset_dir(candidate):
                    return os.path.abspath(candidate)

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
    tr["last_resume_attempt_success"] = bool(success)
    tr["last_resume_attempt_error"] = error

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

    attempts = payload.setdefault("resume_attempts", [])
    if not isinstance(attempts, list):
        attempts = []
        payload["resume_attempts"] = attempts
    attempts.append(
        {
            "at": training_ts["end"],
            "success": bool(success),
            "error": error,
        }
    )

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
                "has_test_dir": diagnosis.has_test_dir,
            },
        }

    _hydrate_training_info(payload, run_dir)

    _atomic_write_json(metadata_path, payload)


def update_resume_test_metadata(
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
    tst = status.setdefault("testing", {})
    if not isinstance(tst, dict):
        tst = {}
        status["testing"] = tst
    tst["success"] = bool(success)
    tst["error"] = error

    ts = payload.setdefault("timestamps", {})
    if not isinstance(ts, dict):
        ts = {}
        payload["timestamps"] = ts
    testing_ts = ts.setdefault("testing", {})
    if not isinstance(testing_ts, dict):
        testing_ts = {}
        ts["testing"] = testing_ts
    testing_ts["end"] = datetime.now().isoformat()
    if not testing_ts.get("start"):
        testing_ts["start"] = testing_ts["end"]

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
                "has_test_dir": diagnosis.has_test_dir,
            },
        }

    _hydrate_training_info(payload, run_dir)

    _atomic_write_json(metadata_path, payload)


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
