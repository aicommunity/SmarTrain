from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from smartrain.core.runtime.mpl_runtime import configure_matplotlib_before_ultralytics, ensure_matplotlib_training_runtime

configure_matplotlib_before_ultralytics()
from ultralytics import YOLO

from smartrain.services.analyze.metrics_reader import results_csv_path, training_args_yaml_path
from smartrain.workflows.testing.model_test_service import has_complete_test_artifacts, missing_test_artifacts
from smartrain.core.runtime.run_artifacts import (
    canonical_run_model_path,
    ensure_run_layout,
    resolve_run_model,
    run_tmp_dir,
)
from smartrain.core.runtime.run_discovery import find_run_directories
from smartrain.core.runtime.workspace_paths import WorkspaceLayout

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


def _warn_if_archive_like_run_path(run_dir: str) -> None:
    for part in Path(run_dir).parts:
        pl = part.lower()
        if pl.endswith(".tar.gz") or pl.endswith(".zip") or pl.endswith(".tar"):
            print(
                f"[WARN] Run directory path contains archive-like segment {part!r}. "
                "Confirm SMART_TRAIN_WORKSPACE is a normal directory (not a cloud virtual path)."
            )
            return


def _load_ckpt_dict(path: str) -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    try:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def is_ultralytics_train_resume_checkpoint(path: str) -> bool:
    """Match Ultralytics ``model.train(resume=True)`` gate: epoch >= 0 and optimizer state present."""
    ckpt = _load_ckpt_dict(path)
    if not ckpt:
        return False
    try:
        epoch_ok = int(ckpt.get("epoch", -1)) >= 0
    except (TypeError, ValueError):
        epoch_ok = False
    return epoch_ok and ckpt.get("optimizer") is not None


def collect_last_pt_paths(run_dir: str) -> list[Path]:
    """All ``last.pt`` paths under the run (any Ultralytics train subfolder)."""
    rd = Path(run_dir).resolve()
    out: list[Path] = []
    seen: set[str] = set()
    for p in (
        rd / "train-ultralytics" / "weights" / "last.pt",
        rd / "train" / "weights" / "last.pt",
    ):
        if p.is_file():
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                out.append(p)
    globs = sorted(
        (p for p in rd.glob("train-ultralytics*/weights/last.pt") if p.is_file()),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for p in globs:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def resolve_last_checkpoint_path(run_dir: str) -> str | None:
    """Return a ``last.pt`` that can resume training (optimizer + epoch), or ``None``.

    Post-training checkpoints often have the optimizer stripped; those must not be used with
    ``resume=True`` (Ultralytics would start a new run with wrong defaults). See
    ``collect_last_pt_paths`` for every on-disk ``last.pt``.
    """
    for p in collect_last_pt_paths(run_dir):
        if is_ultralytics_train_resume_checkpoint(str(p)):
            return str(p)
    return None


def _read_epochs_from_args_yaml(path: str) -> int | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return None
        e = data.get("epochs")
        if e is None:
            return None
        return int(e)
    except Exception:
        return None


def _last_epoch_from_results_csv(path: str) -> int | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None
        last_row = rows[-1]
        for key in ("epoch", "\ufeffepoch"):
            if key in last_row and str(last_row.get(key) or "").strip() != "":
                return int(float(last_row[key]))
        if reader.fieldnames:
            first = reader.fieldnames[0]
            if first in last_row and str(last_row.get(first) or "").strip() != "":
                return int(float(last_row[first]))
    except Exception:
        return None
    return None


def _results_reached_configured_epochs(results_csv: str, args_yaml: str) -> bool:
    target = _read_epochs_from_args_yaml(args_yaml)
    if target is None or target < 1:
        return False
    last_ep = _last_epoch_from_results_csv(results_csv)
    if last_ep is None:
        return False
    return last_ep >= target


def diagnose_run(run_dir: str) -> RunDiagnosis:
    rd = os.path.abspath(run_dir)
    _warn_if_archive_like_run_path(rd)
    ensure_run_layout(rd)
    results_real = results_csv_path(rd)
    results_csv = results_real if results_real is not None else os.path.join(rd, "train", "results.csv")
    args_yaml = training_args_yaml_path(rd, results_real)
    last_pt_candidates = collect_last_pt_paths(rd)
    any_last_pt = len(last_pt_candidates) > 0
    resumable_last = resolve_last_checkpoint_path(rd)
    has_last_pt = resumable_last is not None
    best_pt = canonical_run_model_path(rd, ".pt")
    legacy_best = resolve_run_model(rd)
    metadata_path = os.path.join(rd, "training_metadata.json")
    test_dir = os.path.join(rd, "test")

    has_args_yaml = os.path.isfile(args_yaml)
    has_results_csv = os.path.isfile(results_csv)
    has_best_pt = os.path.isfile(best_pt) or (legacy_best is not None and legacy_best.is_file())
    has_metadata = os.path.isfile(metadata_path)
    has_test_dir = os.path.isdir(test_dir)
    pt_test_complete = has_complete_test_artifacts(rd, "pt")

    reasons: list[str] = []
    if has_args_yaml:
        reasons.append("args_yaml_present")
    if has_results_csv:
        reasons.append("results_csv_present")
        if not _is_results_csv_readable(results_csv):
            reasons.append("results_csv_unreadable")
    if has_last_pt:
        reasons.append("last_checkpoint_present")
    if any_last_pt and not has_last_pt:
        reasons.append("last_pt_present_but_stripped")
    if has_best_pt:
        reasons.append("best_checkpoint_present")
    if has_metadata:
        reasons.append("training_metadata_present")
    if has_test_dir:
        reasons.append("test_dir_present")
    if pt_test_complete:
        reasons.append("pt_test_artifacts_complete")

    meta = _read_json(metadata_path) if has_metadata else None
    training_success = _training_success_from_metadata(meta)

    if (
        not has_metadata
        and not has_last_pt
        and has_best_pt
        and has_results_csv
        and _is_results_csv_readable(results_csv)
        and not pt_test_complete
        and any_last_pt
        and has_args_yaml
        and _results_reached_configured_epochs(results_csv, args_yaml)
    ):
        reasons.append("finalize_stripped_no_metadata_epochs_ok")
        reasons.append("missing_test_artifacts")
        for item in missing_test_artifacts(rd, "pt"):
            reasons.append(f"missing_{item}")
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

    if (
        has_metadata
        and training_success is not True
        and has_best_pt
        and has_results_csv
        and _is_results_csv_readable(results_csv)
        and not pt_test_complete
    ):
        reasons.append("finalize_after_training_error_heuristic")
        reasons.append("missing_test_artifacts")
        for item in missing_test_artifacts(rd, "pt"):
            reasons.append(f"missing_{item}")
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

    if has_metadata and training_success is True:
        reasons.append("metadata_training_success_true")
        if pt_test_complete:
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
        reasons.append("missing_test_artifacts")
        for item in missing_test_artifacts(rd, "pt"):
            reasons.append(f"missing_{item}")
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
    preferred = os.path.join(str(run_tmp_dir(run_dir)), "_runtime_data_train.yaml")
    legacy = os.path.join(run_dir, "_runtime_data_train.yaml")
    data_yaml = preferred if os.path.isfile(preferred) else legacy
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
    rc = results_csv_path(run_dir)
    args_yaml = training_args_yaml_path(run_dir, rc)
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
    token = token.strip().lower()
    if token in {"last", "best"}:
        run_name = os.path.basename(os.path.abspath(run_dir))
        m = re.search(r"(yolo[a-z0-9]*[nslmx](?:-(?:seg|cls|pose|obb))?)", run_name, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower()
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
        cur_model = ti.get("model")
        if not isinstance(cur_model, str) or not cur_model.strip():
            ti["model"] = model

    ds = ti.setdefault("dataset", {})
    if not isinstance(ds, dict):
        ds = {}
        ti["dataset"] = ds

    dataset_name = _infer_dataset_name_from_run_dir(run_dir)
    if dataset_name:
        cur_name = ds.get("name")
        if not isinstance(cur_name, str) or not cur_name.strip():
            ds["name"] = dataset_name

    dataset_path = _load_dataset_from_runtime_yaml(run_dir)
    if dataset_path:
        cur_abs = ds.get("path_absolute")
        if not isinstance(cur_abs, str) or not cur_abs.strip():
            ds["path_absolute"] = dataset_path

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


def _try_resolve_dataset_path_string(candidate: str, workspace_root: str | None = None) -> str | None:
    """Resolve dataset root from directory path, dataset yaml, or legacy runtime yaml pointer."""
    if not candidate:
        return None
    p = os.path.abspath(os.path.expanduser(candidate))
    ws = os.path.abspath(os.path.expanduser(workspace_root)) if workspace_root else None

    if os.path.isfile(p) and os.path.basename(p) in {"_runtime_data_train.yaml", "_runtime_data_test.yaml"}:
        try:
            with open(p, "r", encoding="utf-8") as f:
                runtime_payload = yaml.safe_load(f) or {}
        except Exception:
            runtime_payload = {}
        if isinstance(runtime_payload, dict):
            runtime_path = runtime_payload.get("path")
            if isinstance(runtime_path, str) and runtime_path.strip():
                direct = _try_resolve_dataset_path_string(runtime_path.strip(), workspace_root=ws)
                if direct:
                    return direct
                if ws:
                    ds_name = os.path.basename(os.path.normpath(runtime_path.strip()))
                    fallback = os.path.join(WorkspaceLayout(ws).work_datasets, ds_name)
                    if _is_dataset_dir(fallback):
                        return os.path.abspath(fallback)

    if os.path.isfile(p) and str(p).lower().endswith((".yaml", ".yml")):
        root = str(Path(p).resolve().parent)
        if _is_dataset_dir(root):
            return os.path.abspath(root)
        return None
    if _is_dataset_dir(p):
        return p
    return None


def _resolve_dataset_from_train_args_yaml(run_dir: str, workspace_root: str) -> str | None:
    """Use Ultralytics ``args.yaml`` ``data:`` field (written at train time) when metadata/runtime lack paths."""
    payload = _load_train_args_yaml(run_dir)
    raw = payload.get("data")
    if not isinstance(raw, str) or not raw.strip():
        return None
    data_str = raw.strip()
    ws = os.path.abspath(workspace_root)
    rd = os.path.abspath(run_dir)
    rc = results_csv_path(run_dir)
    args_yaml = training_args_yaml_path(run_dir, rc)
    args_dir = os.path.dirname(args_yaml) if os.path.isfile(args_yaml) else None

    candidates: list[str] = []
    if os.path.isabs(data_str):
        candidates.append(data_str)
    else:
        if args_dir:
            candidates.append(os.path.normpath(os.path.join(args_dir, data_str)))
        candidates.append(os.path.normpath(os.path.join(ws, data_str)))
        candidates.append(os.path.normpath(os.path.join(rd, data_str)))

    seen: set[str] = set()
    for c in candidates:
        key = os.path.abspath(os.path.expanduser(c))
        if key in seen:
            continue
        seen.add(key)
        out = _try_resolve_dataset_path_string(key, workspace_root=ws)
        if out:
            return out
    return None


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

    resolved_from_args = _resolve_dataset_from_train_args_yaml(rd, ws)
    if resolved_from_args:
        return resolved_from_args

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
    inference: dict[str, Any] | None = None,
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

    if isinstance(inference, dict) and inference:
        payload["inference"] = {k: v for k, v in inference.items() if v is not None}

    _hydrate_training_info(payload, run_dir)

    _atomic_write_json(metadata_path, payload)


def resume_training_in_run(run_dir: str) -> None:
    ensure_matplotlib_training_runtime(non_interactive=True)
    last_pt = resolve_last_checkpoint_path(run_dir)
    if not last_pt:
        raise FileNotFoundError(f"Resume checkpoint last.pt not found under {run_dir}")
    model = YOLO(last_pt)
    model.train(resume=True)


def format_resume_option(d: RunDiagnosis) -> str:
    base = Path(d.run_dir).name
    status = d.status
    return f"{base} | {status}"
