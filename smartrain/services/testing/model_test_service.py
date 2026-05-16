from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from smartrain.core.runtime.run_artifacts import (
    canonical_run_model_path,
    ensure_run_layout,
    materialize_canonical_run_model,
    run_test_backend_dir,
    run_test_format_dir,
    run_tests_dir,
)
from smartrain.core.runtime.run_artifacts import model_sidecar_metadata_path

import yaml
from smartrain.core.training.confidence_recommendation import read_recommendation_file, recommendations_complete
from smartrain.core.testing.artifact_paths import (
    ALL_TEST_FORMATS,
    INTERNAL_TEST_FORMATS,
    PUBLIC_TEST_FORMATS,
    SUPPORTED_TEST_FORMATS,
    TEST_ARTIFACTS_MANIFEST,
    artifacts_manifest_path_for_write,
    format_metrics_path,
    format_metrics_path_for_split,
    format_metrics_path_for_split_write,
    format_metrics_path_for_write,
    format_recommendation_path,
    format_recommendation_path_for_write,
    format_suffix,
    format_test_dir,
    format_test_dir_for_write,
    load_test_artifacts_manifest,
    normalize_format_name,
    test_artifacts_manifest_path,
)
from smartrain.core.testing.ultralytics_test_contract import rich_files_required_for_format


@dataclass
class TestFormatArtifact:
    format: str
    target_path: str | None
    dataset_yaml: str | None
    backend: str | None
    metrics_csv: str | None
    test_dir: str | None
    confidence_test_json: str | None
    confidence_val_json: str | None
    status: str
    missing: list[str]
    error: str | None = None
    split_status: dict[str, Any] | None = None
    native_debug: dict[str, Any] | None = None
    performance: dict[str, Any] | None = None
    test_system_profile: dict[str, Any] | None = None
    updated_at: str | None = None


@dataclass
class TestArtifactsStatus:
    root_dir: str
    format: str
    metrics_exists: bool
    test_dir_exists: bool
    confidence_test_complete: bool
    confidence_val_complete: bool
    rich_artifacts_complete: bool
    val_metrics_exists: bool
    missing: list[str]

    @property
    def complete(self) -> bool:
        return not self.missing


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass


def _existing_rich_files(root_dir: str, fmt: str) -> list[str]:
    test_dir = format_test_dir(root_dir, fmt)
    if not os.path.isdir(test_dir):
        return []
    required = rich_files_required_for_format(root_dir, fmt)
    return [name for name in required if os.path.isfile(os.path.join(test_dir, name))]


def get_test_artifacts_status(root_dir: str, format_name: str | None = "pt") -> TestArtifactsStatus:
    fmt = normalize_format_name(format_name)
    metrics_csv = format_metrics_path(root_dir, fmt)
    val_metrics_csv = format_metrics_path_for_split(root_dir, "val", fmt)
    test_dir = format_test_dir(root_dir, fmt)
    test_json = format_recommendation_path(root_dir, "test", fmt)
    val_json = format_recommendation_path(root_dir, "val", fmt)

    metrics_exists = os.path.isfile(metrics_csv)
    val_metrics_exists = os.path.isfile(val_metrics_csv)
    test_dir_exists = os.path.isdir(test_dir)
    confidence_test_complete = recommendations_complete(read_recommendation_file(test_json))
    confidence_val_complete = recommendations_complete(read_recommendation_file(val_json))
    existing_rich = _existing_rich_files(root_dir, fmt) if test_dir_exists else []
    required = rich_files_required_for_format(root_dir, fmt) if test_dir_exists else ()
    missing_rich = (
        [n for n in required if not os.path.isfile(os.path.join(test_dir, n))] if test_dir_exists else []
    )
    rich_artifacts_complete = test_dir_exists and not missing_rich
    missing: list[str] = []
    if not metrics_exists:
        missing.append("metrics_csv")
    dataset_yaml_guess = _read_dataset_yaml_from_test_args(root_dir, fmt)
    if not dataset_yaml_guess:
        manifest = load_test_artifacts_manifest(root_dir)
        formats = manifest.get("formats") if isinstance(manifest, dict) else {}
        entry = formats.get(fmt, {}) if isinstance(formats, dict) else {}
        if isinstance(entry, dict):
            dataset_yaml_guess = entry.get("dataset_yaml")
    dataset_yaml_abs = _normalize_compare_path(root_dir, dataset_yaml_guess)
    if fmt == "pt" and dataset_yaml_abs and os.path.isfile(dataset_yaml_abs):
        try:
            payload = yaml.safe_load(Path(dataset_yaml_abs).read_text(encoding="utf-8")) or {}
            if isinstance(payload, dict) and isinstance(payload.get("val"), str) and payload.get("val").strip():
                if not val_metrics_exists:
                    missing.append("val_metrics_csv")
        except Exception:
            missing.append("dataset_yaml_read_failed")
    if not confidence_test_complete:
        missing.append("confidence_test")
    if not confidence_val_complete:
        missing.append("confidence_val")
    if not test_dir_exists:
        missing.append("test_dir")
    elif missing_rich:
        for name in missing_rich:
            missing.append(f"rich_artifact:{name}")
    return TestArtifactsStatus(
        root_dir=os.path.abspath(root_dir),
        format=fmt,
        metrics_exists=metrics_exists,
        test_dir_exists=test_dir_exists,
        confidence_test_complete=confidence_test_complete,
        confidence_val_complete=confidence_val_complete,
        rich_artifacts_complete=rich_artifacts_complete,
        val_metrics_exists=val_metrics_exists,
        missing=missing,
    )


def missing_test_artifacts(root_dir: str, format_name: str | None = "pt") -> list[str]:
    return list(get_test_artifacts_status(root_dir, format_name).missing)


def has_complete_test_artifacts(root_dir: str, format_name: str | None = "pt") -> bool:
    return get_test_artifacts_status(root_dir, format_name).complete


def update_test_artifacts_manifest(
    root_dir: str,
    *,
    format_name: str,
    target_path: str | None,
    dataset_yaml: str | None = None,
    backend: str | None,
    performance: dict[str, Any] | None = None,
    test_system_profile: dict[str, Any] | None = None,
    status: str | None = None,
    error: str | None = None,
    split_status: dict[str, Any] | None = None,
    native_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_run_layout(root_dir)
    fmt = normalize_format_name(format_name)
    snapshot = get_test_artifacts_status(root_dir, fmt)
    record = TestFormatArtifact(
        format=fmt,
        target_path=os.path.relpath(target_path, root_dir) if target_path and os.path.isabs(target_path) else target_path,
        dataset_yaml=os.path.relpath(dataset_yaml, root_dir) if dataset_yaml and os.path.isabs(dataset_yaml) else dataset_yaml,
        backend=backend,
        metrics_csv=os.path.relpath(format_metrics_path_for_write(root_dir, fmt), root_dir) if snapshot.metrics_exists else None,
        test_dir=os.path.relpath(format_test_dir_for_write(root_dir, fmt), root_dir) if snapshot.test_dir_exists else None,
        confidence_test_json=os.path.relpath(format_recommendation_path_for_write(root_dir, "test", fmt), root_dir)
        if snapshot.confidence_test_complete
        else None,
        confidence_val_json=os.path.relpath(format_recommendation_path_for_write(root_dir, "val", fmt), root_dir)
        if snapshot.confidence_val_complete
        else None,
        status=status or ("ok" if snapshot.complete else "incomplete"),
        missing=list(snapshot.missing),
        error=error,
        split_status=split_status if isinstance(split_status, dict) else None,
        native_debug=native_debug if isinstance(native_debug, dict) else None,
        performance=performance if isinstance(performance, dict) else None,
        test_system_profile=test_system_profile if isinstance(test_system_profile, dict) else None,
        updated_at=datetime.now().isoformat(timespec="seconds"),
    )
    payload = load_test_artifacts_manifest(root_dir)
    formats = payload.get("formats")
    if not isinstance(formats, dict):
        formats = {}
        payload["formats"] = formats
    existing = formats.get(fmt)
    if isinstance(existing, dict):
        artifacts = existing.get("artifacts")
        if not isinstance(artifacts, list):
            artifacts = []
        rec_dict = asdict(record)
        updated = False
        for idx, item in enumerate(artifacts):
            if not isinstance(item, dict):
                continue
            if str(item.get("target_path") or "") == str(rec_dict.get("target_path") or ""):
                artifacts[idx] = rec_dict
                updated = True
                break
        if not updated:
            artifacts.append(rec_dict)
        existing.update(rec_dict)
        existing["artifacts"] = artifacts
        if target_path:
            sidecar = model_sidecar_metadata_path(target_path)
            existing["target_metadata_json"] = os.path.relpath(str(sidecar), root_dir) if sidecar.is_file() else None
        formats[fmt] = existing
    else:
        rec_dict = asdict(record)
        rec_dict["artifacts"] = [asdict(record)]
        if target_path:
            sidecar = model_sidecar_metadata_path(target_path)
            rec_dict["target_metadata_json"] = os.path.relpath(str(sidecar), root_dir) if sidecar.is_file() else None
        formats[fmt] = rec_dict
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json_atomic(artifacts_manifest_path_for_write(root_dir), payload)
    return payload


def sync_test_artifacts_manifest(
    root_dir: str,
    *,
    target_by_format: dict[str, str | None] | None = None,
    backend_by_format: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    ensure_run_layout(root_dir)
    payload = load_test_artifacts_manifest(root_dir)
    formats = payload.get("formats")
    if not isinstance(formats, dict):
        formats = {}
        payload["formats"] = formats
    target_by_format = target_by_format or {}
    backend_by_format = backend_by_format or {}
    for fmt in SUPPORTED_TEST_FORMATS:
        status = get_test_artifacts_status(root_dir, fmt)
        existing_entry = formats.get(fmt) if isinstance(formats.get(fmt), dict) else {}
        if not any(
            (
                status.metrics_exists,
                status.test_dir_exists,
                status.confidence_test_complete,
                status.confidence_val_complete,
                fmt in formats,
            )
        ):
            continue
        synced = asdict(
            TestFormatArtifact(
                format=fmt,
                target_path=target_by_format.get(fmt),
                dataset_yaml=None,
                backend=backend_by_format.get(fmt),
                metrics_csv=os.path.relpath(format_metrics_path_for_write(root_dir, fmt), root_dir) if status.metrics_exists else None,
                test_dir=os.path.relpath(format_test_dir_for_write(root_dir, fmt), root_dir) if status.test_dir_exists else None,
                confidence_test_json=os.path.relpath(format_recommendation_path_for_write(root_dir, "test", fmt), root_dir)
                if status.confidence_test_complete
                else None,
                confidence_val_json=os.path.relpath(format_recommendation_path_for_write(root_dir, "val", fmt), root_dir)
                if status.confidence_val_complete
                else None,
                status="ok" if status.complete else "incomplete",
                missing=list(status.missing),
                updated_at=datetime.now().isoformat(timespec="seconds"),
            )
        )
        # Keep rich telemetry fields already collected by test commands.
        for keep_key in (
            "artifacts",
            "error",
            "split_status",
            "native_debug",
            "performance",
            "test_system_profile",
            "target_metadata_json",
        ):
            if existing_entry.get(keep_key) is not None:
                synced[keep_key] = existing_entry.get(keep_key)
        # Prefer existing non-empty values if sync call doesn't provide explicit values.
        if not synced.get("target_path") and existing_entry.get("target_path") is not None:
            synced["target_path"] = existing_entry.get("target_path")
        if not synced.get("backend") and existing_entry.get("backend") is not None:
            synced["backend"] = existing_entry.get("backend")
        if not synced.get("dataset_yaml") and existing_entry.get("dataset_yaml") is not None:
            synced["dataset_yaml"] = existing_entry.get("dataset_yaml")
        formats[fmt] = synced
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json_atomic(artifacts_manifest_path_for_write(root_dir), payload)
    return payload


def _update_run_metadata_after_test(root_dir: str) -> None:
    metadata_path = os.path.join(root_dir, "training_metadata.json")
    if not os.path.isfile(metadata_path):
        return
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return
        payload["test_artifacts_by_format"] = load_test_artifacts_manifest(root_dir).get("formats", {})
        _write_json_atomic(metadata_path, payload)
    except Exception:
        return


def _update_model_manifest_after_test(root_dir: str) -> None:
    manifest_path = os.path.join(root_dir, "model_manifest.json")
    if not os.path.isfile(manifest_path):
        return
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return
        meta_path = os.path.join(root_dir, "training_metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as mf:
                    md = json.load(mf)
                ti = md.get("training_info") if isinstance(md, dict) else None
                if isinstance(ti, dict):
                    tt = ti.get("task_type")
                    if tt and not str(payload.get("task_type") or "").strip():
                        payload["task_type"] = tt
                    prov = ti.get("provider")
                    if isinstance(prov, dict):
                        pid = prov.get("id")
                        if pid and not str(payload.get("backend_type") or "").strip():
                            payload["backend_type"] = str(pid)
            except Exception:
                pass
        payload["test_artifacts_by_format"] = load_test_artifacts_manifest(root_dir).get("formats", {})
        _write_json_atomic(manifest_path, payload)
    except Exception:
        return


def persist_target_test_artifacts_state(
    root_dir: str,
    *,
    format_name: str,
    target_path: str | None,
    dataset_yaml: str | None = None,
    backend: str | None,
    performance: dict[str, Any] | None = None,
    test_system_profile: dict[str, Any] | None = None,
    status: str | None = None,
    error: str | None = None,
    split_status: dict[str, Any] | None = None,
    native_debug: dict[str, Any] | None = None,
) -> None:
    update_test_artifacts_manifest(
        root_dir,
        format_name=format_name,
        target_path=target_path,
        dataset_yaml=dataset_yaml,
        backend=backend,
        performance=performance,
        test_system_profile=test_system_profile,
        status=status,
        error=error,
        split_status=split_status,
        native_debug=native_debug,
    )
    _update_run_metadata_after_test(root_dir)
    _update_model_manifest_after_test(root_dir)
    if (status or "").strip().lower() == "ok":
        from smartrain.adapters.canonical.write.snapshot_hook import maybe_dual_write_canonical_snapshot

        dual_mode = str(os.getenv("SMARTTRAIN_CANONICAL_DUAL_WRITE_MODE", "canonical_only")).strip().lower()
        if dual_mode not in {"canonical_only", "dual_write_strict", "dual_write_best_effort"}:
            dual_mode = "canonical_only"
        legacy_writer = (
            (lambda: (_update_run_metadata_after_test(root_dir), _update_model_manifest_after_test(root_dir)))
            if dual_mode != "canonical_only"
            else None
        )
        maybe_dual_write_canonical_snapshot(
            root_dir,
            status_ok=True,
            legacy_writer=legacy_writer,
            warn_prefix="[WARN]",
        )


def _normalize_compare_path(root_dir: str, value: str | None) -> str | None:
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = Path(root_dir) / p
    try:
        return str(p.resolve())
    except Exception:
        return str(p.absolute())


def _read_dataset_yaml_from_test_args(root_dir: str, format_name: str) -> str | None:
    args_yaml = Path(format_test_dir(root_dir, format_name)) / "args.yaml"
    if not args_yaml.is_file():
        return None
    try:
        payload = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("data")
    return str(value).strip() if value is not None and str(value).strip() else None


def _read_inference_params_from_test_args(root_dir: str, format_name: str) -> dict[str, float | int | None]:
    args_yaml = Path(format_test_dir(root_dir, format_name)) / "args.yaml"
    if not args_yaml.is_file():
        return {"imgsz": None, "conf": None, "iou": None, "batch": None}
    try:
        payload = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"imgsz": None, "conf": None, "iou": None, "batch": None}
    if not isinstance(payload, dict):
        return {"imgsz": None, "conf": None, "iou": None, "batch": None}
    out: dict[str, float | int | None] = {}
    for key in ("imgsz", "conf", "iou", "batch"):
        value = payload.get(key)
        if value is None:
            out[key] = None
        else:
            try:
                out[key] = int(value) if key in {"imgsz", "batch"} else float(value)
            except Exception:
                out[key] = None
    return out


def _has_all_zero_native_metrics(root_dir: str, fmt: str) -> bool:
    metric_cols = ("mAP50-95", "mAP50", "Box-F1", "Box-P", "Box-R")
    for split in ("test", "val"):
        path = format_metrics_path_for_split(root_dir, split, fmt)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            row = rows[0]
            vals: list[float] = []
            for key in metric_cols:
                raw = row.get(key)
                if raw is None or str(raw).strip() == "":
                    vals = []
                    break
                vals.append(float(raw))
            if vals and all(abs(v) <= 1e-12 for v in vals):
                return True
        except Exception:
            continue
    return False


def has_matching_test_artifacts(
    root_dir: str,
    *,
    format_name: str,
    target_path: str,
    dataset_yaml: str,
    imgsz: int | None = None,
    conf: float | None = None,
    iou: float | None = None,
) -> bool:
    fmt = normalize_format_name(format_name)
    if not has_complete_test_artifacts(root_dir, fmt):
        return False
    payload = load_test_artifacts_manifest(root_dir)
    formats = payload.get("formats")
    if not isinstance(formats, dict):
        return False
    entry = formats.get(fmt)
    if not isinstance(entry, dict):
        return False
    expected_target = _normalize_compare_path(root_dir, target_path)
    expected_dataset = _normalize_compare_path(root_dir, dataset_yaml)
    artifacts = entry.get("artifacts")
    selected_entry: dict[str, Any] | None = None
    if isinstance(artifacts, list) and expected_target:
        for item in artifacts:
            if not isinstance(item, dict):
                continue
            item_target = _normalize_compare_path(root_dir, item.get("target_path"))
            if item_target == expected_target:
                selected_entry = item
                break
    if selected_entry is None:
        selected_entry = entry
    if str(selected_entry.get("status") or "").strip().lower() != "ok":
        return False
    recorded_target = _normalize_compare_path(root_dir, selected_entry.get("target_path"))
    recorded_dataset_raw = selected_entry.get("dataset_yaml")
    if not recorded_dataset_raw:
        recorded_dataset_raw = _read_dataset_yaml_from_test_args(root_dir, fmt)
    recorded_dataset = _normalize_compare_path(root_dir, recorded_dataset_raw)
    if not (recorded_target and recorded_dataset and recorded_target == expected_target and recorded_dataset == expected_dataset):
        return False
    recorded = _read_inference_params_from_test_args(root_dir, fmt)
    expected = {
        "imgsz": int(imgsz) if imgsz is not None else None,
        "conf": float(conf) if conf is not None else None,
        "iou": float(iou) if iou is not None else None,
    }
    for key, value in expected.items():
        if value is None:
            continue
        rec = recorded.get(key)
        if rec is None:
            # Legacy/minimal args.yaml: no recorded value — cannot prove mismatch with expected defaults.
            continue
        if rec != value:
            return False
    if fmt in {"engine", "trt"} and _has_all_zero_native_metrics(root_dir, fmt):
        return False
    return True


def complete_missing_test_artifacts(
    run_dir: str,
    *,
    workspace_root: str,
    pt_test_runner: Callable[..., Any],
    pt_test_runner_kwargs: dict[str, Any] | None = None,
    update_metadata_cb: Callable[..., None] | None = None,
) -> bool:
    from smartrain.core.workflow_adapters.training_runtime_api import (
        diagnose_run,
        resolve_dataset_path_for_resume,
    )

    root_dir = os.path.abspath(run_dir)
    materialized = materialize_canonical_run_model(root_dir, ext=".pt", move=True, normalize_metadata=True)
    canonical_pt = str(materialized) if materialized is not None else canonical_run_model_path(root_dir, ".pt")
    if has_complete_test_artifacts(root_dir, "pt"):
        persist_target_test_artifacts_state(
            root_dir,
            format_name="pt",
            target_path=canonical_pt,
            backend="ultralytics",
            status="ok",
        )
        return False

    dataset_path = resolve_dataset_path_for_resume(root_dir, workspace_root)
    if not dataset_path:
        raise RuntimeError(
            "Cannot resolve dataset path for test stage. Expected valid dataset in runtime yaml/metadata/workspace datasets catalog."
        )

    kwargs = dict(pt_test_runner_kwargs or {})
    inference_payload: dict[str, Any] | None = None
    runner_result = pt_test_runner(root_dir, dataset_path, **kwargs)
    if isinstance(runner_result, (tuple, list)) and len(runner_result) >= 3:
        maybe_inf = runner_result[2]
        if isinstance(maybe_inf, dict):
            inference_payload = maybe_inf
    elif isinstance(runner_result, dict):
        inference_payload = runner_result
    persist_target_test_artifacts_state(
        root_dir,
        format_name="pt",
        target_path=canonical_pt,
        backend="ultralytics",
        status="ok",
    )
    if update_metadata_cb is not None:
        try:
            update_metadata_cb(
                root_dir,
                success=True,
                error=None,
                diagnosis=diagnose_run(root_dir),
                inference=inference_payload,
            )
        except TypeError:
            # Backward compatibility for callbacks that do not accept `inference`.
            update_metadata_cb(
                root_dir,
                success=True,
                error=None,
                diagnosis=diagnose_run(root_dir),
            )
    return True


def resolve_root_dir_for_target(path: str) -> str:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        return str(candidate)
    return str(candidate.parent)
