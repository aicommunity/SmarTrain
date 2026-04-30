from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from smartrain.run_artifacts import (
    canonical_run_model_path,
    ensure_run_layout,
    materialize_canonical_run_model,
    run_test_backend_dir,
    run_test_format_dir,
    run_tests_dir,
)
from smartrain.run_artifacts import model_sidecar_metadata_path

import yaml
from smartrain.confidence_recommendation import read_recommendation_file, recommendations_complete

TEST_ARTIFACTS_MANIFEST = "test_artifacts_manifest.json"
PUBLIC_TEST_FORMATS = ("pt", "onnx", "engine", "trt")
INTERNAL_TEST_FORMATS = ("pt_uni",)
SUPPORTED_TEST_FORMATS = PUBLIC_TEST_FORMATS
ALL_TEST_FORMATS = PUBLIC_TEST_FORMATS + INTERNAL_TEST_FORMATS
_RICH_TEST_FILES = (
    "args.yaml",
    "pr.csv",
    "pr_per_class.csv",
    "BoxF1_curve.png",
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "confusion_matrix.png",
)


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


def _normalize_format_name(format_name: str | None) -> str:
    raw = str(format_name or "pt").strip().lower()
    if raw in {"", "best", "base"}:
        return "pt"
    if raw in {"tensorrt-engine"}:
        return "engine"
    if raw in {"tensorrt-trt"}:
        return "trt"
    if raw not in ALL_TEST_FORMATS:
        raise ValueError(f"Unsupported test format: {format_name}")
    return raw


def format_suffix(format_name: str | None) -> str:
    fmt = _normalize_format_name(format_name)
    return "" if fmt == "pt" else f"_{fmt}"


def format_test_dir(root_dir: str, format_name: str | None = "pt") -> str:
    return _format_test_dir(root_dir, format_name, prefer_legacy_for_read=True)


def format_test_dir_for_write(root_dir: str, format_name: str | None = "pt") -> str:
    return _format_test_dir(root_dir, format_name, prefer_legacy_for_read=False)


def _format_test_dir(root_dir: str, format_name: str | None, *, prefer_legacy_for_read: bool) -> str:
    fmt = _normalize_format_name(format_name)
    preferred = run_test_backend_dir(root_dir, "ultralytics") if fmt == "pt" else run_test_format_dir(root_dir, fmt)
    legacy = os.path.join(root_dir, f"test{format_suffix(fmt)}")
    if prefer_legacy_for_read and os.path.isdir(legacy) and not preferred.exists():
        return legacy
    return str(preferred)


def format_metrics_path(root_dir: str, format_name: str | None = "pt") -> str:
    return _format_metrics_path(root_dir, format_name, prefer_legacy_for_read=True)


def format_metrics_path_for_write(root_dir: str, format_name: str | None = "pt") -> str:
    return _format_metrics_path(root_dir, format_name, prefer_legacy_for_read=False)


def _format_metrics_path(root_dir: str, format_name: str | None, *, prefer_legacy_for_read: bool) -> str:
    fmt = _normalize_format_name(format_name)
    preferred = run_tests_dir(root_dir) / f"test_metrics{format_suffix(fmt)}.csv"
    legacy = os.path.join(root_dir, f"test_metrics{format_suffix(fmt)}.csv")
    if prefer_legacy_for_read and os.path.isfile(legacy) and not preferred.is_file():
        return legacy
    return str(preferred)


def format_metrics_path_for_split(root_dir: str, split: str, format_name: str | None = "pt") -> str:
    return _format_metrics_path_for_split(root_dir, split, format_name, prefer_legacy_for_read=True)


def format_metrics_path_for_split_write(root_dir: str, split: str, format_name: str | None = "pt") -> str:
    return _format_metrics_path_for_split(root_dir, split, format_name, prefer_legacy_for_read=False)


def _format_metrics_path_for_split(
    root_dir: str,
    split: str,
    format_name: str | None,
    *,
    prefer_legacy_for_read: bool,
) -> str:
    split_name = str(split).strip().lower()
    if split_name == "test":
        return _format_metrics_path(root_dir, format_name, prefer_legacy_for_read=prefer_legacy_for_read)
    if split_name == "val":
        fmt = _normalize_format_name(format_name)
        preferred = run_tests_dir(root_dir) / f"val_metrics{format_suffix(fmt)}.csv"
        legacy = os.path.join(root_dir, f"val_metrics{format_suffix(fmt)}.csv")
        if prefer_legacy_for_read and os.path.isfile(legacy) and not preferred.is_file():
            return legacy
        return str(preferred)
    raise ValueError(f"Unsupported split: {split}")


def format_recommendation_path(root_dir: str, split: str, format_name: str | None = "pt") -> str:
    return _format_recommendation_path(root_dir, split, format_name, prefer_legacy_for_read=True)


def format_recommendation_path_for_write(root_dir: str, split: str, format_name: str | None = "pt") -> str:
    return _format_recommendation_path(root_dir, split, format_name, prefer_legacy_for_read=False)


def _format_recommendation_path(
    root_dir: str,
    split: str,
    format_name: str | None,
    *,
    prefer_legacy_for_read: bool,
) -> str:
    split_name = str(split).strip().lower()
    if split_name not in {"test", "val"}:
        raise ValueError(f"Unsupported split: {split}")
    fmt = _normalize_format_name(format_name)
    preferred = run_tests_dir(root_dir) / f"confidence_recommendations_{split_name}{format_suffix(fmt)}.json"
    legacy = os.path.join(root_dir, f"confidence_recommendations_{split_name}{format_suffix(fmt)}.json")
    if prefer_legacy_for_read and os.path.isfile(legacy) and not preferred.is_file():
        return legacy
    return str(preferred)


def test_artifacts_manifest_path(root_dir: str) -> str:
    return _test_artifacts_manifest_path(root_dir, prefer_legacy_for_read=True)


def artifacts_manifest_path_for_write(root_dir: str) -> str:
    return _test_artifacts_manifest_path(root_dir, prefer_legacy_for_read=False)


def _test_artifacts_manifest_path(root_dir: str, *, prefer_legacy_for_read: bool) -> str:
    preferred = run_tests_dir(root_dir) / TEST_ARTIFACTS_MANIFEST
    legacy = os.path.join(root_dir, TEST_ARTIFACTS_MANIFEST)
    if prefer_legacy_for_read and os.path.isfile(legacy) and not preferred.is_file():
        return legacy
    return str(preferred)


def load_test_artifacts_manifest(root_dir: str) -> dict[str, Any]:
    path = test_artifacts_manifest_path(root_dir)
    if not os.path.isfile(path):
        return {"formats": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {"formats": {}}
    except Exception:
        return {"formats": {}}


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


def _existing_rich_files(test_dir: str) -> list[str]:
    out: list[str] = []
    for name in _RICH_TEST_FILES:
        if os.path.exists(os.path.join(test_dir, name)):
            out.append(name)
    return out


def get_test_artifacts_status(root_dir: str, format_name: str | None = "pt") -> TestArtifactsStatus:
    fmt = _normalize_format_name(format_name)
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
    existing_rich = _existing_rich_files(test_dir) if test_dir_exists else []
    rich_artifacts_complete = test_dir_exists and len(existing_rich) >= 3
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
            pass
    if not confidence_test_complete:
        missing.append("confidence_test")
    if not confidence_val_complete:
        missing.append("confidence_val")
    if not test_dir_exists:
        missing.append("test_dir")
    elif not rich_artifacts_complete:
        missing.append("rich_artifacts")
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
) -> dict[str, Any]:
    ensure_run_layout(root_dir)
    fmt = _normalize_format_name(format_name)
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
        formats[fmt] = asdict(
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
    )
    _update_run_metadata_after_test(root_dir)
    _update_model_manifest_after_test(root_dir)


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
    fmt = _normalize_format_name(format_name)
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
        if recorded.get(key) != value:
            return False
    return True


def complete_missing_test_artifacts(
    run_dir: str,
    *,
    workspace_root: str,
    pt_test_runner: Callable[..., tuple[Any, Any, dict[str, Any]]],
    pt_test_runner_kwargs: dict[str, Any] | None = None,
    update_metadata_cb: Callable[..., None] | None = None,
) -> bool:
    from smartrain.train_resume import diagnose_run, resolve_dataset_path_for_resume

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
    pt_test_runner(root_dir, dataset_path, **kwargs)
    persist_target_test_artifacts_state(
        root_dir,
        format_name="pt",
        target_path=canonical_pt,
        backend="ultralytics",
        status="ok",
    )
    if update_metadata_cb is not None:
        update_metadata_cb(root_dir, success=True, error=None, diagnosis=diagnose_run(root_dir))
    return True


def resolve_root_dir_for_target(path: str) -> str:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        return str(candidate)
    return str(candidate.parent)
