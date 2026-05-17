from __future__ import annotations

import json
import os
from glob import glob
from typing import Any

import pandas as pd
import yaml

from smartrain.core.workflow_adapters.analyze_runtime_api import (
    ensure_format_compare_index,
    load_test_artifacts_manifest,
    read_metrics_by_format_for_split,
    read_metrics_by_format_for_split_artifacts,
    read_test_performance_by_format_artifacts,
    read_test_system_profile_by_format_artifacts,
)
from smartrain.core.runtime.run_artifacts import resolve_run_model
from smartrain.orchestrators.unified_gateway import load_metrics as unified_load_metrics

METRIC_AGG_COLUMNS = ("mAP50-95", "mAP50", "Box-F1", "Box-P", "Box-R")


def write_format_compare_artifacts(session_root: str, run_dirs: list[str]) -> dict[str, str] | None:
    backend_fallback = {
        "pt": "ultralytics",
        "pt_uni": "unified_pt",
        "onnx": "onnxruntime",
        "engine": "tensorrt",
        "trt": "tensorrt",
    }
    ext_by_format = {
        "pt": ".pt",
        "pt_uni": ".pt",
        "onnx": ".onnx",
        "engine": ".engine",
        "trt": ".trt",
    }

    def _has_model_artifact(run_dir: str, fmt: str, entry: dict[str, Any]) -> bool:
        target = entry.get("target_path")
        if isinstance(target, str) and target.strip():
            candidate = target if os.path.isabs(target) else os.path.join(run_dir, target)
            if os.path.isfile(candidate):
                return True
        artifacts = entry.get("artifacts")
        if isinstance(artifacts, list):
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                target_item = item.get("target_path")
                if not isinstance(target_item, str) or not target_item.strip():
                    continue
                candidate = target_item if os.path.isabs(target_item) else os.path.join(run_dir, target_item)
                if os.path.isfile(candidate):
                    return True
        if fmt in {"pt", "pt_uni"}:
            return resolve_run_model(run_dir, ".pt") is not None
        ext = ext_by_format.get(fmt)
        if not ext:
            return False
        return any(os.path.isfile(p) for p in glob(os.path.join(run_dir, "**", f"*{ext}"), recursive=True))

    def _pick_target_path(entry: dict[str, Any]) -> str | None:
        target = entry.get("target_path")
        if isinstance(target, str) and target.strip():
            return target
        artifacts = entry.get("artifacts")
        if isinstance(artifacts, list):
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                t = item.get("target_path")
                if isinstance(t, str) and t.strip():
                    return t
        return None

    def _format_alias_prefix(fmt: str) -> str:
        return {
            "pt": "PT",
            "pt_uni": "PTUNI",
            "onnx": "ONNX",
            "engine": "ENGINE",
            "trt": "TRT",
        }.get(fmt, str(fmt).upper())

    def _iter_entry_variants(
        run_dir: str,
        fmt: str,
        entry: dict[str, Any],
        split_metrics: list[dict[str, str]],
        split_name: str,
    ) -> list[dict[str, Any]]:
        def _resolve_metrics_candidate(path_value: str | None) -> str | None:
            raw = str(path_value or "").strip()
            if not raw:
                return None
            candidate = os.path.abspath(os.path.join(run_dir, raw)) if not os.path.isabs(raw) else raw
            if os.path.isfile(candidate):
                return candidate
            base = os.path.basename(raw)
            if base:
                migrated = os.path.join(run_dir, "tests", base)
                if os.path.isfile(migrated):
                    return os.path.abspath(migrated)
            return candidate

        variants: list[dict[str, Any]] = []
        artifacts = entry.get("artifacts")
        if isinstance(artifacts, list):
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                target_rel = str(item.get("target_path") or "").strip()
                target_abs = (
                    os.path.abspath(os.path.join(run_dir, target_rel))
                    if target_rel and not os.path.isabs(target_rel)
                    else (target_rel or None)
                )
                metrics_rel = str(item.get("metrics_csv") or "").strip()
                metrics_abs = _resolve_metrics_candidate(metrics_rel)
                matched = None
                preferred_split_metrics = list(split_metrics)
                split_token = f"{split_name}_metrics"
                split_specific = [
                    rec
                    for rec in split_metrics
                    if split_token in os.path.basename(str(rec.get("metrics_path") or "")).lower()
                ]
                if split_specific:
                    preferred_split_metrics = split_specific
                if target_abs:
                    for rec in preferred_split_metrics:
                        if rec.get("target_path") == target_abs:
                            matched = rec
                            break
                if matched is None and metrics_abs:
                    for rec in preferred_split_metrics:
                        if rec.get("metrics_path") == metrics_abs:
                            matched = rec
                            break
                if matched is None and preferred_split_metrics and (not target_abs or fmt in {"pt", "pt_uni"}):
                    matched = preferred_split_metrics[0]
                variants.append(
                    {
                        "target_path": target_rel or None,
                        "metrics_path": (matched or {}).get("metrics_path") or metrics_abs,
                        "status": item.get("status", entry.get("status")),
                        "error": item.get("error", entry.get("error")),
                        "backend": item.get("backend", entry.get("backend")),
                        "performance": item.get("performance"),
                    }
                )
        if not variants:
            fallback_metrics = split_metrics[0].get("metrics_path") if split_metrics else None
            variants.append(
                {
                    "target_path": _pick_target_path(entry),
                    "metrics_path": fallback_metrics,
                    "status": entry.get("status"),
                    "error": entry.get("error"),
                    "backend": entry.get("backend"),
                    "performance": entry.get("performance"),
                }
            )
        deduped: dict[tuple[str, str], dict[str, Any]] = {}
        for item in variants:
            key = (str(item.get("target_path") or ""), str(item.get("metrics_path") or ""))
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = item
                continue
            cur_has_metrics = bool(str(item.get("metrics_path") or "").strip())
            prev_has_metrics = bool(str(existing.get("metrics_path") or "").strip())
            if cur_has_metrics and not prev_has_metrics:
                deduped[key] = item
        variants = list(deduped.values())
        if len(variants) > 1:
            with_target = [v for v in variants if str(v.get("target_path") or "").strip()]
            if with_target:
                variants = with_target
        if len(variants) > 1:
            existing_target_variants = []
            for v in variants:
                t = str(v.get("target_path") or "").strip()
                if not t:
                    continue
                candidate = t if os.path.isabs(t) else os.path.join(run_dir, t)
                if os.path.isfile(candidate):
                    existing_target_variants.append(v)
            if existing_target_variants:
                variants = existing_target_variants
        variants.sort(key=lambda v: (0 if str(v.get("metrics_path") or "").strip() else 1, str(v.get("target_path") or "")))
        return variants

    def _format_from_metrics_path(metrics_path: str) -> str:
        base = os.path.basename(str(metrics_path or "")).lower()
        if "pt_uni" in base:
            return "pt_uni"
        if "_onnx" in base:
            return "onnx"
        if "_engine" in base:
            return "engine"
        if "_trt" in base:
            return "trt"
        return "pt"

    def _read_eval_args(run_dir: str, fmt: str) -> dict[str, Any]:
        if fmt == "pt":
            args_yaml = os.path.join(run_dir, "tests", "test-ultralytics", "args.yaml")
            if not os.path.isfile(args_yaml):
                args_yaml = os.path.join(run_dir, "test-ultralytics", "args.yaml")
            if not os.path.isfile(args_yaml):
                args_yaml = os.path.join(run_dir, "test", "args.yaml")
        else:
            args_yaml = os.path.join(run_dir, "tests", f"test_{fmt}", "args.yaml")
            if not os.path.isfile(args_yaml):
                args_yaml = os.path.join(run_dir, f"test_{fmt}", "args.yaml")
        if not os.path.isfile(args_yaml):
            if fmt != "pt":
                return {}
            metadata_path = os.path.join(run_dir, "training_metadata.json")
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}
            inf = payload.get("inference") if isinstance(payload, dict) else {}
            if isinstance(inf, dict) and inf:
                return {
                    "imgsz": inf.get("imgsz"),
                    "conf": inf.get("conf", 0.001 if inf.get("conf") is None else inf.get("conf")),
                    "iou": inf.get("iou"),
                    "inference_source": "ultralytics_model_val",
                    "gt_source": "ultralytics_validator",
                    "nms_profile": "ultralytics_validator_multilabel",
                }
            ti = payload.get("training_info") if isinstance(payload, dict) else {}
            if isinstance(ti, dict):
                ut = ti.get("ultralytics_train") if isinstance(ti.get("ultralytics_train"), dict) else {}
                hp = ti.get("hyperparameters") if isinstance(ti.get("hyperparameters"), dict) else {}
                imgsz = ut.get("imgsz")
                if imgsz is None:
                    imgsz = hp.get("image_size")
                if imgsz is not None:
                    return {
                        "imgsz": imgsz,
                        "conf": 0.001,
                        "iou": 0.7,
                        "inference_source": "ultralytics_model_val",
                        "gt_source": "ultralytics_validator",
                        "nms_profile": "ultralytics_validator_multilabel",
                    }
            return {}
        try:
            with open(args_yaml, "r", encoding="utf-8") as f:
                payload = yaml.safe_load(f) or {}
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _read_metric_row(metrics_path: str | None) -> dict[str, Any]:
        if not metrics_path or not os.path.isfile(metrics_path):
            return {}
        try:
            mdf = pd.read_csv(metrics_path)
            if len(mdf) == 0:
                return {}
            mdf.columns = [str(c).strip() for c in mdf.columns]
            if "Class" in mdf.columns:
                cls = mdf["Class"].astype(str).str.strip().str.lower()
                all_mask = cls.eq("all")
                if bool(all_mask.any()):
                    return dict(mdf.loc[all_mask].iloc[0].to_dict())
            if "Class" in mdf.columns and len(mdf) > 1:
                out: dict[str, Any] = {}
                for col in METRIC_AGG_COLUMNS:
                    if col in mdf.columns:
                        out[col] = pd.to_numeric(mdf[col], errors="coerce").mean()
                if out:
                    out["Class"] = "all"
                    return out
            return dict(mdf.iloc[0].to_dict())
        except Exception:
            return {}

    def _metrics_path_matches_split(metrics_path: str | None, split_name: str) -> bool:
        if not metrics_path:
            return False
        base = os.path.basename(str(metrics_path)).lower()
        token = f"{split_name}_metrics"
        return token in base

    def _normalize_issue_reason(reason: str) -> tuple[str, str]:
        raw = str(reason or "").strip()
        if raw.startswith("[") and "]" in raw:
            maybe_code = raw[1 : raw.index("]")].strip().lower()
            detail = raw[raw.index("]") + 1 :].strip()
            if maybe_code:
                return maybe_code, detail or raw
        lower = raw.lower()
        if "timeout" in lower:
            return "timeout", raw
        if "out of memory" in lower or "bfc_arena" in lower or "cudamalloc" in lower:
            return "oom_gpu", raw
        if "terminated by signal" in lower:
            return "signal_terminated", raw
        if "runtime_exception" in lower or "onnxruntimeerror" in lower:
            return "runtime_exception", raw
        if "session init" in lower or "inferencesession" in lower:
            return "init_session_failed", raw
        if "missing" in lower:
            return "missing_artifact", raw
        return "unknown", raw

    def _is_invalid_zero_metrics(fmt: str, metric_row: dict[str, Any]) -> bool:
        if fmt not in {"engine", "trt"}:
            return False
        vals: list[float] = []
        for col in METRIC_AGG_COLUMNS:
            raw_v = metric_row.get(col)
            if raw_v is None or (isinstance(raw_v, float) and pd.isna(raw_v)):
                return False
            try:
                vals.append(float(raw_v))
            except (TypeError, ValueError):
                return False
        return bool(vals) and all(abs(v) <= 1e-12 for v in vals)

    def _perf_context_for_variant(run_dir: str, fmt: str, target_path: Any) -> dict[str, Any]:
        profile_map = read_test_system_profile_by_format_artifacts(run_dir)
        records = profile_map.get(fmt) if isinstance(profile_map, dict) else None
        if not isinstance(records, list) or not records:
            return {}
        target_abs = ""
        if isinstance(target_path, str) and target_path.strip():
            target_abs = os.path.abspath(os.path.join(run_dir, target_path))
        target_name = os.path.basename(target_abs) if target_abs else ""
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rec_target = str(rec.get("target_path") or "")
            rec_name = os.path.basename(rec_target) if rec_target else ""
            profile = rec.get("test_system_profile")
            if not isinstance(profile, dict) or not profile:
                continue
            if target_abs and rec_target and os.path.abspath(rec_target) == target_abs:
                return profile
            if target_name and rec_name and rec_name == target_name:
                return profile
        return {}

    def _extract_perf_details(
        perf: dict[str, Any], eval_args: dict[str, Any], profile: dict[str, Any]
    ) -> dict[str, Any]:
        lat_all = perf.get("latency_ms") if isinstance(perf.get("latency_ms"), dict) else {}
        all_stats = lat_all.get("all") if isinstance(lat_all.get("all"), dict) else {}
        steady_stats = lat_all.get("steady") if isinstance(lat_all.get("steady"), dict) else {}
        breakdown = perf.get("breakdown_ms") if isinstance(perf.get("breakdown_ms"), dict) else {}

        def _stage(*names: str) -> dict[str, Any]:
            for name in names:
                candidate = breakdown.get(name)
                if isinstance(candidate, dict):
                    return candidate
            return {}

        preprocess = _stage("preprocess", "preprocess_ms")
        inference = _stage("infer", "inference", "infer_ms")
        postprocess = _stage("postprocess", "decode_nms", "decode_nms_ms")
        total = _stage("total", "total_ms", "infer_total_only_ms")
        io_load = _stage("io_load_ms")
        diag_alloc = _stage("diagnostics_alloc_ms")
        diag_h2d = _stage("diagnostics_h2d_ms")
        diag_exec = _stage("diagnostics_execute_ms")
        diag_d2h = _stage("diagnostics_d2h_ms")
        diagnostics = perf.get("diagnostics_overhead") if isinstance(perf.get("diagnostics_overhead"), dict) else {}

        runtime = profile.get("runtime") if isinstance(profile.get("runtime"), dict) else {}
        device = (
            perf.get("eval_device")
            if perf.get("eval_device") is not None
            else (runtime.get("device") if runtime.get("device") is not None else eval_args.get("device"))
        )
        batch_raw = perf.get("eval_batch") if perf.get("eval_batch") is not None else eval_args.get("batch")
        try:
            batch_val = int(batch_raw) if batch_raw is not None else None
        except (TypeError, ValueError):
            batch_val = None
        if batch_val is None:
            batch_val = 1
        if device is None and batch_val is not None:
            device = "0"

        infer_p50 = inference.get("p50") if inference.get("p50") is not None else inference.get("mean")
        infer_p95 = inference.get("p95") if inference.get("p95") is not None else inference.get("p90")
        try:
            infer_ms = float(inference.get("mean")) if inference.get("mean") is not None else None
        except (TypeError, ValueError):
            infer_ms = None
        pure_infer_throughput = (1000.0 / infer_ms) if (infer_ms is not None and infer_ms > 0) else None
        throughput_value = pure_infer_throughput if pure_infer_throughput is not None else perf.get("throughput_img_s")
        return {
            "throughput_img_s": throughput_value,
            "latency_p50_ms": infer_p50 if infer_p50 is not None else steady_stats.get("p50", all_stats.get("p50")),
            "latency_p95_ms": infer_p95 if infer_p95 is not None else steady_stats.get("p95", all_stats.get("p95")),
            "perf_preprocess_ms_per_frame": preprocess.get("mean"),
            "perf_inference_ms_per_frame": inference.get("mean"),
            "perf_postprocess_ms_per_frame": postprocess.get("mean"),
            "perf_total_ms_per_frame": total.get("mean", steady_stats.get("mean", all_stats.get("mean"))),
            "perf_warmup_images": perf.get("warmup_images"),
            "perf_sample_count": perf.get("images_total"),
            "perf_batch": batch_val,
            "perf_device": device,
            "perf_io_load_ms_per_frame": io_load.get("mean"),
            "perf_diag_alloc_ms_per_frame": diag_alloc.get("mean"),
            "perf_diag_h2d_ms_per_frame": diag_h2d.get("mean"),
            "perf_diag_execute_ms_per_frame": diag_exec.get("mean"),
            "perf_diag_d2h_ms_per_frame": diag_d2h.get("mean"),
            "perf_diag_session_init_ms": diagnostics.get("session_init_ms"),
            "perf_diag_engine_init_ms": diagnostics.get("engine_init_ms"),
            "perf_diag_worker_wall_ms": diagnostics.get("worker_wall_ms"),
            "perf_diag_retries_count": diagnostics.get("retries_count"),
            "perf_diag_retry_sleep_ms": diagnostics.get("retry_sleep_ms"),
            "perf_diag_provider_switched_to_cpu": diagnostics.get("provider_switched_to_cpu"),
        }

    def _resolve_perf_and_reason(
        run_dir: str,
        fmt: str,
        target_path: Any,
        perf_candidate: Any,
        entry: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        perf = perf_candidate if isinstance(perf_candidate, dict) else {}
        if perf:
            return perf, "perf_present"
        perf_map = read_test_performance_by_format_artifacts(run_dir)
        records = perf_map.get(fmt) if isinstance(perf_map, dict) else None
        if not isinstance(records, list) or not records:
            artifacts = entry.get("artifacts") if isinstance(entry, dict) else None
            if isinstance(artifacts, list) and artifacts:
                return {}, "perf_not_collected_for_target"
            return {}, "perf_missing_manifest_entry"
        target_abs = ""
        if isinstance(target_path, str) and target_path.strip():
            target_abs = os.path.abspath(os.path.join(run_dir, target_path))
        target_name = os.path.basename(target_abs) if target_abs else ""
        target_stem = os.path.splitext(target_name)[0] if target_name else ""
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rec_perf = rec.get("performance")
            if not isinstance(rec_perf, dict) or not rec_perf:
                continue
            rec_target = str(rec.get("target_path") or "")
            rec_name = os.path.basename(rec_target) if rec_target else ""
            rec_stem = os.path.splitext(rec_name)[0] if rec_name else ""
            if target_abs and rec_target and os.path.abspath(rec_target) == target_abs:
                return rec_perf, "perf_present"
            if target_name and rec_name and rec_name == target_name:
                return rec_perf, "perf_present"
            if target_stem and rec_stem and rec_stem == target_stem:
                return rec_perf, "perf_present"
        if target_abs:
            return {}, "perf_target_mismatch_legacy_variant"
        return {}, "perf_not_collected_for_target"

    def _build_format_rows(
        split_name: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        eval_rows: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        for run_dir in run_dirs:
            run_name = os.path.basename(run_dir.rstrip(os.sep))
            # unified_gateway.load_metrics can trigger legacy-to-unified
            # migration with file moves; recompute legacy paths after migration.
            metrics_ref_by_raw_path: dict[str, dict[str, Any]] = {}
            canonical_metrics_by_format: dict[str, list[dict[str, str]]] = {}
            for ref in unified_load_metrics(run_dir, source_kind="run", split=split_name):
                merged = dict(ref.primary_metrics or {})
                merged.update(dict(ref.secondary_metrics or {}))
                raw_abs = os.path.abspath(str(ref.raw_path))
                metrics_ref_by_raw_path[raw_abs] = merged
                fmt = _format_from_metrics_path(raw_abs)
                canonical_metrics_by_format.setdefault(fmt, []).append(
                    {"metrics_path": raw_abs, "target_path": ""}
                )

            manifest = load_test_artifacts_manifest(run_dir)
            formats_meta = manifest.get("formats") if isinstance(manifest, dict) else {}
            metrics_paths = read_metrics_by_format_for_split(run_dir, split_name)
            metrics_artifacts = read_metrics_by_format_for_split_artifacts(run_dir, split_name)
            for fmt in ("pt", "onnx", "engine", "trt"):
                entry = formats_meta.get(fmt, {}) if isinstance(formats_meta, dict) else {}
                if not isinstance(entry, dict):
                    entry = {}
                fmt_metrics = list(canonical_metrics_by_format.get(fmt) or [])
                metrics_read_policy = "unified_gateway"
                if not fmt_metrics:
                    fmt_metrics = list(metrics_artifacts.get(fmt) or [])
                    if fmt_metrics:
                        metrics_read_policy = "legacy_artifacts_fallback"
                if not fmt_metrics and metrics_paths.get(fmt):
                    fmt_metrics = [{"metrics_path": os.path.abspath(str(metrics_paths[fmt])), "target_path": ""}]
                    metrics_read_policy = "legacy_path_fallback"
                variants = _iter_entry_variants(run_dir, fmt, entry, fmt_metrics, split_name)
                if len(variants) > 1:
                    with_metrics = [v for v in variants if str(v.get("metrics_path") or "").strip()]
                    if with_metrics:
                        variants = with_metrics
                if not variants and not _has_model_artifact(run_dir, fmt, entry):
                    continue
                eval_args = _read_eval_args(run_dir, fmt)
                has_any_metrics_variant = False
                for _v in variants:
                    _mp = str(_v.get("metrics_path") or "").strip()
                    if _mp and os.path.isfile(_mp):
                        has_any_metrics_variant = True
                        break
                for var in variants:
                    metrics_path = str(var.get("metrics_path") or "").strip() or None
                    metrics_exists = bool(metrics_path and os.path.isfile(metrics_path))
                    if split_name == "val" and metrics_exists and not _metrics_path_matches_split(metrics_path, split_name):
                        metrics_exists = False
                    if not metrics_exists and not _has_model_artifact(run_dir, fmt, entry):
                        continue
                    status_raw = str(var.get("status") or "").strip()
                    err_raw = str(var.get("error") or "").strip()
                    status_lower = status_raw.lower()
                    has_explicit_failure = bool(err_raw) or status_lower in {
                        "failed",
                        "error",
                        "timeout",
                        "terminated",
                        "unavailable",
                    }
                    if not metrics_exists:
                        if split_name == "val" and not has_explicit_failure:
                            continue
                        if split_name != "val" and not has_explicit_failure:
                            continue
                    metric_row = metrics_ref_by_raw_path.get(os.path.abspath(str(metrics_path or "")), {}) if metrics_path else {}
                    invalid_zero_metrics = metrics_exists and _is_invalid_zero_metrics(fmt, metric_row)
                    row: dict[str, Any] = {
                        "run_dir": run_dir,
                        "run_name": run_name,
                        "split": split_name,
                        "format": fmt,
                        "backend_status": var.get("backend"),
                        "target_path": var.get("target_path"),
                        "metrics_source": os.path.relpath(metrics_path, run_dir) if metrics_exists and metrics_path else None,
                        "inference_source": eval_args.get("inference_source"),
                        "gt_source": eval_args.get("gt_source"),
                        "nms_profile": eval_args.get("nms_profile"),
                        "metrics_read_policy": metrics_read_policy,
                        "mAP50-95": None if invalid_zero_metrics else metric_row.get("mAP50-95"),
                        "mAP50": None if invalid_zero_metrics else metric_row.get("mAP50"),
                        "Box-F1": None if invalid_zero_metrics else metric_row.get("Box-F1"),
                        "Box-P": None if invalid_zero_metrics else metric_row.get("Box-P"),
                        "Box-R": None if invalid_zero_metrics else metric_row.get("Box-R"),
                    }
                    perf, perf_reason = _resolve_perf_and_reason(run_dir, fmt, var.get("target_path"), var.get("performance"), entry)
                    profile = _perf_context_for_variant(run_dir, fmt, var.get("target_path"))
                    row.update(_extract_perf_details(perf, eval_args, profile))
                    row["performance_status"] = "ok" if isinstance(perf, dict) and len(perf) > 0 else "performance_not_collected"
                    row["performance_reason"] = perf_reason
                    try:
                        thr = float(row["throughput_img_s"]) if row.get("throughput_img_s") is not None else None
                    except (TypeError, ValueError):
                        thr = None
                    try:
                        p50 = float(row["latency_p50_ms"]) if row.get("latency_p50_ms") is not None else None
                    except (TypeError, ValueError):
                        p50 = None
                    row["avg_inference_fps"] = thr
                    row["avg_inference_ms_per_frame"] = p50 if p50 is not None else ((1000.0 / thr) if thr and thr > 0 else None)
                    eval_rows.append(
                        {
                            "run_dir": run_dir,
                            "run_name": run_name,
                            "split": split_name,
                            "format": fmt,
                            "target_path": var.get("target_path"),
                            "eval_imgsz": eval_args.get("imgsz"),
                            "eval_conf": eval_args.get("conf"),
                            "eval_iou": eval_args.get("iou"),
                            "inference_source": eval_args.get("inference_source"),
                            "gt_source": eval_args.get("gt_source"),
                            "nms_profile": eval_args.get("nms_profile"),
                        }
                    )
                    if metrics_exists:
                        if not row.get("backend_status"):
                            row["backend_status"] = backend_fallback.get(fmt)
                        if invalid_zero_metrics:
                            issues.append(
                                {
                                    "run_name": run_name,
                                    "split": split_name,
                                    "format": fmt,
                                    "target_path": var.get("target_path"),
                                    "status": str(var.get("status") or "ok"),
                                    "reason": "metrics are all zeros; treated as invalid native evaluation output",
                                    "reason_code": "invalid_metrics",
                                }
                            )
                    else:
                        if status_raw or err_raw:
                            if has_any_metrics_variant:
                                continue
                            reason_code, reason_detail = _normalize_issue_reason(err_raw or "metrics missing")
                            issues.append(
                                {
                                    "run_name": run_name,
                                    "split": split_name,
                                    "format": fmt,
                                    "target_path": var.get("target_path"),
                                    "status": status_raw or "unknown",
                                    "reason": reason_detail,
                                    "reason_code": reason_code,
                                }
                            )
                    rows.append(row)
                    sources.append(
                        {
                            "run_dir": run_dir,
                            "run_name": run_name,
                            "split": split_name,
                            "format": fmt,
                            "target_path": var.get("target_path"),
                            "metrics_source": row.get("metrics_source"),
                            "inference_source": row.get("inference_source"),
                            "gt_source": row.get("gt_source"),
                            "nms_profile": row.get("nms_profile"),
                            "metrics_read_policy": row.get("metrics_read_policy"),
                        }
                    )
        return rows, sources, eval_rows, issues

    def _build_pt_uni_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        eval_rows: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        for split_name in ("test", "val"):
            for run_dir in run_dirs:
                run_name = os.path.basename(run_dir.rstrip(os.sep))
                metrics_ref_by_raw_path: dict[str, dict[str, Any]] = {}
                canonical_metrics_by_format: dict[str, list[dict[str, str]]] = {}
                for ref in unified_load_metrics(run_dir, source_kind="run", split=split_name):
                    merged = dict(ref.primary_metrics or {})
                    merged.update(dict(ref.secondary_metrics or {}))
                    raw_abs = os.path.abspath(str(ref.raw_path))
                    metrics_ref_by_raw_path[raw_abs] = merged
                    fmt = _format_from_metrics_path(raw_abs)
                    canonical_metrics_by_format.setdefault(fmt, []).append(
                        {"metrics_path": raw_abs, "target_path": ""}
                    )

                manifest = load_test_artifacts_manifest(run_dir)
                formats_meta = manifest.get("formats") if isinstance(manifest, dict) else {}
                metrics_paths = read_metrics_by_format_for_split(run_dir, split_name, include_internal=True)
                metrics_artifacts = read_metrics_by_format_for_split_artifacts(run_dir, split_name, include_internal=True)
                for fmt in ("pt", "pt_uni"):
                    entry = formats_meta.get(fmt, {}) if isinstance(formats_meta, dict) else {}
                    if not isinstance(entry, dict):
                        entry = {}
                    fmt_metrics = list(canonical_metrics_by_format.get(fmt) or [])
                    metrics_read_policy = "unified_gateway"
                    if not fmt_metrics:
                        fmt_metrics = list(metrics_artifacts.get(fmt) or [])
                        if fmt_metrics:
                            metrics_read_policy = "legacy_artifacts_fallback"
                    if not fmt_metrics and metrics_paths.get(fmt):
                        fmt_metrics = [{"metrics_path": os.path.abspath(str(metrics_paths[fmt])), "target_path": ""}]
                        metrics_read_policy = "legacy_path_fallback"
                    variants = _iter_entry_variants(run_dir, fmt, entry, fmt_metrics, split_name)
                    if len(variants) > 1:
                        with_metrics = [v for v in variants if str(v.get("metrics_path") or "").strip()]
                        if with_metrics:
                            variants = with_metrics
                    eval_args = _read_eval_args(run_dir, fmt)
                    has_any_metrics_variant = False
                    for _v in variants:
                        _mp = str(_v.get("metrics_path") or "").strip()
                        if _mp and os.path.isfile(_mp):
                            has_any_metrics_variant = True
                            break
                    for var in variants:
                        metrics_path = str(var.get("metrics_path") or "").strip() or None
                        metrics_exists = bool(metrics_path and os.path.isfile(metrics_path))
                        if split_name == "val" and metrics_exists and not _metrics_path_matches_split(metrics_path, split_name):
                            metrics_exists = False
                        if not metrics_exists and not _has_model_artifact(run_dir, fmt, entry):
                            continue
                        status_raw = str(var.get("status") or "").strip()
                        err_raw = str(var.get("error") or "").strip()
                        status_lower = status_raw.lower()
                        has_explicit_failure = bool(err_raw) or status_lower in {
                            "failed",
                            "error",
                            "timeout",
                            "terminated",
                            "unavailable",
                        }
                        if not metrics_exists:
                            if split_name == "val" and not has_explicit_failure:
                                continue
                            if split_name != "val" and not has_explicit_failure:
                                continue
                        metric_row = (
                            metrics_ref_by_raw_path.get(os.path.abspath(str(metrics_path or "")), {})
                            if metrics_path
                            else {}
                        )
                        row: dict[str, Any] = {
                            "run_dir": run_dir,
                            "run_name": run_name,
                            "split": split_name,
                            "format": fmt,
                            "backend_status": var.get("backend"),
                            "target_path": var.get("target_path"),
                            "metrics_source": os.path.relpath(metrics_path, run_dir) if metrics_exists and metrics_path else None,
                            "inference_source": eval_args.get("inference_source"),
                            "gt_source": eval_args.get("gt_source"),
                            "nms_profile": eval_args.get("nms_profile"),
                            "metrics_read_policy": metrics_read_policy,
                            "mAP50-95": metric_row.get("mAP50-95"),
                            "mAP50": metric_row.get("mAP50"),
                            "Box-F1": metric_row.get("Box-F1"),
                            "Box-P": metric_row.get("Box-P"),
                            "Box-R": metric_row.get("Box-R"),
                        }
                        perf, perf_reason = _resolve_perf_and_reason(run_dir, fmt, var.get("target_path"), var.get("performance"), entry)
                        profile = _perf_context_for_variant(run_dir, fmt, var.get("target_path"))
                        row.update(_extract_perf_details(perf, eval_args, profile))
                        row["performance_status"] = "ok" if isinstance(perf, dict) and len(perf) > 0 else "performance_not_collected"
                        row["performance_reason"] = perf_reason
                        try:
                            thr = float(row["throughput_img_s"]) if row.get("throughput_img_s") is not None else None
                        except (TypeError, ValueError):
                            thr = None
                        try:
                            p50 = float(row["latency_p50_ms"]) if row.get("latency_p50_ms") is not None else None
                        except (TypeError, ValueError):
                            p50 = None
                        row["avg_inference_fps"] = thr
                        row["avg_inference_ms_per_frame"] = p50 if p50 is not None else ((1000.0 / thr) if thr and thr > 0 else None)
                        eval_rows.append(
                            {
                                "run_dir": run_dir,
                                "run_name": run_name,
                                "split": split_name,
                                "format": fmt,
                                "target_path": var.get("target_path"),
                                "eval_imgsz": eval_args.get("imgsz"),
                                "eval_conf": eval_args.get("conf"),
                                "eval_iou": eval_args.get("iou"),
                                "inference_source": eval_args.get("inference_source"),
                                "gt_source": eval_args.get("gt_source"),
                                "nms_profile": eval_args.get("nms_profile"),
                            }
                        )
                        if metrics_exists:
                            if not row.get("backend_status"):
                                row["backend_status"] = backend_fallback.get(fmt)
                        else:
                            if status_raw or err_raw:
                                if has_any_metrics_variant:
                                    continue
                                reason_code, reason_detail = _normalize_issue_reason(err_raw or "metrics missing")
                                issues.append(
                                    {
                                        "run_name": run_name,
                                        "split": split_name,
                                        "format": fmt,
                                        "target_path": var.get("target_path"),
                                        "status": status_raw or "unknown",
                                        "reason": reason_detail,
                                        "reason_code": reason_code,
                                    }
                                )
                        rows.append(row)
                        sources.append(
                            {
                                "run_dir": run_dir,
                                "run_name": run_name,
                                "split": split_name,
                                "format": fmt,
                                "target_path": var.get("target_path"),
                                "metrics_source": row.get("metrics_source"),
                                "inference_source": row.get("inference_source"),
                                "gt_source": row.get("gt_source"),
                                "nms_profile": row.get("nms_profile"),
                                "metrics_read_policy": row.get("metrics_read_policy"),
                            }
                        )
        return rows, sources, eval_rows, issues

    test_rows, test_sources, test_eval_rows, test_issues = _build_format_rows("test")
    val_rows, val_sources, val_eval_rows, val_issues = _build_format_rows("val")
    pt_uni_rows, pt_uni_sources, pt_uni_eval_rows, pt_uni_issues = _build_pt_uni_rows()
    if not test_rows and not val_rows and not pt_uni_rows:
        return None
    out_dir = os.path.join(session_root, "artifacts", "format_compare")
    os.makedirs(out_dir, exist_ok=True)
    out: dict[str, str] = {}
    all_rows = test_rows + val_rows + pt_uni_rows
    alias_legend: list[dict[str, str]] = []
    alias_counters: dict[str, int] = {}
    for row in sorted(
        all_rows,
        key=lambda r: (str(r.get("format") or ""), str(r.get("run_name") or ""), str(r.get("target_path") or "")),
    ):
        fmt = str(row.get("format") or "")
        prefix = _format_alias_prefix(fmt)
        alias_counters[prefix] = int(alias_counters.get(prefix, 0)) + 1
        alias = f"{prefix}{alias_counters[prefix]}"
        row["alias"] = alias
        alias_legend.append(
            {
                "alias": alias,
                "format": fmt,
                "run_name": str(row.get("run_name") or ""),
                "target_path": str(row.get("target_path") or ""),
            }
        )
    if test_rows:
        out_csv = os.path.join(out_dir, "format_metrics_compare_test.csv")
        pd.DataFrame(test_rows).to_csv(out_csv, index=False, encoding="utf-8")
        out["test_csv"] = os.path.relpath(out_csv, session_root)
        perf_csv = os.path.join(out_dir, "format_performance_compare_test.csv")
        pd.DataFrame(test_rows).to_csv(perf_csv, index=False, encoding="utf-8")
        out["perf_test_csv"] = os.path.relpath(perf_csv, session_root)
    if val_rows:
        out_csv = os.path.join(out_dir, "format_metrics_compare_val.csv")
        pd.DataFrame(val_rows).to_csv(out_csv, index=False, encoding="utf-8")
        out["val_csv"] = os.path.relpath(out_csv, session_root)
    if pt_uni_rows:
        out_csv = os.path.join(out_dir, "format_metrics_compare_pt_uni.csv")
        pd.DataFrame(pt_uni_rows).to_csv(out_csv, index=False, encoding="utf-8")
        out["pt_uni_csv"] = os.path.relpath(out_csv, session_root)
    eval_rows = test_eval_rows + val_eval_rows + pt_uni_eval_rows
    alias_by_key = {
        (str(r.get("run_name") or ""), str(r.get("split") or ""), str(r.get("format") or ""), str(r.get("target_path") or "")): str(
            r.get("alias") or ""
        )
        for r in all_rows
    }
    for er in eval_rows:
        er["alias"] = alias_by_key.get(
            (
                str(er.get("run_name") or ""),
                str(er.get("split") or ""),
                str(er.get("format") or ""),
                str(er.get("target_path") or ""),
            ),
            "",
        )
    if eval_rows:
        eval_csv = os.path.join(out_dir, "format_eval_settings.csv")
        pd.DataFrame(eval_rows).drop_duplicates().to_csv(eval_csv, index=False, encoding="utf-8")
        out["eval_csv"] = os.path.relpath(eval_csv, session_root)
    if alias_legend:
        alias_csv = os.path.join(out_dir, "format_alias_legend.csv")
        pd.DataFrame(alias_legend).to_csv(alias_csv, index=False, encoding="utf-8")
        out["alias_legend_csv"] = os.path.relpath(alias_csv, session_root)
    issues = test_issues + val_issues + pt_uni_issues
    if issues:
        deduped_issues: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            key = (
                str(issue.get("run_name") or ""),
                str(issue.get("split") or ""),
                str(issue.get("format") or ""),
                str(issue.get("reason_code") or ""),
            )
            existing = deduped_issues.get(key)
            if existing is None:
                deduped_issues[key] = issue
                continue
            cur_target = str(issue.get("target_path") or "").strip()
            prev_target = str(existing.get("target_path") or "").strip()
            cur_failed = str(issue.get("status") or "").strip().lower() in {"failed", "unavailable"}
            prev_failed = str(existing.get("status") or "").strip().lower() in {"failed", "unavailable"}
            if (cur_target and not prev_target) or (prev_failed and not cur_failed):
                deduped_issues[key] = issue
        issues = list(deduped_issues.values())
    if issues:
        issues_json = os.path.join(out_dir, "format_compare_issues.json")
        with open(issues_json, "w", encoding="utf-8") as f:
            json.dump(issues, f, ensure_ascii=False, indent=2)
        out["issues_json"] = os.path.relpath(issues_json, session_root)
    out_sources = os.path.join(out_dir, "format_metrics_sources.json")
    with open(out_sources, "w", encoding="utf-8") as f:
        json.dump(test_sources + val_sources + pt_uni_sources, f, ensure_ascii=False, indent=2)
    if "csv" not in out:
        out["csv"] = str(out.get("test_csv") or out.get("val_csv") or out.get("pt_uni_csv") or "")
    return ensure_format_compare_index(out)
