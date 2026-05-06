from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np
from smartrain.core.runtime.run_artifacts import run_tests_dir


DEFAULT_FALLBACK_CONFIDENCE = 0.25


def _as_1d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.asarray([], dtype=float)
    if arr.ndim > 1:
        arr = np.ravel(arr)
    return arr


def _as_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim > 2:
        if arr.shape[0] > 0:
            arr = arr.reshape(arr.shape[0], -1)
        else:
            return np.asarray([[]], dtype=float)
    return arr


def _safe_nanmean_axis0(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim < 2:
        arr = _as_2d(arr)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    if arr.shape[-1] == 0:
        return np.asarray([], dtype=float)
    valid_rows = ~np.all(np.isnan(arr), axis=1)
    if not bool(np.any(valid_rows)):
        return np.full(arr.shape[-1], np.nan, dtype=float)
    return np.nanmean(arr[valid_rows], axis=0)


def _extract_curve_map(metrics_obj: Any) -> dict[str, tuple[np.ndarray, np.ndarray | np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray | np.ndarray]] = {}
    sources = [metrics_obj, getattr(metrics_obj, "box", None)]
    for src in sources:
        if src is None:
            continue
        curves = getattr(src, "curves_results", None)
        if not curves:
            continue
        for item in curves:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            x = _as_1d(item[0])
            y_raw = np.asarray(item[1], dtype=float)
            marker = " ".join(str(v) for v in item[2:5]).lower()
            if "confidence" not in marker:
                continue
            if "f1" in marker:
                out["f1"] = (x, y_raw)
            elif "precision" in marker:
                out["precision"] = (x, y_raw)
            elif "recall" in marker:
                out["recall"] = (x, y_raw)
    return out


def _normalize_confidence_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray | np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    required = {"f1", "precision", "recall"}
    if not required.issubset(set(curves.keys())):
        return None

    x_f1, y_f1_raw = curves["f1"]
    x_p, y_p_raw = curves["precision"]
    x_r, y_r_raw = curves["recall"]

    n = min(len(x_f1), len(x_p), len(x_r))
    if n <= 0:
        return None

    conf = x_f1[:n]
    p2d = _as_2d(y_p_raw)
    r2d = _as_2d(y_r_raw)
    f1_2d = _as_2d(y_f1_raw)

    n_points = min(n, p2d.shape[-1], r2d.shape[-1], f1_2d.shape[-1])
    if n_points <= 0:
        return None
    conf = conf[:n_points]
    p2d = p2d[:, :n_points]
    r2d = r2d[:, :n_points]
    f1_2d = f1_2d[:, :n_points]

    p_global = _safe_nanmean_axis0(p2d)
    r_global = _safe_nanmean_axis0(r2d)
    _f1_global = _safe_nanmean_axis0(f1_2d)
    return conf, p2d, r2d, f1_2d, p_global, r_global


def _safe_name_map(metrics_obj: Any) -> dict[int, str]:
    names = getattr(metrics_obj, "names", None)
    if isinstance(names, dict):
        out: dict[int, str] = {}
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def _compute_fbeta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    beta2 = float(beta) * float(beta)
    den = (beta2 * precision) + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        score = (1.0 + beta2) * (precision * recall) / den
    score[~np.isfinite(score)] = np.nan
    return score


def _best_idx(metric: np.ndarray) -> int | None:
    if metric.size == 0:
        return None
    if np.all(np.isnan(metric)):
        return None
    return int(np.nanargmax(metric))


def _build_objective_payload(
    objective: str,
    beta: float,
    conf: np.ndarray,
    p2d: np.ndarray,
    r2d: np.ndarray,
    p_global: np.ndarray,
    r_global: np.ndarray,
    class_names: dict[int, str],
    fallback_confidence: float,
) -> dict[str, Any]:
    fbeta_global = _compute_fbeta(p_global, r_global, beta)
    gidx = _best_idx(fbeta_global)
    if gidx is None:
        global_payload = {
            "threshold": float(fallback_confidence),
            "metric_value": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "status": "fallback",
            "reason": "global_curve_unavailable",
        }
    else:
        f1_g = _compute_fbeta(p_global, r_global, 1.0)
        global_payload = {
            "threshold": float(conf[gidx]),
            "metric_value": float(fbeta_global[gidx]),
            "precision": float(p_global[gidx]),
            "recall": float(r_global[gidx]),
            "f1": float(f1_g[gidx]) if np.isfinite(f1_g[gidx]) else None,
            "status": "ok",
            "reason": None,
        }

    rows: list[dict[str, Any]] = []
    class_count = int(p2d.shape[0])
    for class_id in range(class_count):
        pc = p2d[class_id]
        rc = r2d[class_id]
        fbeta = _compute_fbeta(pc, rc, beta)
        cidx = _best_idx(fbeta)
        if cidx is None:
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, f"class_{class_id}"),
                    "threshold": float(fallback_confidence),
                    "metric_value": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "status": "fallback",
                    "reason": "class_curve_unavailable",
                    "support_instances": None,
                }
            )
            continue
        f1_c = _compute_fbeta(pc, rc, 1.0)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, f"class_{class_id}"),
                "threshold": float(conf[cidx]),
                "metric_value": float(fbeta[cidx]),
                "precision": float(pc[cidx]),
                "recall": float(rc[cidx]),
                "f1": float(f1_c[cidx]) if np.isfinite(f1_c[cidx]) else None,
                "status": "ok",
                "reason": None,
                "support_instances": None,
            }
        )
    return {
        "objective": objective,
        "beta": float(beta),
        "global": global_payload,
        "per_class": rows,
    }


def compute_confidence_recommendations(
    metrics_obj: Any,
    *,
    split: str,
    beta_recall: float = 2.0,
    beta_precision: float = 0.5,
    fallback_confidence: float = DEFAULT_FALLBACK_CONFIDENCE,
) -> dict[str, Any]:
    class_names = _safe_name_map(metrics_obj)
    curves = _extract_curve_map(metrics_obj)
    normalized = _normalize_confidence_curves(curves)
    base = {
        "split": split,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "source": "ultralytics.val.curves_results",
        "objectives": {},
    }
    if normalized is None:
        for objective, beta in (("A", 1.0), ("B", beta_recall), ("C", beta_precision)):
            base["objectives"][objective] = {
                "objective": objective,
                "beta": float(beta),
                "global": {
                    "threshold": float(fallback_confidence),
                    "metric_value": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "status": "fallback",
                    "reason": "confidence_curves_not_available",
                },
                "per_class": [],
            }
        base["status"] = "fallback"
        base["reason"] = "confidence_curves_not_available"
        return base

    conf, p2d, r2d, _f1_2d, p_global, r_global = normalized
    base["objectives"]["A"] = _build_objective_payload(
        "A",
        1.0,
        conf,
        p2d,
        r2d,
        p_global,
        r_global,
        class_names,
        fallback_confidence,
    )
    base["objectives"]["B"] = _build_objective_payload(
        "B",
        beta_recall,
        conf,
        p2d,
        r2d,
        p_global,
        r_global,
        class_names,
        fallback_confidence,
    )
    base["objectives"]["C"] = _build_objective_payload(
        "C",
        beta_precision,
        conf,
        p2d,
        r2d,
        p_global,
        r_global,
        class_names,
        fallback_confidence,
    )
    base["status"] = "ok"
    base["reason"] = None
    return base


def recommendation_file_path(model_dir: str, split: str) -> str:
    preferred = run_tests_dir(model_dir) / f"confidence_recommendations_{split}.json"
    legacy = os.path.join(model_dir, f"confidence_recommendations_{split}.json")
    if os.path.isfile(legacy) and not preferred.is_file():
        return legacy
    return str(preferred)


def write_recommendation_file(path: str, payload: dict[str, Any]) -> None:
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


def read_recommendation_file(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def recommendations_complete(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    objectives = payload.get("objectives")
    if not isinstance(objectives, dict):
        return False
    for key in ("A", "B", "C"):
        item = objectives.get(key)
        if not isinstance(item, dict):
            return False
        g = item.get("global")
        if not isinstance(g, dict):
            return False
        if g.get("threshold") is None:
            return False
    return True


def write_not_available_recommendations(
    *,
    model_dir: str,
    split: str,
    reason: str,
    beta_recall: float = 2.0,
    beta_precision: float = 0.5,
    fallback_confidence: float = DEFAULT_FALLBACK_CONFIDENCE,
) -> str:
    payload = {
        "split": split,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "source": "smartrain.fallback",
        "status": "not_available",
        "reason": reason,
        "objectives": {
            "A": {
                "objective": "A",
                "beta": 1.0,
                "global": {"threshold": float(fallback_confidence), "status": "not_available", "reason": reason},
                "per_class": [],
            },
            "B": {
                "objective": "B",
                "beta": float(beta_recall),
                "global": {"threshold": float(fallback_confidence), "status": "not_available", "reason": reason},
                "per_class": [],
            },
            "C": {
                "objective": "C",
                "beta": float(beta_precision),
                "global": {"threshold": float(fallback_confidence), "status": "not_available", "reason": reason},
                "per_class": [],
            },
        },
    }
    path = recommendation_file_path(model_dir, split)
    write_recommendation_file(path, payload)
    return path
