"""Optimal LRP confidence recommendation (Oksuz et al., arXiv:1807.01696).

Computes Localization-Recall-Precision Error over a confidence grid and picks
``argmin_s LRP`` per class and globally. Requires prediction–GT matches
(score + matched IoU); without matches the caller should skip and record a reason.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np

from smartrain.core.runtime.run_artifacts import run_tests_dir
from smartrain.core.training.confidence_recommendation import (
    DEFAULT_FALLBACK_CONFIDENCE,
    write_recommendation_file,
)


def lrp_recommendation_file_path(model_dir: str, split: str) -> str:
    preferred = run_tests_dir(model_dir) / f"lrp_recommendations_{split}.json"
    legacy = os.path.join(model_dir, f"lrp_recommendations_{split}.json")
    if os.path.isfile(legacy) and not preferred.is_file():
        return legacy
    return str(preferred)


def read_lrp_recommendation_file(path: str) -> dict[str, Any] | None:
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


def lrp_error(
    *,
    n_tp: int,
    n_fp: int,
    n_fn: int,
    sum_one_minus_iou: float,
    iou_thr: float = 0.5,
) -> float | None:
    """Compact LRP (Eq. 5): localization + FP + FN, normalized by contributors."""
    denom = int(n_tp) + int(n_fp) + int(n_fn)
    if denom <= 0:
        return None
    tau = float(iou_thr)
    loc_norm = float(sum_one_minus_iou) / max(1e-12, 1.0 - tau)
    return float((loc_norm + float(n_fp) + float(n_fn)) / float(denom))


def _detections_as_arrays(
    detections: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores: list[float] = []
    class_ids: list[int] = []
    ious: list[float] = []
    for det in detections:
        scores.append(float(det["score"]))
        class_ids.append(int(det["class_id"]))
        raw = det.get("iou")
        ious.append(float(raw) if raw is not None else 0.0)
    return (
        np.asarray(scores, dtype=float),
        np.asarray(class_ids, dtype=int),
        np.asarray(ious, dtype=float),
    )


def compute_lrp_at_score(
    detections: Sequence[Mapping[str, Any]],
    gt_count: Mapping[int, int],
    *,
    score_thr: float,
    iou_thr: float = 0.5,
    class_id: int | None = None,
) -> float | None:
    """LRP for detections with ``score >= score_thr``.

    Each detection must already be assigned (greedy AP-style) at ``iou_thr``:
    TP candidates carry matched ``iou >= iou_thr``; FPs carry ``iou < iou_thr``
    (typically 0). Matching is **not** recomputed here.
    """
    if not detections:
        n_gt = int(sum(int(v) for v in gt_count.values())) if class_id is None else int(gt_count.get(class_id, 0))
        if n_gt <= 0:
            return None
        return lrp_error(n_tp=0, n_fp=0, n_fn=n_gt, sum_one_minus_iou=0.0, iou_thr=iou_thr)

    scores, class_ids, ious = _detections_as_arrays(detections)
    keep = scores >= float(score_thr)
    if class_id is not None:
        keep = keep & (class_ids == int(class_id))
    kept_iou = ious[keep]
    tp_mask = kept_iou >= float(iou_thr)
    n_tp = int(np.count_nonzero(tp_mask))
    n_fp = int(kept_iou.size - n_tp)
    if class_id is None:
        n_gt = int(sum(int(v) for v in gt_count.values()))
    else:
        n_gt = int(gt_count.get(int(class_id), 0))
    n_fn = max(0, n_gt - n_tp)
    sum_loc = float(np.sum(1.0 - kept_iou[tp_mask])) if n_tp else 0.0
    return lrp_error(
        n_tp=n_tp,
        n_fp=n_fp,
        n_fn=n_fn,
        sum_one_minus_iou=sum_loc,
        iou_thr=iou_thr,
    )


def _default_conf_grid(scores: Sequence[float], *, n: int = 51) -> np.ndarray:
    if scores:
        lo = 0.0
        hi = 1.0
        return np.linspace(lo, hi, max(2, int(n)), dtype=float)
    return np.linspace(0.0, 1.0, max(2, int(n)), dtype=float)


def optimal_lrp_threshold(
    detections: Sequence[Mapping[str, Any]],
    gt_count: Mapping[int, int],
    *,
    iou_thr: float = 0.5,
    class_id: int | None = None,
    conf_grid: Sequence[float] | None = None,
    fallback_confidence: float = DEFAULT_FALLBACK_CONFIDENCE,
) -> dict[str, Any]:
    """Return ``argmin_s LRP`` for one class or pooled global detections."""
    scores = [float(d["score"]) for d in detections]
    grid = np.asarray(conf_grid if conf_grid is not None else _default_conf_grid(scores), dtype=float)
    best_s = float(fallback_confidence)
    best_lrp: float | None = None
    curve: list[dict[str, Any]] = []
    for s in grid.tolist():
        err = compute_lrp_at_score(
            detections,
            gt_count,
            score_thr=float(s),
            iou_thr=iou_thr,
            class_id=class_id,
        )
        curve.append({"threshold": float(s), "lrp": err})
        if err is None:
            continue
        if best_lrp is None or err < best_lrp - 1e-15:
            best_lrp = float(err)
            best_s = float(s)
    if best_lrp is None:
        return {
            "threshold": float(fallback_confidence),
            "lrp": None,
            "status": "fallback",
            "reason": "lrp_undefined_empty",
            "curve": curve,
        }
    return {
        "threshold": best_s,
        "lrp": best_lrp,
        "status": "ok",
        "reason": None,
        "curve": curve,
    }


def compute_lrp_recommendations(
    detections: Sequence[Mapping[str, Any]] | None,
    gt_count: Mapping[int, int] | None,
    *,
    split: str,
    iou_thr: float = 0.5,
    conf_grid: Sequence[float] | None = None,
    class_names: Mapping[int, str] | None = None,
    fallback_confidence: float = DEFAULT_FALLBACK_CONFIDENCE,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    """Build Optimal LRP payload (separate from A/B/C confidence JSON)."""
    base: dict[str, Any] = {
        "split": split,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "source": "smartrain.lrp",
        "objective": "D",
        "metric": "optimal_lrp",
        "iou_thr": float(iou_thr),
        "citation": "arXiv:1807.01696",
    }
    if skip_reason:
        base["status"] = "skipped"
        base["reason"] = skip_reason
        base["global"] = {
            "threshold": float(fallback_confidence),
            "lrp": None,
            "status": "skipped",
            "reason": skip_reason,
        }
        base["per_class"] = []
        return base
    if detections is None or gt_count is None:
        return compute_lrp_recommendations(
            None,
            None,
            split=split,
            iou_thr=iou_thr,
            fallback_confidence=fallback_confidence,
            skip_reason="prediction_gt_matches_unavailable",
        )

    names = {int(k): str(v) for k, v in (class_names or {}).items()}
    global_opt = optimal_lrp_threshold(
        detections,
        gt_count,
        iou_thr=iou_thr,
        class_id=None,
        conf_grid=conf_grid,
        fallback_confidence=fallback_confidence,
    )
    per_class: list[dict[str, Any]] = []
    for cid in sorted({int(k) for k in gt_count.keys()} | {int(d["class_id"]) for d in detections}):
        opt = optimal_lrp_threshold(
            detections,
            gt_count,
            iou_thr=iou_thr,
            class_id=int(cid),
            conf_grid=conf_grid,
            fallback_confidence=fallback_confidence,
        )
        per_class.append(
            {
                "class_id": int(cid),
                "class_name": names.get(int(cid), f"class_{cid}"),
                "threshold": opt["threshold"],
                "lrp": opt["lrp"],
                "status": opt["status"],
                "reason": opt["reason"],
                "gt_count": int(gt_count.get(int(cid), 0)),
            }
        )
    base["status"] = "ok"
    base["reason"] = None
    base["global"] = {
        "threshold": global_opt["threshold"],
        "lrp": global_opt["lrp"],
        "status": global_opt["status"],
        "reason": global_opt["reason"],
    }
    base["per_class"] = per_class
    return base


def write_lrp_recommendations(
    *,
    model_dir: str,
    split: str,
    payload: dict[str, Any],
) -> str:
    path = lrp_recommendation_file_path(model_dir, split)
    write_recommendation_file(path, payload)
    return path


def maybe_write_lrp_recommendations(
    *,
    model_dir: str,
    split: str,
    compute_lrp: bool,
    detections: Sequence[Mapping[str, Any]] | None = None,
    gt_count: Mapping[int, int] | None = None,
    iou_thr: float = 0.5,
    class_names: Mapping[int, str] | None = None,
    fallback_confidence: float = DEFAULT_FALLBACK_CONFIDENCE,
) -> str | None:
    """Opt-in writer: no-op when ``compute_lrp`` is False (keeps A/B/C contract)."""
    if not compute_lrp:
        return None
    if detections is None or gt_count is None:
        payload = compute_lrp_recommendations(
            None,
            None,
            split=split,
            iou_thr=iou_thr,
            fallback_confidence=fallback_confidence,
            skip_reason="prediction_gt_matches_unavailable",
        )
    else:
        payload = compute_lrp_recommendations(
            detections,
            gt_count,
            split=split,
            iou_thr=iou_thr,
            class_names=class_names,
            fallback_confidence=fallback_confidence,
        )
    return write_lrp_recommendations(model_dir=model_dir, split=split, payload=payload)
