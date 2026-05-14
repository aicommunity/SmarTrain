"""Extract PR curve tensors from Ultralytics validation metrics (no analyze-layer imports)."""

from __future__ import annotations

from typing import Any

import numpy as np


def extract_pr_curve_from_ultralytics_metrics(metrics_obj: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Try to extract all-classes PR curve from Ultralytics metrics object."""
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
            x = np.asarray(item[0], dtype=float)
            y = np.asarray(item[1], dtype=float)
            x_label = str(item[2]) if len(item) > 2 else ""
            y_label = str(item[3]) if len(item) > 3 else ""
            title = str(item[4]) if len(item) > 4 else ""
            marker = f"{x_label} {y_label} {title}".lower()

            if "recall" not in marker or "precision" not in marker:
                continue

            if y.ndim >= 2:
                valid_rows = ~np.all(np.isnan(y), axis=1)
                if bool(np.any(valid_rows)):
                    y = np.nanmean(y[valid_rows], axis=0)
                else:
                    continue
            if x.ndim > 1:
                x = np.ravel(x)
            if y.ndim > 1:
                y = np.ravel(y)

            n = min(len(x), len(y))
            if n == 0:
                continue
            return x[:n], y[:n]
    return None


def extract_pr_curve_per_class_from_ultralytics_metrics(metrics_obj: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Return recall grid and per-class precision curves if available."""
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
            x = np.asarray(item[0], dtype=float)
            y = np.asarray(item[1], dtype=float)
            marker = " ".join(str(v) for v in item[2:5]).lower()
            if "recall" not in marker or "precision" not in marker:
                continue
            if y.ndim < 2:
                continue
            if x.ndim > 1:
                x = np.ravel(x)
            n_points = min(len(x), int(y.shape[-1]))
            if n_points <= 0:
                continue
            y2d = y[:, :n_points] if y.shape[1] >= n_points else y[:n_points, :].T
            if y2d.shape[1] != n_points:
                continue
            return x[:n_points], y2d
    return None
