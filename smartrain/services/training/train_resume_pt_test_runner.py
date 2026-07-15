"""Resume-time Ultralytics PT test (same artifacts as ``smartrain test``)."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from smartrain.core.runtime.mpl_runtime import ensure_matplotlib_training_runtime
from smartrain.core.runtime.run_artifacts import (
    ensure_runtime_layout_for_yaml,
    ensure_runtime_tmp_dir,
    preferred_run_model_path,
)
from smartrain.run_model_contract.gateway import resolve_task_context
from smartrain.services.testing.backends.format_runners import run_ultralytics_backend
from smartrain.services.training.train_runtime_data_yaml_service import build_runtime_data_yaml


def resume_ultralytics_pt_test_runner(
    model_dir: str,
    dataset_path: str,
    training_start_time=None,
    training_end_time=None,
    train_img_size=None,
    val_imgsz=None,
    val_conf=None,
    val_iou=None,
    val_batch=None,
    conf_rec_disable: bool = False,
    conf_rec_beta_recall: float = 2.0,
    conf_rec_beta_precision: float = 0.5,
    conf_rec_fallback: float = 0.25,
    *,
    non_interactive: bool = False,
):
    mpl_rt = ensure_matplotlib_training_runtime(non_interactive=bool(non_interactive))
    data_yaml = build_runtime_data_yaml(
        dataset_path,
        model_dir,
        stage="test",
        ensure_run_layout_cb=ensure_runtime_layout_for_yaml,
        run_tmp_dir_cb=ensure_runtime_tmp_dir,
    )
    weights = preferred_run_model_path(model_dir, ".pt")
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"canonical run model is missing: {weights}")
    try:
        ctx = resolve_task_context(model_dir)
        t_raw = (ctx.task_type or "detect").strip().lower()
    except Exception:
        t_raw = "detect"
    if t_raw in {"classification", "classify", "cls"}:
        task_type = "classification"
    elif t_raw in {"segmentation", "segment", "seg"}:
        task_type = "segmentation"
    else:
        task_type = "detection"

    res = run_ultralytics_backend(
        root_dir=model_dir,
        weights_path=weights,
        dataset_yaml_path=data_yaml,
        format_name="pt",
        imgsz=val_imgsz if val_imgsz is not None else train_img_size,
        val_conf=val_conf,
        val_iou=val_iou,
        val_batch=val_batch,
        conf_rec_disable=bool(conf_rec_disable),
        conf_rec_beta_recall=float(conf_rec_beta_recall),
        conf_rec_beta_precision=float(conf_rec_beta_precision),
        conf_rec_fallback=float(conf_rec_fallback),
        deep_diagnostics=False,
        collect_performance=False,
        perf_warmup_images=5,
        runtime_device=None,
        task_type=task_type,
    )
    if not res.success:
        raise RuntimeError(str(res.error or "run_ultralytics_backend failed"))

    inference_record: dict[str, Any] = {
        "imgsz": val_imgsz if val_imgsz is not None else train_img_size,
        "conf": val_conf,
        "iou": val_iou,
        "batch": val_batch,
        "matplotlib_runtime": mpl_rt.as_dict(),
    }
    if isinstance(res.inference, dict):
        inference_record.update(res.inference)
    return res.test_start_time or datetime.now(), res.test_end_time or datetime.now(), inference_record
