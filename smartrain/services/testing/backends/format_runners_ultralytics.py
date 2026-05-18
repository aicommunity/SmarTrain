"""Ultralytics YOLO val test backend runner."""

from __future__ import annotations

from smartrain.services.testing.backends import format_runners_support as _support
from smartrain.services.testing.backends.format_runners_support import (
    Any,
    BackendRunResult,
    _Pred,
    _collect_gt,
    _collect_test_system_profile,
    _ensure_confidence_recommendations_for_explicit_artifact,
    _finalize_ultralytics_pt_test_dir,
    _load_names,
    _release_cuda_memory_best_effort,
    _save_metrics_csv_for_format,
    _split_images_from_yaml,
    _ultralytics_val_task_kw,
    _write_deep_diagnostics_artifacts,
    _write_perf_artifact,
    best_effort_prune_runs_detect_near_run,
    normalize_ultralytics_run_layout,
    datetime,
    ensure_run_layout,
    normalize_eval_params,
    np,
    persist_target_test_artifacts_state,
    prune_empty_sidecar_dirs,
    run_tests_dir,
    shutil,
    sys,
    tqdm,
    ultralytics_sidecar_dir,
    YOLO,
)


def run_ultralytics_backend(
    *,
    root_dir: str,
    weights_path: str,
    dataset_yaml_path: str,
    format_name: str,
    imgsz: int | None = None,
    val_conf: float | None = None,
    val_iou: float | None = None,
    val_batch: int | None = None,
    conf_rec_disable: bool = False,
    conf_rec_beta_recall: float = 2.0,
    conf_rec_beta_precision: float = 0.5,
    conf_rec_fallback: float = 0.25,
    deep_diagnostics: bool = False,
    collect_performance: bool = False,
    perf_warmup_images: int = 5,
    runtime_device: str | None = None,
    task_type: str | None = None,
) -> BackendRunResult:
    def _ultralytics_perf_payload_from_result(
        val_result: Any, *, duration_s: float, warmup_images: int, images_count: int | None = None
    ) -> dict[str, Any]:
        speed = getattr(val_result, "speed", None)
        speed_map = speed if isinstance(speed, dict) else {}

        def _as_float(v: Any) -> float | None:
            try:
                if v is None:
                    return None
                return float(v)
            except (TypeError, ValueError):
                return None

        preprocess_ms = _as_float(speed_map.get("preprocess"))
        infer_ms = _as_float(speed_map.get("inference"))
        postprocess_ms = _as_float(speed_map.get("postprocess"))
        total_ms = _as_float(speed_map.get("total"))
        if total_ms is None:
            parts = [x for x in (preprocess_ms, infer_ms, postprocess_ms) if x is not None]
            total_ms = float(sum(parts)) if parts else None

        throughput = None
        if infer_ms is not None and infer_ms > 0:
            throughput = 1000.0 / infer_ms
        elif total_ms is not None and total_ms > 0:
            throughput = 1000.0 / total_ms

        count_guess = int(images_count) if isinstance(images_count, int) and images_count > 0 else None
        for attr in ("seen", "nt_per_image", "nt_per_class"):
            if isinstance(images_count, int) and images_count > 0:
                break
            raw = getattr(val_result, attr, None)
            try:
                if raw is None:
                    continue
                if hasattr(raw, "sum"):
                    count_guess = int(raw.sum())
                elif isinstance(raw, (list, tuple)):
                    count_guess = int(sum(int(x) for x in raw))
                else:
                    count_guess = int(raw)
            except Exception:
                count_guess = None
            if count_guess and count_guess > 0:
                break

        def _stats(val_ms: float | None) -> dict[str, Any]:
            if val_ms is None:
                return {}
            return {
                "count": int(count_guess or 0),
                "mean": float(val_ms),
                "p50": float(val_ms),
                "p90": float(val_ms),
                "p95": float(val_ms),
                "min": float(val_ms),
                "max": float(val_ms),
                "std": 0.0,
            }

        breakdown: dict[str, Any] = {}
        if preprocess_ms is not None:
            breakdown["preprocess_ms"] = _stats(preprocess_ms)
        if infer_ms is not None:
            breakdown["infer_ms"] = _stats(infer_ms)
        if postprocess_ms is not None:
            breakdown["decode_nms_ms"] = _stats(postprocess_ms)
        if total_ms is not None:
            breakdown["infer_total_only_ms"] = _stats(total_ms)

        latency_stats = _stats(infer_ms if infer_ms is not None else total_ms)
        return {
            "images_total": int(count_guess) if isinstance(count_guess, int) and count_guess > 0 else None,
            "warmup_images": int(max(0, warmup_images)),
            "duration_s": float(max(0.0, duration_s)),
            "throughput_img_s": float(throughput) if throughput is not None else 0.0,
            "latency_ms": {"all": latency_stats, "steady": latency_stats},
            "breakdown_ms": breakdown,
            "infer_total_only": True,
            "source": "ultralytics_speed_dict",
            "eval_batch": int(val_batch) if val_batch is not None else None,
            "eval_device": str(runtime_device) if runtime_device else None,
        }

    test_start_time = datetime.now()
    model = _support.YOLO(weights_path)
    val_kwargs = {
        "data": dataset_yaml_path,
        "split": "test",
        "project": str(run_tests_dir(root_dir)),
        "name": "test-ultralytics",
        "exist_ok": True,
        "plots": True,
        "save": True,
    }
    if imgsz is not None:
        val_kwargs["imgsz"] = imgsz
    if val_conf is not None:
        val_kwargs["conf"] = val_conf
    if val_iou is not None:
        val_kwargs["iou"] = val_iou
    if val_batch is not None:
        val_kwargs["batch"] = int(val_batch)
    if runtime_device is not None and str(runtime_device).strip():
        val_kwargs["device"] = str(runtime_device).strip()
    val_kwargs.update(_support._ultralytics_val_task_kw(task_type))
    try:
        ensure_run_layout(root_dir)
        test_image_count = len(_support._split_images_from_yaml(dataset_yaml_path, "test", 0))
        result = model.val(**val_kwargs)
        _support._save_metrics_csv_for_format(result, root_dir, format_name)
        _support._finalize_ultralytics_pt_test_dir(
            root_dir=root_dir,
            format_name=format_name,
            result=result,
            weights_path=weights_path,
            dataset_yaml_path=dataset_yaml_path,
            imgsz=imgsz,
            val_conf=val_conf,
            val_iou=val_iou,
            val_batch=val_batch,
        )
        if not conf_rec_disable:
            _support._ensure_confidence_recommendations_for_explicit_artifact(
                model=model,
                primary_test_result=result,
                root_dir=root_dir,
                format_name=format_name,
                data_yaml=dataset_yaml_path,
                imgsz=imgsz,
                val_conf=val_conf,
                val_iou=val_iou,
                val_batch=val_batch,
                beta_recall=conf_rec_beta_recall,
                beta_precision=conf_rec_beta_precision,
                fallback_confidence=conf_rec_fallback,
            )
        if deep_diagnostics:
            eval_params = normalize_eval_params(imgsz=imgsz, conf=val_conf, iou=val_iou)
            input_hw = (int(eval_params["imgsz"]), int(eval_params["imgsz"]))
            conf_thr = float(eval_params["conf"])
            iou_thr = float(eval_params["iou"])
            names = _support._load_names(dataset_yaml_path)
            pred_proj = ultralytics_sidecar_dir(root_dir, ".ultralytics_predict_scratch")
            pred_common = {
                "save": False,
                "project": pred_proj,
                "name": "deep-diagnostics",
                "exist_ok": True,
            }

            # Deep diagnostics are optional, but when enabled they must be produced for test and val.
            for split in ("test", "val"):
                try:
                    gt_rows_split, _bgv_split, image_paths_split = _support._collect_gt(dataset_yaml_path, split)
                except Exception as exc:
                    if split == "test":
                        raise
                    print(f"[WARN] {format_name}: deep-diagnostics could not collect GT for split={split}: {exc}")
                    continue

                # Batched prediction for deep diagnostics can spike memory.
                # We chunk the input to keep peak RSS low and avoid OOM-killer.
                preds_split: list[_Pred] = []
                predict_chunk_size = 10

                def _append_preds_from_results(results_iter: Any, *, chunk_paths: list[str], chunk_start_idx: int) -> None:
                    # idx->image_path mapping relies on Ultralytics preserving input order.
                    for idx, r0 in tqdm(
                        enumerate(results_iter),
                        desc=f"{format_name}:deep_{split}",
                        unit="img",
                        file=sys.stdout,
                        total=len(chunk_paths),
                    ):
                        image_path = (
                            chunk_paths[idx]
                            if idx < len(chunk_paths)
                            else str(getattr(r0, "path", ""))
                        )
                        boxes = getattr(r0, "boxes", None)
                        if boxes is None:
                            del r0
                            continue
                        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
                        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
                        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
                        for b, c, k in zip(xyxy, confs, clss):
                            preds_split.append(
                                _Pred(
                                    image_path=image_path,
                                    cls_id=int(k),
                                    conf=float(c),
                                    x1=float(b[0]),
                                    y1=float(b[1]),
                                    x2=float(b[2]),
                                    y2=float(b[3]),
                                )
                            )
                        # Best-effort cleanup: Results objects keep references to large arrays.
                        del r0, boxes, xyxy, confs, clss
                        # Periodic GC to control peak RSS without destroying performance.
                        if (chunk_start_idx + idx) % 10 == 0:
                            _release_cuda_memory_best_effort()

                def _is_cuda_oom_text(exc: Exception) -> bool:
                    msg = str(exc).lower()
                    return ("out of memory" in msg and "cuda" in msg) or "cudamemoryerror" in msg

                for chunk_start in range(0, len(image_paths_split), predict_chunk_size):
                    chunk_paths = image_paths_split[chunk_start : chunk_start + predict_chunk_size]
                    if not chunk_paths:
                        continue
                    try:
                        # stream=True prevents Ultralytics from buffering all Results objects in RAM.
                        results_iter = model.predict(
                            source=chunk_paths,
                            imgsz=int(input_hw[0]),
                            conf=float(conf_thr),
                            iou=float(iou_thr),
                            verbose=False,
                            batch=int(val_batch) if val_batch is not None else 1,
                            stream=True,
                            **pred_common,
                        )
                        _append_preds_from_results(results_iter, chunk_paths=chunk_paths, chunk_start_idx=chunk_start)
                    except Exception as exc:
                        if _is_cuda_oom_text(exc):
                            print(
                                f"[WARN] {format_name}: deep-diagnostics predict OOM on GPU for split={split}, chunk={chunk_start}. "
                                "Retrying on CPU.",
                                file=sys.stderr,
                            )
                            _release_cuda_memory_best_effort()
                            results_iter = model.predict(
                                source=chunk_paths,
                                imgsz=int(input_hw[0]),
                                conf=float(conf_thr),
                                iou=float(iou_thr),
                                verbose=False,
                                device="cpu",
                                batch=1,
                                stream=True,
                                **pred_common,
                            )
                            _append_preds_from_results(results_iter, chunk_paths=chunk_paths, chunk_start_idx=chunk_start)
                        else:
                            raise
                    _release_cuda_memory_best_effort()

                _support._write_deep_diagnostics_artifacts(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name="ultralytics_predict",
                    weights_path=weights_path,
                    data_yaml_path=dataset_yaml_path,
                    split=split,
                    preds=preds_split,
                    gt_rows=gt_rows_split,
                    image_paths=image_paths_split,
                    names=names,
                    conf_thr=conf_thr,
                    iou_thr=iou_thr,
                    imgsz=input_hw[0],
                    batch=val_batch,
                    inference_source="ultralytics_model_predict",
                    gt_source="ultralytics_verify_image_label",
                    nms_profile="ultralytics_validator_multilabel",
                )
        test_end_time = datetime.now()
        test_system_profile = _collect_test_system_profile(
            root_dir=root_dir,
            format_name=format_name,
            backend_name="ultralytics",
            runtime_provider="ultralytics",
            runtime_device=str(val_kwargs.get("device", "")) or (str(runtime_device) if runtime_device else None),
        )
        perf_payload: dict[str, Any] | None = None
        if collect_performance:
            duration_s = max(0.0, (test_end_time - test_start_time).total_seconds())
            perf_payload = _ultralytics_perf_payload_from_result(
                result,
                duration_s=duration_s,
                warmup_images=perf_warmup_images,
                images_count=test_image_count,
            )
            _support._write_perf_artifact(root_dir, format_name, weights_path, perf_payload)
        persist_target_test_artifacts_state(
            root_dir,
            format_name=format_name,
            target_path=weights_path,
            dataset_yaml=dataset_yaml_path,
            backend="ultralytics",
            performance=perf_payload,
            test_system_profile=test_system_profile,
            status="ok",
        )
        return BackendRunResult(
            format=format_name,
            backend="ultralytics",
            success=True,
            test_start_time=test_start_time,
            test_end_time=test_end_time,
            inference={
                "imgsz": imgsz,
                "conf": val_conf,
                "iou": val_iou,
                "batch": val_batch,
                "inference_source": "ultralytics_model_val",
                "gt_source": "ultralytics_validator",
                "nms_profile": "ultralytics_validator_multilabel",
                "performance": perf_payload,
                "test_system_profile": test_system_profile,
            },
            target_path=weights_path,
        )
    except Exception as exc:
        persist_target_test_artifacts_state(
            root_dir,
            format_name=format_name,
            target_path=weights_path,
            dataset_yaml=dataset_yaml_path,
            backend="ultralytics",
            test_system_profile=_collect_test_system_profile(
                root_dir=root_dir,
                format_name=format_name,
                backend_name="ultralytics",
                runtime_provider="ultralytics",
            ),
            status="failed",
            error=str(exc),
        )
        return BackendRunResult(
            format=format_name,
            backend="ultralytics",
            success=False,
            test_start_time=test_start_time,
            test_end_time=datetime.now(),
            inference={
                "imgsz": imgsz,
                "conf": val_conf,
                "iou": val_iou,
                "batch": val_batch,
                "inference_source": "ultralytics_model_val",
                "gt_source": "ultralytics_validator",
                "nms_profile": "ultralytics_validator_multilabel",
            },
            target_path=weights_path,
            error=str(exc),
        )
    finally:
        try:
            normalize_ultralytics_run_layout(root_dir)
        except Exception:
            pass
        best_effort_prune_runs_detect_near_run(root_dir)
        try:
            prune_empty_sidecar_dirs(root_dir)
        except Exception:
            pass


