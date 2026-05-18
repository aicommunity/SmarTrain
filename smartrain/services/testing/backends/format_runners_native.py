"""Native ONNX/TRT/pt_uni test backend runner."""

from __future__ import annotations

from smartrain.services.testing.backends import format_runners_support as _support
from smartrain.services.testing.backends.format_runners_support import (
    BackendRunResult,
    EvalProvenance,
    PerfCollector,
    _Pred,
    datetime,
    format_test_dir_for_write,
    normalize_eval_params,
    os,
    persist_target_test_artifacts_state,
    sys,
    time,
    tqdm,
)


def run_native_format_backend(
    *,
    root_dir: str,
    weights_path: str,
    dataset_yaml_path: str,
    format_name: str,
    imgsz: int | None = None,
    val_conf: float | None = None,
    val_iou: float | None = None,
    val_batch: int | None = None,
    deep_diagnostics: bool = False,
    collect_performance: bool = False,
    perf_warmup_images: int = 5,
    onnx_provider_policy: str | None = None,
    runtime_device: str | None = None,
    task_type: str | None = None,
) -> BackendRunResult:
    backend_name = "onnxruntime" if format_name == "onnx" else ("unified_pt" if format_name == "pt_uni" else "tensorrt")
    provider_by_format = {
        "onnx": "onnxruntime",
        "engine": "tensorrt",
        "trt": "tensorrt",
        "pt_uni": "ultralytics",
    }
    try:
        if format_name == "pt_uni":
            from smartrain.services.testing.backends.format_runners import run_ultralytics_backend

            eval_params = normalize_eval_params(imgsz=imgsz, conf=val_conf, iou=val_iou)
            result = run_ultralytics_backend(
                root_dir=root_dir,
                weights_path=weights_path,
                dataset_yaml_path=dataset_yaml_path,
                format_name="pt_uni",
                imgsz=int(eval_params["imgsz"]),
                val_conf=float(eval_params["conf"]),
                val_iou=float(eval_params["iou"]),
                val_batch=val_batch,
                deep_diagnostics=deep_diagnostics,
                collect_performance=collect_performance,
                perf_warmup_images=perf_warmup_images,
                task_type=task_type,
            )
            if result.success:
                test_dir = format_test_dir_for_write(root_dir, "pt_uni")
                os.makedirs(test_dir, exist_ok=True)
                _support._write_test_args_yaml(
                    test_dir,
                    backend=backend_name,
                    format_name="pt_uni",
                    weights_path=weights_path,
                    data_yaml_path=dataset_yaml_path,
                    imgsz=int(eval_params["imgsz"]),
                    conf=float(eval_params["conf"]),
                    iou=float(eval_params["iou"]),
                    batch=val_batch,
                    inference_source="ultralytics_model_val",
                    gt_source="ultralytics_validator",
                    nms_profile="ultralytics_validator_multilabel",
                )
                persist_target_test_artifacts_state(
                    root_dir,
                    format_name="pt_uni",
                    target_path=weights_path,
                    dataset_yaml=dataset_yaml_path,
                    backend=backend_name,
                    performance=result.inference.get("performance") if isinstance(result.inference, dict) else None,
                    test_system_profile=(
                        result.inference.get("test_system_profile")
                        if isinstance(result.inference, dict)
                        else None
                    ),
                    status="ok",
                )
            else:
                persist_target_test_artifacts_state(
                    root_dir,
                    format_name="pt_uni",
                    target_path=weights_path,
                    dataset_yaml=dataset_yaml_path,
                    backend=backend_name,
                    test_system_profile=_support._collect_test_system_profile(
                        root_dir=root_dir,
                        format_name="pt_uni",
                        backend_name=backend_name,
                        runtime_provider="ultralytics",
                        runtime_device=runtime_device,
                    ),
                    status="failed",
                    error=result.error,
                )
            result.backend = backend_name
            if isinstance(result.inference, dict):
                result.inference.update(
                    EvalProvenance(
                        inference_source="ultralytics_model_val",
                        gt_source="ultralytics_validator",
                        nms_profile="ultralytics_validator_multilabel",
                    ).as_dict()
                )
            return result
        if format_name == "onnx":
            inferred_policy = "cpu_only" if str(runtime_device or "").strip().lower() == "cpu" else None
            policy = str(onnx_provider_policy or inferred_policy or os.getenv("SMARTTRAIN_ONNX_PROVIDER_POLICY", "gpu_preferred")).strip().lower()
            if policy not in {"gpu_strict", "gpu_preferred", "cpu_only"}:
                policy = "gpu_preferred"
            providers = (
                ["CPUExecutionProvider"]
                if policy == "cpu_only"
                else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            names = _support._load_names(dataset_yaml_path)
            gt_rows, _by_image_gt, image_paths = _support._collect_gt(dataset_yaml_path, "test")
            eval_params = normalize_eval_params(imgsz=imgsz, conf=val_conf, iou=val_iou)
            conf_thr = float(eval_params["conf"])
            iou_thr = float(eval_params["iou"])
            requested_imgsz = int(eval_params["imgsz"])
            strict_imgsz = str(os.getenv("SMARTTRAIN_ONNX_IMGSZ_STRICT", "0")).strip().lower() in {"1", "true", "yes"}
            use_worker = str(os.getenv("SMARTTRAIN_ONNX_USE_SUBPROCESS", "1")).strip().lower() not in {"0", "false", "no"}
            perf_collector = PerfCollector(warmup_images=perf_warmup_images) if collect_performance else None
            perf_payload: dict[str, Any] | None = None
            provider_actual: str | None = None
            session_init_ns = 0
            provider_switched_to_cpu = False
            if use_worker:
                preds, input_hw, perf_payload, provider_actual = _support._run_onnx_split_in_subprocess(
                    split_name="test",
                    image_paths=image_paths,
                    weights_path=weights_path,
                    dataset_yaml_path=dataset_yaml_path,
                    imgsz=requested_imgsz,
                    conf_thr=conf_thr,
                    iou_thr=iou_thr,
                    providers=providers,
                    provider_policy=policy,
                    collect_performance=collect_performance,
                    perf_warmup_images=perf_warmup_images,
                )
            else:
                import onnxruntime as ort  # type: ignore

                available = list(ort.get_available_providers())
                providers_local = [p for p in providers if p in available]
                if policy == "gpu_strict" and "CUDAExecutionProvider" not in providers_local:
                    raise RuntimeError(_support._format_onnx_error("provider_unavailable", f"CUDAExecutionProvider is unavailable. available={available}"))
                try:
                    t_sess0 = time.perf_counter_ns()
                    session = _support._build_onnx_session_with_retry(ort, weights_path, providers_local)
                    session_init_ns = int(time.perf_counter_ns() - t_sess0)
                except Exception as primary_exc:
                    if policy == "gpu_strict":
                        raise RuntimeError(_support._format_onnx_error(_support._classify_onnx_error_text(str(primary_exc)), str(primary_exc))) from primary_exc
                    cpu_only = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else None
                    if not cpu_only:
                        raise primary_exc
                    print("[WARN] onnx: switching to CPUExecutionProvider after repeated initialization failures.")
                    provider_switched_to_cpu = True
                    t_sess0 = time.perf_counter_ns()
                    session = ort.InferenceSession(weights_path, providers=cpu_only)
                    session_init_ns = int(time.perf_counter_ns() - t_sess0)
                try:
                    provider_actual = str((session.get_providers() or [None])[0] or "")
                except Exception:
                    provider_actual = None
                input_hw = _support._resolve_imgsz_from_onnx(session, requested_imgsz)
                preds = _support._run_onnx_split_with_retry(
                    split_name="test",
                    image_paths=image_paths,
                    session=session,
                    input_hw=input_hw,
                    conf_thr=conf_thr,
                    iou_thr=iou_thr,
                    names=names,
                    format_name=format_name,
                    weights_path=weights_path,
                    perf_collector=perf_collector,
                )
                if perf_collector is not None:
                    perf_payload = perf_collector.to_payload()
            if isinstance(perf_payload, dict):
                perf_payload.setdefault("diagnostics_overhead", {})
                perf_payload["diagnostics_overhead"].update(
                    {
                        "session_init_ms": float(session_init_ns / 1_000_000.0),
                        "provider_switched_to_cpu": bool(provider_switched_to_cpu),
                    }
                )
            if not isinstance(input_hw, tuple):
                input_hw = (requested_imgsz, requested_imgsz)
            if strict_imgsz and requested_imgsz != int(input_hw[0]):
                raise RuntimeError(
                    _support._format_onnx_error(
                        "shape_mismatch",
                        f"requested imgsz={requested_imgsz} does not match model input={int(input_hw[0])}",
                    )
                )
            if perf_payload:
                _support._write_perf_artifact(root_dir, format_name, weights_path, perf_payload)
            inference = _support._write_native_eval_artifacts(
                root_dir=root_dir,
                format_name=format_name,
                backend_name=backend_name,
                weights_path=weights_path,
                data_yaml_path=dataset_yaml_path,
                split="test",
                preds=preds,
                gt_rows=gt_rows,
                names=names,
                conf_thr=conf_thr,
                iou_thr=iou_thr,
                imgsz=input_hw[0],
                batch=val_batch,
                inference_source="onnxruntime_session",
                gt_source="ultralytics_verify_image_label",
                nms_profile="ultralytics_nms_multilabel",
            )
            if deep_diagnostics:
                _support._write_deep_diagnostics_artifacts(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name=backend_name,
                    weights_path=weights_path,
                    data_yaml_path=dataset_yaml_path,
                    split="test",
                    preds=preds,
                    gt_rows=gt_rows,
                    image_paths=image_paths,
                    names=names,
                    conf_thr=conf_thr,
                    iou_thr=iou_thr,
                    imgsz=input_hw[0],
                    batch=val_batch,
                    inference_source="onnxruntime_session",
                    gt_source="ultralytics_verify_image_label",
                    nms_profile="ultralytics_nms_multilabel",
                )
            split_status: dict[str, Any] = {"test": {"status": "ok", "error": None}, "val": {"status": "ok", "error": None}}
            try:
                gt_rows_val, _bgv, image_paths_val = _support._collect_gt(dataset_yaml_path, "val")
                if use_worker:
                    preds_val, _input_hw_val, _perf_val, _provider_val = _support._run_onnx_split_in_subprocess(
                        split_name="val",
                        image_paths=image_paths_val,
                        weights_path=weights_path,
                        dataset_yaml_path=dataset_yaml_path,
                        imgsz=input_hw[0],
                        conf_thr=conf_thr,
                        iou_thr=iou_thr,
                        providers=providers,
                        provider_policy=policy,
                        collect_performance=False,
                    )
                else:
                    preds_val = _support._run_onnx_split_with_retry(
                        split_name="val",
                        image_paths=image_paths_val,
                        session=session,
                        input_hw=input_hw,
                        conf_thr=conf_thr,
                        iou_thr=iou_thr,
                        names=names,
                        format_name=format_name,
                        weights_path=weights_path,
                    )
                _support._write_native_eval_artifacts(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name=backend_name,
                    weights_path=weights_path,
                    data_yaml_path=dataset_yaml_path,
                    split="val",
                    preds=preds_val,
                    gt_rows=gt_rows_val,
                    names=names,
                    conf_thr=conf_thr,
                    iou_thr=iou_thr,
                    imgsz=input_hw[0],
                    batch=val_batch,
                    inference_source="onnxruntime_session",
                    gt_source="ultralytics_verify_image_label",
                    nms_profile="ultralytics_nms_multilabel",
                )
                if deep_diagnostics:
                    _support._write_deep_diagnostics_artifacts(
                        root_dir=root_dir,
                        format_name=format_name,
                        backend_name=backend_name,
                        weights_path=weights_path,
                        data_yaml_path=dataset_yaml_path,
                        split="val",
                        preds=preds_val,
                        gt_rows=gt_rows_val,
                        image_paths=image_paths_val,
                        names=names,
                        conf_thr=conf_thr,
                        iou_thr=iou_thr,
                        imgsz=input_hw[0],
                        batch=val_batch,
                        inference_source="onnxruntime_session",
                        gt_source="ultralytics_verify_image_label",
                        nms_profile="ultralytics_nms_multilabel",
                    )
            except Exception as val_exc:
                split_status["val"] = {"status": "failed", "error": str(val_exc)}
            overall_status = "ok" if split_status["val"]["status"] == "ok" else "partial_ok"
            overall_error = None if overall_status == "ok" else f"val split failed: {split_status['val']['error']}"
            persist_target_test_artifacts_state(
                root_dir,
                format_name=format_name,
                target_path=weights_path,
                dataset_yaml=dataset_yaml_path,
                backend=backend_name,
                performance=perf_payload,
                test_system_profile=_support._collect_test_system_profile(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name=backend_name,
                    runtime_provider=("onnxruntime-worker" if use_worker else "onnxruntime-session"),
                    runtime_provider_actual=provider_actual,
                    runtime_device=runtime_device,
                ),
                status=overall_status,
                error=overall_error,
                split_status=split_status,
            )
            if isinstance(inference, dict):
                inference["onnx_provider_policy"] = policy
                inference["onnx_provider_actual"] = provider_actual
                inference["performance"] = perf_payload
                inference["test_system_profile"] = _support._collect_test_system_profile(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name=backend_name,
                    runtime_provider=("onnxruntime-worker" if use_worker else "onnxruntime-session"),
                    runtime_provider_actual=provider_actual,
                    runtime_device=runtime_device,
                )
            return BackendRunResult(
                format=format_name,
                backend=backend_name,
                success=True,
                test_start_time=datetime.now(),
                test_end_time=datetime.now(),
                inference=inference,
                target_path=weights_path,
            )
        elif format_name in {"engine", "trt"}:
            names = _support._load_names(dataset_yaml_path)
            gt_rows, _by_image_gt, image_paths = _support._collect_gt(dataset_yaml_path, "test")
            eval_params = normalize_eval_params(imgsz=imgsz, conf=val_conf, iou=val_iou)
            conf_thr = float(eval_params["conf"])
            iou_thr = float(eval_params["iou"])
            input_hw = _support._resolve_input_hw_from_native_artifact(weights_path, int(eval_params["imgsz"]))
            perf_collector = PerfCollector(warmup_images=perf_warmup_images) if collect_performance else None
            trt_runtime = _support._prepare_trt_runtime(weights_path)
            preds: list[_Pred] = []
            total_images = len(image_paths)
            print(f"[INFO] {format_name}: running native test on {total_images} images with {weights_path}")
            for image_path in tqdm(image_paths, desc=f"{format_name}:test", unit="img", file=sys.stdout):
                infer_out = _support._infer_with_trt_engine(trt_runtime, image_path, input_hw, conf_thr, iou_thr, names)
                if isinstance(infer_out, tuple) and len(infer_out) == 2:
                    image_preds, perf_ns = infer_out
                else:
                    image_preds, perf_ns = infer_out, {}
                preds.extend(image_preds)
                if perf_collector is not None:
                    perf_collector.record_total_image(int(perf_ns.get("total", 0)))
                    perf_collector.record_stage("preprocess_ms", int(perf_ns.get("preprocess", 0)))
                    perf_collector.record_stage("infer_ms", int(perf_ns.get("infer", 0)))
                    perf_collector.record_stage("decode_nms_ms", int(perf_ns.get("decode_nms", 0)))
                    perf_collector.record_stage("io_load_ms", int(perf_ns.get("io_load", 0)))
                    perf_collector.record_stage("diagnostics_alloc_ms", int(perf_ns.get("diagnostics_alloc", 0)))
                    perf_collector.record_stage("diagnostics_h2d_ms", int(perf_ns.get("diagnostics_h2d", 0)))
                    perf_collector.record_stage("diagnostics_execute_ms", int(perf_ns.get("diagnostics_execute", 0)))
                    perf_collector.record_stage("diagnostics_d2h_ms", int(perf_ns.get("diagnostics_d2h", 0)))
            print(f"[INFO] {format_name}: native test completed ({total_images}/{total_images} images).")
            perf_payload = perf_collector.to_payload() if perf_collector is not None else None
            if isinstance(perf_payload, dict):
                perf_payload.setdefault("diagnostics_overhead", {})
                perf_payload["diagnostics_overhead"]["engine_init_ms"] = float(
                    int(trt_runtime.get("init_ns", 0)) / 1_000_000.0
                )
            if perf_payload:
                _support._write_perf_artifact(root_dir, format_name, weights_path, perf_payload)
            inference = _support._write_native_eval_artifacts(
                root_dir=root_dir,
                format_name=format_name,
                backend_name=backend_name,
                weights_path=weights_path,
                data_yaml_path=dataset_yaml_path,
                split="test",
                preds=preds,
                gt_rows=gt_rows,
                names=names,
                conf_thr=conf_thr,
                iou_thr=iou_thr,
                imgsz=input_hw[0],
                batch=val_batch,
                inference_source="tensorrt_engine",
                gt_source="ultralytics_verify_image_label",
                nms_profile="ultralytics_nms_multilabel",
            )
            split_status: dict[str, Any] = {"test": {"status": "ok", "error": None}, "val": {"status": "ok", "error": None}}
            native_debug: dict[str, Any] = {
                "imgsz": input_hw[0],
                "test_gt_count": len(gt_rows),
                "test_pred_count": len(preds),
            }
            try:
                native_debug["invalid_metrics_candidate"] = bool(
                    all(abs(float(inference.get(k) or 0.0)) <= 1e-12 for k in ("mAP50-95", "mAP50", "Box-F1", "Box-P", "Box-R"))
                )
            except Exception:
                native_debug["invalid_metrics_candidate"] = False
            try:
                gt_rows_val, _bgv, image_paths_val = _support._collect_gt(dataset_yaml_path, "val")
                preds_val: list[_Pred] = []
                print(f"[INFO] {format_name}: running native val on {len(image_paths_val)} images with {weights_path}")
                for image_path in tqdm(image_paths_val, desc=f"{format_name}:val", unit="img", file=sys.stdout):
                    infer_out = _support._infer_with_trt_engine(trt_runtime, image_path, input_hw, conf_thr, iou_thr, names)
                    if isinstance(infer_out, tuple) and len(infer_out) == 2:
                        preds_val.extend(infer_out[0])
                    else:
                        preds_val.extend(infer_out)
                print(f"[INFO] {format_name}: native val completed ({len(image_paths_val)}/{len(image_paths_val)} images).")
                _support._write_native_eval_artifacts(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name=backend_name,
                    weights_path=weights_path,
                    data_yaml_path=dataset_yaml_path,
                    split="val",
                    preds=preds_val,
                    gt_rows=gt_rows_val,
                    names=names,
                    conf_thr=conf_thr,
                    iou_thr=iou_thr,
                    imgsz=input_hw[0],
                    batch=val_batch,
                    inference_source="tensorrt_engine",
                    gt_source="ultralytics_verify_image_label",
                    nms_profile="ultralytics_nms_multilabel",
                )
                native_debug["val_gt_count"] = len(gt_rows_val)
                native_debug["val_pred_count"] = len(preds_val)
            except Exception as val_exc:
                split_status["val"] = {"status": "failed", "error": str(val_exc)}
            overall_status = "ok" if split_status["val"]["status"] == "ok" else "partial_ok"
            overall_error = None if overall_status == "ok" else f"val split failed: {split_status['val']['error']}"
            persist_target_test_artifacts_state(
                root_dir,
                format_name=format_name,
                target_path=weights_path,
                dataset_yaml=dataset_yaml_path,
                backend=backend_name,
                performance=perf_payload,
                test_system_profile=_support._collect_test_system_profile(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name=backend_name,
                    runtime_provider="tensorrt",
                    runtime_device=runtime_device,
                ),
                status=overall_status,
                error=overall_error,
                split_status=split_status,
                native_debug=native_debug,
            )
            if isinstance(inference, dict):
                inference["performance"] = perf_payload
                inference["test_system_profile"] = _support._collect_test_system_profile(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name=backend_name,
                    runtime_provider="tensorrt",
                    runtime_device=runtime_device,
                )
            return BackendRunResult(
                format=format_name,
                backend=backend_name,
                success=True,
                test_start_time=datetime.now(),
                test_end_time=datetime.now(),
                inference=inference,
                target_path=weights_path,
            )
        raise RuntimeError(f"Unsupported native backend format: {format_name}")
    except Exception as exc:
        err_text = str(exc)
        if format_name == "onnx" and not err_text.strip().startswith("["):
            err_text = _support._format_onnx_error(_support._classify_onnx_error_text(err_text), err_text)
        persist_target_test_artifacts_state(
            root_dir,
            format_name=format_name,
            target_path=weights_path,
            dataset_yaml=dataset_yaml_path,
            backend=backend_name,
            test_system_profile=_support._collect_test_system_profile(
                root_dir=root_dir,
                format_name=format_name,
                backend_name=backend_name,
                runtime_provider=provider_by_format.get(format_name),
                runtime_device=runtime_device,
            ),
            status="unavailable",
            error=err_text,
        )
        return BackendRunResult(
            format=format_name,
            backend=backend_name,
            success=False,
            test_start_time=datetime.now(),
            test_end_time=datetime.now(),
            inference={"imgsz": imgsz, "conf": val_conf, "iou": val_iou, "batch": val_batch},
            target_path=weights_path,
            error=err_text,
        )
