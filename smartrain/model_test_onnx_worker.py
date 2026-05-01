from __future__ import annotations

import json
import sys
import time
from typing import Any

from tqdm import tqdm

from smartrain.model_test_backends import (
    PerfCollector,
    _Pred,
    _infer_with_onnx_session,
    _is_onnx_cuda_oom_error,
    _load_names,
    _release_cuda_memory_best_effort,
    _resolve_imgsz_from_onnx,
)


def _pred_to_dict(pred: _Pred) -> dict[str, Any]:
    return {
        "image_path": pred.image_path,
        "cls_id": int(pred.cls_id),
        "conf": float(pred.conf),
        "x1": float(pred.x1),
        "y1": float(pred.y1),
        "x2": float(pred.x2),
        "y2": float(pred.y2),
    }


def _run(request: dict[str, Any]) -> dict[str, Any]:
    import onnxruntime as ort  # type: ignore

    weights_path = str(request["weights_path"])
    dataset_yaml_path = str(request["dataset_yaml_path"])
    split_name = str(request["split_name"])
    image_paths = [str(x) for x in request.get("image_paths", [])]
    imgsz = request.get("imgsz")
    conf_thr = float(request.get("conf_thr", 0.001))
    iou_thr = float(request.get("iou_thr", 0.7))
    max_retries = int(request.get("max_retries", 3))
    providers = [str(x) for x in request.get("providers", [])]
    provider_policy = str(request.get("provider_policy", "gpu_preferred")).strip().lower()
    collect_performance = bool(request.get("collect_performance", False))
    perf_warmup_images = int(request.get("perf_warmup_images", 5))
    worker_started_ns = time.perf_counter_ns()

    available = list(ort.get_available_providers())
    selected = [p for p in providers if p in available] or [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    if provider_policy == "gpu_strict" and "CUDAExecutionProvider" not in selected:
        return {
            "ok": False,
            "error": f"[provider_unavailable] CUDAExecutionProvider is unavailable. available={available}",
            "provider": None,
        }
    if not selected:
        selected = None  # type: ignore[assignment]

    names = _load_names(dataset_yaml_path)
    last_error: Exception | None = None
    retries_count = 0
    retry_sleep_ns = 0
    session_init_total_ns = 0
    provider_switched_to_cpu = False

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                print(f"[WARN] onnx-worker: retrying {split_name} ({attempt}/{max_retries}).", file=sys.stderr)
            t_sess0 = time.perf_counter_ns()
            session = ort.InferenceSession(weights_path, providers=selected)
            session_init_total_ns += int(time.perf_counter_ns() - t_sess0)
            input_hw = _resolve_imgsz_from_onnx(session, int(imgsz) if imgsz is not None else None)
            preds: list[_Pred] = []
            perf_collector = PerfCollector(warmup_images=perf_warmup_images) if collect_performance else None
            print(
                f"[INFO] onnx-worker: running {split_name} on {len(image_paths)} images with {weights_path}",
                file=sys.stderr,
            )
            for image_path in tqdm(image_paths, desc=f"onnx:{split_name}", unit="img", file=sys.stderr):
                image_preds, perf_ns = _infer_with_onnx_session(session, image_path, input_hw, conf_thr, iou_thr, names)
                preds.extend(image_preds)
                if perf_collector is not None:
                    perf_collector.record_total_image(int(perf_ns.get("total", 0)))
                    perf_collector.record_stage("preprocess_ms", int(perf_ns.get("preprocess", 0)))
                    perf_collector.record_stage("infer_ms", int(perf_ns.get("infer", 0)))
                    perf_collector.record_stage("decode_nms_ms", int(perf_ns.get("decode_nms", 0)))
                    perf_collector.record_stage("io_load_ms", int(perf_ns.get("io_load", 0)))
            print(
                f"[INFO] onnx-worker: completed {split_name} ({len(image_paths)}/{len(image_paths)} images).",
                file=sys.stderr,
            )
            perf_payload = perf_collector.to_payload() if perf_collector is not None else None
            if isinstance(perf_payload, dict):
                perf_payload.setdefault("diagnostics_overhead", {})
                perf_payload["diagnostics_overhead"].update(
                    {
                        "worker_wall_ms": float(max(0, time.perf_counter_ns() - worker_started_ns) / 1_000_000.0),
                        "session_init_ms": float(session_init_total_ns / 1_000_000.0),
                        "retries_count": int(retries_count),
                        "retry_sleep_ms": float(retry_sleep_ns / 1_000_000.0),
                        "provider_switched_to_cpu": bool(provider_switched_to_cpu),
                    }
                )
            return {
                "ok": True,
                "preds": [_pred_to_dict(p) for p in preds],
                "input_hw": [int(input_hw[0]), int(input_hw[1])],
                "provider": selected[0] if selected else None,
                "performance": perf_payload,
            }
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            if _is_onnx_cuda_oom_error(exc) and attempt < max_retries:
                retries_count += 1
                # If CUDA initialization fails (OOM / CUBLAS alloc_failed), keep retrying can be pointless.
                # Switch to CPUExecutionProvider for remaining attempts (if available).
                if isinstance(selected, list) and "CUDAExecutionProvider" in selected and "CPUExecutionProvider" in available:
                    if provider_policy == "gpu_strict":
                        break
                    selected = ["CPUExecutionProvider"]
                    provider_switched_to_cpu = True
                    print(f"[WARN] onnx-worker: switching to CPUExecutionProvider after CUDA init OOM for {split_name}.", file=sys.stderr)
                _release_cuda_memory_best_effort()
                sleep_s = 0.5 * (2 ** (attempt - 1))
                retry_sleep_ns += int(sleep_s * 1_000_000_000.0)
                time.sleep(sleep_s)
                continue
            break

    return {
        "ok": False,
        "error": str(last_error) if last_error is not None else f"onnx worker failed for split={split_name}",
    }


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({"ok": False, "error": "empty worker request"}))
        return 2
    try:
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"invalid worker request: {exc}"}))
        return 2

    response = _run(request)
    print(json.dumps(response, ensure_ascii=False))
    return 0 if bool(response.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
