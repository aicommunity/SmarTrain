from __future__ import annotations

import json
import sys
import time
from typing import Any

from tqdm import tqdm

from smartrain.model_test_backends import (
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

    available = list(ort.get_available_providers())
    selected = [p for p in providers if p in available] or [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    if not selected:
        selected = None  # type: ignore[assignment]

    names = _load_names(dataset_yaml_path)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                print(f"[WARN] onnx-worker: retrying {split_name} ({attempt}/{max_retries}).", file=sys.stderr)
            session = ort.InferenceSession(weights_path, providers=selected)
            input_hw = _resolve_imgsz_from_onnx(session, int(imgsz) if imgsz is not None else None)
            preds: list[_Pred] = []
            print(
                f"[INFO] onnx-worker: running {split_name} on {len(image_paths)} images with {weights_path}",
                file=sys.stderr,
            )
            for image_path in tqdm(image_paths, desc=f"onnx:{split_name}", unit="img", file=sys.stderr):
                preds.extend(_infer_with_onnx_session(session, image_path, input_hw, conf_thr, iou_thr, names))
            print(
                f"[INFO] onnx-worker: completed {split_name} ({len(image_paths)}/{len(image_paths)} images).",
                file=sys.stderr,
            )
            return {
                "ok": True,
                "preds": [_pred_to_dict(p) for p in preds],
                "input_hw": [int(input_hw[0]), int(input_hw[1])],
                "provider": selected[0] if selected else None,
            }
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            if _is_onnx_cuda_oom_error(exc) and attempt < max_retries:
                # If CUDA initialization fails (OOM / CUBLAS alloc_failed), keep retrying can be pointless.
                # Switch to CPUExecutionProvider for remaining attempts (if available).
                if isinstance(selected, list) and "CUDAExecutionProvider" in selected and "CPUExecutionProvider" in available:
                    selected = ["CPUExecutionProvider"]
                    print(f"[WARN] onnx-worker: switching to CPUExecutionProvider after CUDA init OOM for {split_name}.", file=sys.stderr)
                _release_cuda_memory_best_effort()
                time.sleep(0.5 * (2 ** (attempt - 1)))
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
