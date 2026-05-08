from __future__ import annotations

import json
import os
from typing import Any

import yaml


def is_cuda_oom_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg and "cuda" in msg


def default_resume_test_batch(run_dir: str) -> int:
    metadata_path = os.path.join(run_dir, "training_metadata.json")
    meta: dict[str, Any] | None = None
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                meta = payload
        except Exception:
            meta = None
    if isinstance(meta, dict):
        inf = meta.get("inference")
        if isinstance(inf, dict) and inf.get("batch") is not None:
            try:
                val = int(inf.get("batch"))
                if val > 0:
                    return val
            except Exception:
                pass

    for p in (
        os.path.join(run_dir, "train-ultralytics", "args.yaml"),
        os.path.join(run_dir, "train", "args.yaml"),
    ):
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = yaml.safe_load(f) or {}
            if isinstance(payload, dict) and payload.get("batch") is not None:
                val = int(payload.get("batch"))
                if val > 0:
                    return val
        except Exception:
            continue
    return 4


def next_backoff_batch(current: int, min_batch: int, backoff: int) -> int:
    if current <= min_batch:
        return current
    nxt = (current + backoff - 1) // backoff
    return max(min_batch, nxt)


def complete_missing_test_with_backoff(
    run_dir: str,
    *,
    workspace_root: str,
    initial_batch: int | None,
    min_batch: int,
    backoff: int,
    complete_missing_test_artifacts_cb,
    pt_test_runner_cb,
    update_metadata_cb,
    maybe_free_cuda_memory_cb,
) -> None:
    batch = int(initial_batch) if initial_batch is not None else default_resume_test_batch(run_dir)
    batch = max(min_batch, batch)
    attempt = 0
    while True:
        attempt += 1
        print(f"[INFO] Resume test attempt {attempt}: batch={batch}")
        try:
            complete_missing_test_artifacts_cb(
                run_dir,
                workspace_root=workspace_root,
                pt_test_runner=pt_test_runner_cb,
                pt_test_runner_kwargs={"non_interactive": True, "val_batch": batch},
                update_metadata_cb=update_metadata_cb,
            )
            return
        except Exception as e:
            if not is_cuda_oom_error(e):
                raise
            next_batch = next_backoff_batch(batch, min_batch, backoff)
            if next_batch == batch:
                raise RuntimeError(
                    f"CUDA OOM at minimal test batch={batch}; backoff exhausted."
                ) from e
            print(
                f"[WARN] CUDA OOM during resume test with batch={batch}; "
                + f"retrying with batch={next_batch}"
            )
            maybe_free_cuda_memory_cb()
            batch = next_batch

