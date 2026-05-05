from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from datetime import datetime


def build_run_name(
    provider_id: str,
    model_version: str,
    epochs: int,
    batch: int,
    dataset_hash: str | None,
    *,
    timestamp: datetime | None = None,
) -> str:
    ts = timestamp or datetime.now()
    timestamp_str = ts.strftime("%Y-%m-%d_%H-%M")
    provider = str(provider_id or "ultralytics").strip().lower().replace(" ", "-")
    model_token = Path(str(model_version)).name
    if model_token.endswith(".pt"):
        model_token = model_token[:-3]
    if model_token.endswith(".yaml"):
        model_token = model_token[:-5]
    model_token = re.sub(r"[^a-zA-Z0-9._+-]+", "-", model_token).strip("-") or "model"
    folder_name = f"{timestamp_str}_{provider}_{model_token}_{epochs}epochs_b{batch}"
    if dataset_hash:
        folder_name = f"{folder_name}-{dataset_hash}"
    return folder_name


def resolve_external_eval_source(dataset_path: str) -> str:
    root = Path(dataset_path).expanduser().resolve()
    candidates = [
        root / "test" / "images",
        root / "val" / "images",
        root / "test",
        root / "val",
    ]
    for cand in candidates:
        if cand.is_dir():
            return str(cand)
    return str(root)


def json_safe_train_summary(train_kw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not train_kw:
        return None
    out: dict[str, Any] = {}
    for k, v in train_kw.items():
        if k in ("data",):
            continue
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def load_batch_from_training_metadata(model_dir: str) -> int | None:
    try:
        meta_path = os.path.join(model_dir, "training_metadata.json")
        if not os.path.isfile(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        bs = (
            meta.get("training_info", {})
            .get("hyperparameters", {})
            .get("batch_size")
        )
        if bs is None:
            return None
        bs_i = int(bs)
        return bs_i if bs_i > 0 else None
    except Exception:
        return None
