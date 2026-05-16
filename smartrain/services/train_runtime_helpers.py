from __future__ import annotations

import json
import os
import re
import subprocess
import gc
from pathlib import Path
from typing import Any
from datetime import datetime

from smartrain.core.runtime.run_artifacts import (
    materialize_canonical_run_model,
    resolve_run_model,
    run_test_backend_dir,
    run_tests_dir,
    run_train_backend_dir,
)


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


def normalize_external_run_layout(run_dir: str) -> None:
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        return
    train_dir = run_train_backend_dir(str(root), "ultralytics")
    train_dir.mkdir(parents=True, exist_ok=True)
    for entry in list(root.iterdir()):
        if entry.name in {"training_metadata.json", "models", "tmp", "tests"} or entry.name.startswith("train-"):
            continue
        target = train_dir / entry.name
        if target.exists():
            continue
        entry.rename(target)


def _materialize_canonical_run_model(run_dir: str, source_path: str | None = None) -> str | None:
    target = materialize_canonical_run_model(
        run_dir,
        ext=".pt",
        source_path=source_path,
        move=True,
        normalize_metadata=True,
    )
    return str(target) if target is not None else None


def _find_external_best_checkpoint(run_dir: str) -> str | None:
    found = resolve_run_model(run_dir, ".pt")
    return str(found) if found is not None else None


def ensure_external_best_checkpoint_layout(run_dir: str) -> str | None:
    return _materialize_canonical_run_model(run_dir, _find_external_best_checkpoint(run_dir))


_EXTERNAL_EVAL_SUBSTITUTE_MODES = frozenset({"external_eval_substitute", "external_infer_fallback"})


def write_external_fallback_metrics(model_dir: str, *, provider_id: str, rc: int) -> str:
    test_dir = str(run_test_backend_dir(model_dir, "ultralytics"))
    os.makedirs(test_dir, exist_ok=True)
    marker = os.path.join(test_dir, "fallback_eval_substitute.txt")
    with open(marker, "w", encoding="utf-8") as f:
        f.write("external eval substitute was used for test stage\n")
    legacy_marker = os.path.join(test_dir, "fallback_infer.txt")
    if os.path.isfile(legacy_marker):
        try:
            os.remove(legacy_marker)
        except OSError:
            pass
    csv_path = os.path.join(str(run_tests_dir(model_dir)), "test_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("provider,test_mode,return_code\n")
        f.write(f"{provider_id},external_eval_substitute,{int(rc)}\n")
    return csv_path


def normalize_external_eval_substitute_mode(mode: str | None) -> str | None:
    key = str(mode or "").strip()
    if not key:
        return None
    if key in _EXTERNAL_EVAL_SUBSTITUTE_MODES:
        return "external_eval_substitute"
    return key


def run_mfel_external_eval_substitute(
    *,
    repo_path: str,
    venv_path: str,
    model_path: str,
    data_yaml: str,
    model_dir: str,
    imgsz: int,
    conf: float | None,
    iou: float | None,
    batch: int | None,
    device: str | None,
) -> int:
    python_bin = os.path.join(venv_path, "Scripts" if os.name == "nt" else "bin", "python")
    launcher = (
        Path(__file__).resolve().parent.parent
        / "external_providers"
        / "launchers"
        / "mfel_val_launcher.py"
    )
    cmd = [
        python_bin,
        str(launcher),
        "--repo",
        repo_path,
        "--model",
        model_path,
        "--data",
        data_yaml,
        "--imgsz",
        str(int(imgsz)),
        "--project",
        model_dir,
        "--name",
        "test",
    ]
    if conf is not None:
        cmd.extend(["--conf", str(float(conf))])
    if iou is not None:
        cmd.extend(["--iou", str(float(iou))])
    if batch is not None:
        cmd.extend(["--batch", str(int(batch))])
    if device:
        cmd.extend(["--device", str(device)])
    proc = subprocess.run(cmd, cwd=repo_path)
    return int(proc.returncode)


def maybe_free_cuda_memory() -> None:
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
