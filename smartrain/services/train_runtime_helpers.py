from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import gc
from pathlib import Path
from typing import Any
from datetime import datetime

import yaml

from smartrain.core.runtime.path_portable import relativize_if_under
from smartrain.core.runtime.run_artifacts import (
    materialize_preferred_run_model,
    resolve_run_model,
    run_models_dir,
    run_test_backend_dir,
    run_tests_dir,
    run_train_backend_dir,
)
from smartrain.services.analyze.metrics_reader import training_args_yaml_path

logger = logging.getLogger(__name__)

def normalize_imgsz_token(imgsz: Any) -> int | None:
    if imgsz is None:
        return None
    if isinstance(imgsz, (list, tuple)) and imgsz:
        try:
            return int(imgsz[0])
        except (TypeError, ValueError):
            return None
    try:
        return int(imgsz)
    except (TypeError, ValueError):
        return None


def format_batch_token(batch: int | float) -> str:
    if isinstance(batch, float) and not batch.is_integer():
        return "b" + str(batch).replace(".", "p")
    return f"b{int(batch)}"


def build_run_name(
    provider_id: str,
    model_version: str,
    epochs: int,
    batch: int | float,
    dataset_hash: str | None,
    *,
    img_size: int | None = None,
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
    img_token = f"_{int(img_size)}px" if img_size is not None else ""
    batch_token = format_batch_token(batch)
    folder_name = f"{timestamp_str}_{provider}_{model_token}{img_token}_{epochs}epochs_{batch_token}"
    if dataset_hash:
        folder_name = f"{folder_name}-{dataset_hash}"
    return folder_name


def read_effective_ultralytics_train_hyperparams(run_dir: str) -> dict[str, Any]:
    args_path = training_args_yaml_path(run_dir)
    if not os.path.isfile(args_path):
        return {}
    try:
        with open(args_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    if payload.get("epochs") is not None:
        try:
            out["epochs"] = int(payload["epochs"])
        except (TypeError, ValueError):
            pass
    if payload.get("batch") is not None:
        try:
            raw_batch = payload["batch"]
            out["batch"] = float(raw_batch) if isinstance(raw_batch, float) else int(raw_batch)
        except (TypeError, ValueError):
            pass
    img_size = normalize_imgsz_token(payload.get("imgsz"))
    if img_size is not None:
        out["img_size"] = img_size
    return out


def _rename_run_model_files(run_dir: str, old_name: str, new_name: str) -> None:
    models_dir = run_models_dir(run_dir)
    if not models_dir.is_dir():
        return
    for entry in list(models_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.name == old_name or entry.name.startswith(f"{old_name}."):
            suffix = entry.name[len(old_name) :]
            target = models_dir / f"{new_name}{suffix}"
            if target.exists():
                logger.warning("Skip model rename %s -> %s: target exists", entry, target)
                continue
            entry.rename(target)


def _patch_metadata_after_run_rename(
    metadata_path: Path,
    *,
    old_name: str,
    new_name: str,
    effective: dict[str, Any],
    workspace_root: str | None,
    new_run_dir: str,
) -> None:
    if not metadata_path.is_file():
        return
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    changed = False

    paths = payload.get("paths")
    if isinstance(paths, dict):
        best_model = paths.get("best_model")
        if isinstance(best_model, str):
            if best_model == f"{old_name}.pt" or best_model == old_name:
                paths["best_model"] = f"{new_name}.pt"
                changed = True
            elif best_model.startswith(f"{old_name}."):
                paths["best_model"] = f"{new_name}{best_model[len(old_name):]}"
                changed = True

    ti = payload.get("training_info")
    if isinstance(ti, dict):
        hp = ti.get("hyperparameters")
        if not isinstance(hp, dict):
            hp = {}
            ti["hyperparameters"] = hp
        for key, eff_key in (("epochs", "epochs"), ("batch_size", "batch"), ("image_size", "img_size")):
            if effective.get(eff_key) is not None:
                hp[key] = effective[eff_key]
                changed = True

    if workspace_root is not None:
        wb = payload.get("workspace")
        if isinstance(wb, dict):
            rel = relativize_if_under(workspace_root, new_run_dir)
            if rel is not None and wb.get("run_directory_relative") != rel:
                wb["run_directory_relative"] = rel
                changed = True

    source = payload.get("source")
    if isinstance(source, dict):
        src_weights = source.get("source_weights")
        if isinstance(src_weights, str) and src_weights.startswith(f"{old_name}."):
            source["source_weights"] = f"{new_name}{src_weights[len(old_name):]}"
            changed = True

    if not changed:
        return
    tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(metadata_path)


def finalize_run_dir_naming(
    model_dir: str,
    *,
    provider_id: str,
    model_version: str,
    dataset_hash: str | None,
    training_start_time: datetime | None,
    workspace_root: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Reconcile run directory name with effective Ultralytics hyperparameters from args.yaml."""
    run_path = Path(model_dir).expanduser().resolve()
    if not run_path.is_dir() or not run_dir_has_train_artifacts(str(run_path)):
        return str(run_path), {}

    effective = read_effective_ultralytics_train_hyperparams(str(run_path))
    if not effective:
        return str(run_path), {}

    old_name = run_path.name
    eff_epochs = effective.get("epochs")
    eff_batch = effective.get("batch")
    eff_img_size = effective.get("img_size")
    if eff_epochs is None or eff_batch is None or eff_img_size is None:
        return str(run_path), effective

    new_name = build_run_name(
        provider_id,
        model_version,
        int(eff_epochs),
        eff_batch,
        dataset_hash,
        img_size=int(eff_img_size),
        timestamp=training_start_time,
    )
    if new_name == old_name:
        return str(run_path), effective

    new_run_dir = run_path.parent / new_name
    if new_run_dir.exists():
        logger.warning(
            "Skip run rename %s -> %s: target directory already exists",
            run_path,
            new_run_dir,
        )
        return str(run_path), effective

    try:
        run_path.rename(new_run_dir)
    except OSError as exc:
        logger.warning("Failed to rename run directory %s -> %s: %s", run_path, new_run_dir, exc)
        return str(run_path), effective

    _rename_run_model_files(str(new_run_dir), old_name, new_name)
    _patch_metadata_after_run_rename(
        new_run_dir / "training_metadata.json",
        old_name=old_name,
        new_name=new_name,
        effective=effective,
        workspace_root=workspace_root,
        new_run_dir=str(new_run_dir),
    )
    print(f"[INFO] Run directory renamed to match effective training hyperparameters: {new_run_dir}")
    return str(new_run_dir), effective


def run_dir_has_train_artifacts(run_dir: str) -> bool:
    if resolve_run_model(run_dir) is not None:
        return True
    train_backend = run_train_backend_dir(run_dir, "ultralytics")
    for rel in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml"):
        if (train_backend / rel).is_file():
            return True
    legacy_train = Path(run_dir) / "train"
    for rel in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml"):
        if (legacy_train / rel).is_file():
            return True
    return False


def recover_builtin_run_dir_after_train_error(
    *,
    target_dir: str,
    dataset_path: str,
    model_version: str,
    epochs: int,
    batch: int,
    img_size: int | None,
    dataset_hash: str | None,
    training_start_time: datetime | None,
) -> str | None:
    """Find an existing run directory when train_yolo failed after creating one."""
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    runs_root = os.path.join(target_dir, dataset_name)
    if not os.path.isdir(runs_root):
        return None

    if training_start_time is not None:
        folder_name = build_run_name(
            "ultralytics",
            model_version,
            epochs,
            batch,
            dataset_hash,
            img_size=img_size,
            timestamp=training_start_time,
        )
        candidate = os.path.join(runs_root, folder_name)
        if run_dir_has_train_artifacts(candidate):
            return candidate

    candidates: list[str] = []
    for name in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, name)
        if os.path.isdir(run_dir) and run_dir_has_train_artifacts(run_dir):
            candidates.append(run_dir)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


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


def _materialize_preferred_run_model(run_dir: str, source_path: str | None = None) -> str | None:
    target = materialize_preferred_run_model(
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
    return _materialize_preferred_run_model(run_dir, _find_external_best_checkpoint(run_dir))


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
