"""Train execution pipeline (post-config merge)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from smartrain.backends.external_provider_adapter import ExternalProviderAdapter
from smartrain.cli_contracts import emit_replay
from smartrain.confidence_recommendation import write_not_available_recommendations
from smartrain.dataset_hash import calculate_dataset_hash
from smartrain.external_providers.runner import run_external_infer, run_external_train
from smartrain.mpl_runtime import ensure_matplotlib_training_runtime
from smartrain.provider_global_index import get_provider_location, list_provider_records, reconcile_stale_provider_paths
from smartrain.run_artifacts import run_test_backend_dir, run_tests_dir
from smartrain.services.train_runtime_helpers import (
    build_run_name,
    ensure_external_best_checkpoint_layout,
    json_safe_train_summary,
    load_batch_from_training_metadata,
    maybe_free_cuda_memory,
    normalize_external_run_layout,
    run_mfel_external_val_fallback,
    resolve_external_eval_source,
    write_external_fallback_metrics,
)
from smartrain.train_model_catalog import TrainModelCatalog, is_supported_external_provider_model
from smartrain.train_profile import task_to_metadata_task_type


def _confidence_recommendation_params(args: argparse.Namespace) -> tuple[bool, float, float, float]:
    return (
        bool(getattr(args, "conf_rec_disable", False)),
        float(getattr(args, "conf_rec_beta_recall", 2.0)),
        float(getattr(args, "conf_rec_beta_precision", 0.5)),
        float(getattr(args, "conf_rec_fallback", 0.25)),
    )


def _confidence_recommendation_config(args: argparse.Namespace) -> dict[str, Any]:
    conf_rec_disable, conf_rec_beta_recall, conf_rec_beta_precision, conf_rec_fallback = (
        _confidence_recommendation_params(args)
    )
    return {
        "enabled": not conf_rec_disable,
        "beta_recall": conf_rec_beta_recall,
        "beta_precision": conf_rec_beta_precision,
        "fallback_confidence": conf_rec_fallback,
    }


def _get_installed_external_provider_record(provider_id: str) -> dict[str, Any] | None:
    key = str(provider_id or "").strip().lower()
    if not key:
        return None
    reconcile_stale_provider_paths()
    for rec in list_provider_records():
        pid = str(rec.get("provider_id", "")).strip().lower()
        if pid != key:
            continue
        if str(rec.get("install_state", "")).strip().lower() != "installed":
            continue
        repo_path = Path(str(rec.get("repo_path", "")).strip()).expanduser()
        venv_path = Path(str(rec.get("venv_path", "")).strip()).expanduser()
        if not repo_path.is_dir() or not venv_path.is_dir():
            continue
        return rec
    return None


class _MtmRuntimeOps:
    """Runtime operations bundle for train/test execution."""

    def __init__(
        self,
        *,
        train_yolo_fn: Callable[..., Any],
        test_yolo_fn: Callable[..., Any],
        save_training_metadata_fn: Callable[..., Any],
        collect_system_profile_fn: Callable[..., Any],
        build_run_name_fn: Callable[..., Any] | None = None,
        resolve_external_eval_source_fn: Callable[..., Any] | None = None,
        json_safe_train_summary_fn: Callable[..., Any] | None = None,
        load_batch_from_training_metadata_fn: Callable[..., Any] | None = None,
        run_external_train_fn: Callable[..., Any] | None = None,
        run_external_infer_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._train_yolo_fn = train_yolo_fn
        self._test_yolo_fn = test_yolo_fn
        self._save_training_metadata_fn = save_training_metadata_fn
        self._collect_system_profile_fn = collect_system_profile_fn
        self._build_run_name_fn = build_run_name_fn or build_run_name
        self._resolve_external_eval_source_fn = resolve_external_eval_source_fn or resolve_external_eval_source
        self._json_safe_train_summary_fn = json_safe_train_summary_fn or json_safe_train_summary
        self._load_batch_from_training_metadata_fn = load_batch_from_training_metadata_fn or load_batch_from_training_metadata
        self._run_external_train_fn = run_external_train_fn or run_external_train
        self._run_external_infer_fn = run_external_infer_fn or run_external_infer

    def train_yolo(self, **kwargs: Any) -> Any:
        return self._train_yolo_fn(**kwargs)

    def test_yolo(self, *args: Any, **kwargs: Any) -> Any:
        return self._test_yolo_fn(*args, **kwargs)

    def save_training_metadata(self, **kwargs: Any) -> Any:
        return self._save_training_metadata_fn(**kwargs)

    def collect_system_profile(self, model_dir: str) -> dict[str, Any]:
        return self._collect_system_profile_fn(model_dir)

    def build_run_name(self, *args: Any, **kwargs: Any) -> str:
        return self._build_run_name_fn(*args, **kwargs)

    def resolve_external_eval_source(self, dataset_path: str) -> str:
        return self._resolve_external_eval_source_fn(dataset_path)

    def json_safe_train_summary(self, train_kw: dict[str, Any] | None) -> dict[str, Any] | None:
        return self._json_safe_train_summary_fn(train_kw)

    def load_batch_from_training_metadata(self, model_dir: str) -> int | None:
        return self._load_batch_from_training_metadata_fn(model_dir)

    def run_external_train(self, *args: Any, **kwargs: Any) -> Any:
        return self._run_external_train_fn(*args, **kwargs)

    def run_external_infer(self, *args: Any, **kwargs: Any) -> Any:
        return self._run_external_infer_fn(*args, **kwargs)


def _run_external_provider_flow(
    *,
    runtime_ops: _MtmRuntimeOps,
    args: argparse.Namespace,
    u_cfg: dict[str, Any],
    workspace_root: str,
    data: str,
    target_dir: str,
    model_version: str,
    epochs: int,
    batch: int,
    img_size: int,
) -> int:
    conf_rec_disable, conf_rec_beta_recall, conf_rec_beta_precision, conf_rec_fallback = (
        _confidence_recommendation_params(args)
    )
    external_provider = str(getattr(args, "external_provider", "") or "").strip()
    rec = _get_installed_external_provider_record(external_provider)
    repo_for_catalog = str(rec.get("repo_path", "")).strip() if isinstance(rec, dict) else None
    requested_model = str(getattr(args, "model", "") or model_version)
    if not os.path.isfile(requested_model):
        is_supported = is_supported_external_provider_model(
            external_provider,
            requested_model,
            provider_repo_path=repo_for_catalog or None,
        )
        if not is_supported:
            ext_aliases = TrainModelCatalog(
                provider=external_provider,
                provider_repo_path=repo_for_catalog or None,
            ).supported_aliases()
            known = ", ".join(ext_aliases) if ext_aliases else "<none>"
            print(
                f"[ERROR] Model {requested_model!r} is not supported by external provider "
                f"{external_provider!r}. Supported aliases: {known}"
            )
            return 2
    training_start_time = datetime.now()
    location = get_provider_location(external_provider)
    if location is None and not getattr(args, "external_repo", None):
        print(
            f"[ERROR] External provider {external_provider!r} is not installed. "
            "Use `smartrain providers install` or pass --external-repo."
        )
        return 1
    repo_path = str(getattr(args, "external_repo", "") or "").strip() or (location.repo_path if location else "")
    venv_path = location.venv_path if location else os.path.join(repo_path, "venv")
    external_adapter = ExternalProviderAdapter(
        provider_id=external_provider,
        repo_path=repo_path,
        venv_path=venv_path,
        train_runner=runtime_ops.run_external_train,
        infer_runner=runtime_ops.run_external_infer,
    )

    if not venv_path:
        print(f"[ERROR] Missing venv for external provider {external_provider!r}. Reinstall provider.")
        return 1
    try:
        dataset_hash = calculate_dataset_hash(data)
    except Exception:
        dataset_hash = None
    run_name = runtime_ops.build_run_name(external_provider, model_version, epochs, batch, dataset_hash)
    print(f"[INFO] External run name: {run_name}")
    rc = external_adapter.run_train(
        dataset_path=data,
        model=model_version,
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        device=str(u_cfg.get("device")) if u_cfg.get("device") is not None else None,
        target_dir=target_dir,
        run_name=run_name,
    )
    training_end_time = datetime.now()
    dataset_name = os.path.basename(os.path.normpath(data))
    external_run_dir = os.path.join(target_dir, dataset_name, run_name)
    os.makedirs(external_run_dir, exist_ok=True)
    normalize_external_run_layout(external_run_dir)
    ensure_external_best_checkpoint_layout(external_run_dir)
    test_success = False
    test_error = None
    test_start_time = None
    test_end_time = None
    inference_info = None
    if rc == 0:
        try:
            maybe_free_cuda_memory()
            val_batch = args.val_batch if args.val_batch is not None else batch
            test_start_time, test_end_time, inference_info = runtime_ops.test_yolo(
                external_run_dir,
                data,
                training_start_time=training_start_time,
                training_end_time=training_end_time,
                train_img_size=img_size,
                val_imgsz=args.val_imgsz,
                val_conf=args.val_conf,
                val_iou=args.val_iou,
                val_batch=val_batch,
                conf_rec_disable=conf_rec_disable,
                conf_rec_beta_recall=conf_rec_beta_recall,
                conf_rec_beta_precision=conf_rec_beta_precision,
                conf_rec_fallback=conf_rec_fallback,
                non_interactive=args.non_interactive,
            )
            test_success = True
        except Exception as e:
            test_error = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[ERROR] Error during external provider testing: {e}")
            best_model = ensure_external_best_checkpoint_layout(external_run_dir)
            if best_model:
                fallback_start = datetime.now()
                fallback_source = runtime_ops.resolve_external_eval_source(data)
                fallback_conf = float(args.val_conf) if args.val_conf is not None else 0.25
                fallback_imgsz = int(args.val_imgsz) if args.val_imgsz is not None else int(img_size)
                if external_provider == "mfel-yolo":
                    fallback_rc = run_mfel_external_val_fallback(
                        repo_path=repo_path,
                        venv_path=venv_path,
                        model_path=best_model,
                        data_yaml=os.path.join(data, "data.yaml"),
                        model_dir=external_run_dir,
                        imgsz=fallback_imgsz,
                        conf=args.val_conf,
                        iou=args.val_iou,
                        batch=args.val_batch if args.val_batch is not None else batch,
                        device=str(u_cfg.get("device")) if u_cfg.get("device") is not None else None,
                    )
                else:
                    fallback_rc = external_adapter.run_batch(
                        model_path=best_model,
                        source_path=fallback_source,
                        conf=fallback_conf,
                        imgsz=fallback_imgsz,
                        device=str(u_cfg.get("device")) if u_cfg.get("device") is not None else None,
                        target_dir=external_run_dir,
                        run_name="test",
                    )
                fallback_end = datetime.now()
                if fallback_rc == 0:
                    if external_provider == "mfel-yolo":
                        test_results_csv = os.path.join(
                            str(run_test_backend_dir(external_run_dir, "ultralytics")), "results.csv"
                        )
                        if os.path.isfile(test_results_csv):
                            shutil.copy2(
                                test_results_csv, os.path.join(str(run_tests_dir(external_run_dir)), "test_metrics.csv")
                            )
                        else:
                            write_external_fallback_metrics(
                                external_run_dir, provider_id=external_provider, rc=fallback_rc
                            )
                    else:
                        write_external_fallback_metrics(
                            external_run_dir, provider_id=external_provider, rc=fallback_rc
                        )
                    test_start_time = fallback_start
                    test_end_time = fallback_end
                    inference_info = {
                        "imgsz": fallback_imgsz,
                        "conf": fallback_conf,
                        "mode": "external_infer_fallback",
                    }
                    reason = "external_fallback_without_ultralytics_val_metrics"
                    write_not_available_recommendations(
                        model_dir=external_run_dir,
                        split="test",
                        reason=reason,
                        beta_recall=conf_rec_beta_recall,
                        beta_precision=conf_rec_beta_precision,
                        fallback_confidence=conf_rec_fallback,
                    )
                    write_not_available_recommendations(
                        model_dir=external_run_dir,
                        split="val",
                        reason=reason,
                        beta_recall=conf_rec_beta_recall,
                        beta_precision=conf_rec_beta_precision,
                        fallback_confidence=conf_rec_fallback,
                    )
                    test_success = True
                    test_error = None
                else:
                    test_success = False
                    test_error = (
                        f"{test_error}\nExternal infer fallback failed with return code {fallback_rc}"
                    )
            else:
                test_success = False
    _ext_mpl = None
    if isinstance(inference_info, dict):
        _c = inference_info.get("matplotlib_runtime")
        _ext_mpl = _c if isinstance(_c, dict) else None
    if _ext_mpl is None:
        _ext_mpl = ensure_matplotlib_training_runtime(non_interactive=args.non_interactive).as_dict()
    runtime_ops.save_training_metadata(
        model_dir=external_run_dir,
        dataset_path=data,
        model_version=model_version.replace(".pt", ""),
        training_start_time=training_start_time,
        training_end_time=training_end_time,
        test_start_time=test_start_time,
        test_end_time=test_end_time,
        epochs=epochs,
        batch=batch,
        img_size=img_size,
        training_success=(rc == 0),
        training_error=None if rc == 0 else f"external provider returned code {rc}",
        test_success=test_success if rc == 0 else False,
        test_error=test_error if rc == 0 else "test skipped because external train failed",
        dataset_hash=dataset_hash,
        inference=inference_info,
        workspace_root=workspace_root,
        task_type=task_to_metadata_task_type(u_cfg.get("task")),
        training_provider=external_provider,
        external_provider_id=external_provider,
        system_profile=runtime_ops.collect_system_profile(external_run_dir),
        matplotlib_runtime=_ext_mpl,
        confidence_recommendation_config=_confidence_recommendation_config(args),
    )
    try:
        marker = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "provider": {"type": "external", "id": external_provider},
            "model": model_version,
            "dataset_path": data,
            "target_dir": target_dir,
            "run_dir": external_run_dir,
            "repo_path": repo_path,
            "venv_path": venv_path,
            "return_code": int(rc),
        }
        marker_path = os.path.join(target_dir, "_external_train_last.json")
        os.makedirs(target_dir, exist_ok=True)
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(marker, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return rc


def _run_builtin_train_and_eval_flow(
    *,
    runtime_ops: _MtmRuntimeOps,
    args: argparse.Namespace,
    u_cfg: dict[str, Any],
    sm_opts: dict[str, Any],
    workspace_root: str,
    data: str,
    target_dir: str,
    model_version: str,
    epochs: int,
    batch: int,
    img_size: int,
    task_type: str,
) -> None:
    from smartrain.backends.train_test_registry import resolve_train_backend
    conf_rec_disable, conf_rec_beta_recall, conf_rec_beta_precision, conf_rec_fallback = (
        _confidence_recommendation_params(args)
    )

    training_success = False
    training_error = None
    test_success = True
    test_error = None
    training_start_time = None
    training_end_time = None
    test_start_time = None
    test_end_time = None
    model_dir = None
    inference_info = None
    dataset_hash = None
    meta_extras: dict[str, Any] = {}

    try:
        (
            model_dir,
            training_start_time,
            training_end_time,
            dataset_hash,
            _,
            meta_extras,
        ) = runtime_ops.train_yolo(
            dataset_path=data,
            target_dir=target_dir,
            non_interactive=args.non_interactive,
            workspace_root=workspace_root,
            ultralytics_cfg=u_cfg,
            smartrain_opts=sm_opts,
        )
        training_success = bool(meta_extras.get("training_ok"))
    except Exception as e:
        training_success = False
        training_error = str(e)
        training_end_time = datetime.now()
        print(f"[ERROR] Error during training: {e}")
        training_error = f"{str(e)}\n{traceback.format_exc()}"
        try:
            dataset_hash = calculate_dataset_hash(data)
        except Exception:
            dataset_hash = None
        if not model_dir:
            dataset_name = os.path.basename(os.path.normpath(data))
            folder_name = runtime_ops.build_run_name(
                "ultralytics",
                model_version,
                epochs,
                batch,
                dataset_hash,
                timestamp=training_start_time,
            )
            model_dir = os.path.join(target_dir, dataset_name, folder_name)
            os.makedirs(model_dir, exist_ok=True)
        meta_extras = {
            "task_type": task_to_metadata_task_type(u_cfg.get("task")),
            "train_kw": {k: v for k, v in u_cfg.items() if k != "data"},
            "training_ok": False,
            "mpl_runtime": ensure_matplotlib_training_runtime(
                non_interactive=args.non_interactive
            ).as_dict(),
        }

    if training_success and model_dir:
        try:
            maybe_free_cuda_memory()
            val_batch = args.val_batch if args.val_batch is not None else batch
            test_start_time, test_end_time, inference_info = runtime_ops.test_yolo(
                model_dir,
                data,
                training_start_time=training_start_time,
                training_end_time=training_end_time,
                train_img_size=img_size,
                val_imgsz=args.val_imgsz,
                val_conf=args.val_conf,
                val_iou=args.val_iou,
                val_batch=val_batch,
                conf_rec_disable=conf_rec_disable,
                conf_rec_beta_recall=conf_rec_beta_recall,
                conf_rec_beta_precision=conf_rec_beta_precision,
                conf_rec_fallback=conf_rec_fallback,
                non_interactive=args.non_interactive,
            )
        except Exception as e:
            test_success = False
            test_error = str(e)
            test_end_time = datetime.now()
            print(f"[ERROR] Error during testing: {e}")
            test_error = f"{str(e)}\n{traceback.format_exc()}"

    if model_dir:
        _mpl_meta = (
            meta_extras.get("mpl_runtime") if isinstance(meta_extras.get("mpl_runtime"), dict) else None
        )
        if _mpl_meta is None and isinstance(inference_info, dict):
            _cand = inference_info.get("matplotlib_runtime")
            _mpl_meta = _cand if isinstance(_cand, dict) else None
        runtime_ops.save_training_metadata(
            model_dir=model_dir,
            dataset_path=data,
            model_version=model_version.replace(".pt", ""),
            training_start_time=training_start_time,
            training_end_time=training_end_time,
            test_start_time=test_start_time,
            test_end_time=test_end_time,
            epochs=epochs,
            batch=batch,
            img_size=img_size,
            training_success=training_success,
            training_error=training_error,
            test_success=test_success,
            test_error=test_error,
            dataset_hash=dataset_hash,
            inference=inference_info,
            workspace_root=workspace_root,
            task_type=meta_extras.get("task_type") or task_to_metadata_task_type(u_cfg.get("task")),
            ultralytics_train_summary=runtime_ops.json_safe_train_summary(meta_extras.get("train_kw")),
            training_provider=resolve_train_backend(task_type=task_type, model_format="pt").backend,
            external_provider_id=None,
            system_profile=runtime_ops.collect_system_profile(model_dir),
            matplotlib_runtime=_mpl_meta,
            confidence_recommendation_config=_confidence_recommendation_config(args),
        )


def _run_test_only_flow(
    *,
    runtime_ops: _MtmRuntimeOps,
    args: argparse.Namespace,
    u_cfg: dict[str, Any],
    workspace_root: str,
    data: str,
    batch: int,
    img_size: int,
    task_type: str,
) -> None:
    from smartrain.backends.train_test_registry import resolve_train_backend
    conf_rec_disable, conf_rec_beta_recall, conf_rec_beta_precision, conf_rec_fallback = (
        _confidence_recommendation_params(args)
    )

    model_dir = args.model_dir
    if not model_dir:
        print("[ERROR] Model path not specified")
        return
    test_success = True
    test_error = None
    test_start_time = None
    test_end_time = None
    inference_info = None
    try:
        val_batch = (
            args.val_batch
            if args.val_batch is not None
            else (runtime_ops.load_batch_from_training_metadata(model_dir) or batch)
        )
        test_start_time, test_end_time, inference_info = runtime_ops.test_yolo(
            model_dir,
            data,
            train_img_size=img_size,
            val_imgsz=args.val_imgsz,
            val_conf=args.val_conf,
            val_iou=args.val_iou,
            val_batch=val_batch,
            conf_rec_disable=conf_rec_disable,
            conf_rec_beta_recall=conf_rec_beta_recall,
            conf_rec_beta_precision=conf_rec_beta_precision,
            conf_rec_fallback=conf_rec_fallback,
            non_interactive=args.non_interactive,
        )
    except Exception as e:
        test_success = False
        test_error = str(e)
        test_end_time = datetime.now()
        print(f"[ERROR] Error during testing: {e}")
        test_error = f"{str(e)}\n{traceback.format_exc()}"

    _test_only_mpl = None
    if isinstance(inference_info, dict):
        _tc = inference_info.get("matplotlib_runtime")
        _test_only_mpl = _tc if isinstance(_tc, dict) else None
    runtime_ops.save_training_metadata(
        model_dir=model_dir,
        dataset_path=data,
        test_start_time=test_start_time,
        test_end_time=test_end_time,
        test_success=test_success,
        test_error=test_error,
        inference=inference_info,
        workspace_root=workspace_root,
        task_type=task_to_metadata_task_type(u_cfg.get("task")),
        training_provider=resolve_train_backend(task_type=task_type, model_format="pt").backend,
        external_provider_id=None,
        system_profile=runtime_ops.collect_system_profile(model_dir),
        matplotlib_runtime=_test_only_mpl,
        confidence_recommendation_config=_confidence_recommendation_config(args),
    )


def run_train_after_setup(
    *,
    args: argparse.Namespace,
    request: Any,
    parser: argparse.ArgumentParser,
    u_cfg: dict[str, Any],
    sm_opts: dict[str, Any],
    workspace_root: str,
    data: str,
    target_dir: str,
    model_version: str,
    epochs: int,
    batch: int,
    img_size: int,
    replay_cmd: str | None,
) -> int | None:
    """Runs training+test (local or external), writes metadata, returns exit code."""
    from smartrain import model_training_module as mtm
    # Keep behavior identical while localizing mtm wiring in one composition root.
    runtime_ops = _MtmRuntimeOps(
        train_yolo_fn=mtm.train_yolo,
        test_yolo_fn=mtm.test_yolo,
        save_training_metadata_fn=mtm.save_training_metadata,
        collect_system_profile_fn=mtm.collect_system_profile,
        build_run_name_fn=getattr(mtm, "_build_run_name", None),
        resolve_external_eval_source_fn=getattr(mtm, "_resolve_external_eval_source", None),
        json_safe_train_summary_fn=getattr(mtm, "_json_safe_train_summary", None),
        load_batch_from_training_metadata_fn=getattr(mtm, "_load_batch_from_training_metadata", None),
        run_external_train_fn=getattr(mtm, "run_external_train", None),
        run_external_infer_fn=getattr(mtm, "run_external_infer", None),
    )
    task_type = task_to_metadata_task_type(u_cfg.get("task"))

    external_provider = str(getattr(args, "external_provider", "") or "").strip()
    if external_provider:
        return _run_external_provider_flow(
            runtime_ops=runtime_ops,
            args=args,
            u_cfg=u_cfg,
            workspace_root=workspace_root,
            data=data,
            target_dir=target_dir,
            model_version=model_version,
            epochs=epochs,
            batch=batch,
            img_size=img_size,
        )

    if not args.test_only:
        _run_builtin_train_and_eval_flow(
            runtime_ops=runtime_ops,
            args=args,
            u_cfg=u_cfg,
            sm_opts=sm_opts,
            workspace_root=workspace_root,
            data=data,
            target_dir=target_dir,
            model_version=model_version,
            epochs=epochs,
            batch=batch,
            img_size=img_size,
            task_type=task_type,
        )
    else:
        _run_test_only_flow(
            runtime_ops=runtime_ops,
            args=args,
            u_cfg=u_cfg,
            workspace_root=workspace_root,
            data=data,
            batch=batch,
            img_size=img_size,
            task_type=task_type,
        )

    if replay_cmd:
        emit_replay(command_name="train", parser=parser, args=args, stage="after execution")

    return None
