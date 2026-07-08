"""Ultralytics train_yolo / test_yolo execution (extracted from workflows.training)."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from smartrain.core.runtime.mpl_runtime import configure_matplotlib_before_ultralytics, ensure_matplotlib_training_runtime

configure_matplotlib_before_ultralytics()
from ultralytics import YOLO  # noqa: E402

from smartrain.core.runtime.run_artifacts import (
    preferred_run_model_path,
    normalize_ultralytics_run_layout,
    ensure_run_layout,
    materialize_preferred_run_model,
    resolve_run_model,
    run_tests_dir,
    run_tmp_dir,
)
from smartrain.core.testing.artifact_paths import format_metrics_path_for_split
from smartrain.core.training.confidence_recommendation import (
    compute_confidence_recommendations,
    read_recommendation_file,
    recommendation_file_path,
    recommendations_complete,
    write_not_available_recommendations,
    write_recommendation_file,
)
from smartrain.core.training.train_profile import task_to_metadata_task_type
from smartrain.core.workflow_adapters.training_runtime_api import calculate_dataset_hash
from smartrain.core.workflow_adapters.testing_runtime_api import persist_target_test_artifacts_state
from smartrain.services.train_runtime_helpers import build_run_name
from smartrain.services.training.train_config_kwargs_service import finalize_train_kwargs
from smartrain.services.training.train_metadata_io_service import (
    ensure_initial_training_metadata,
    save_metrics_csv,
)
from smartrain.services.training.train_model_resolution_service import (
    extract_effective_loaded_model,
    extract_model_family_scale,
    normalize_model_spec,
)
from smartrain.services.training.train_runtime_data_yaml_service import build_runtime_data_yaml

DEFAULT_MODEL_VERSION = "yolov8n"
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 16
DEFAULT_IMG_SIZE = 640


@dataclass(frozen=True)
class TrainYoloHooks:
    setup_weighted_sampling_env: Callable[[], None] | None = None
    register_weighted_sampling_callback: Callable[[Any], None] | None = None


def validate_dataset_dir(dataset_path: str) -> None:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Yaml file not found: {data_yaml}")


def model_kw_model(train_kw: dict[str, Any]) -> str:
    return str(train_kw.get("model", ""))


def validate_train_launch_paths(
    *,
    train_kw: dict[str, Any],
    model_dir: str,
    data_yaml: str,
) -> None:
    project_path = str(train_kw.get("project", ""))
    if os.path.normpath(project_path) != os.path.normpath(model_dir):
        raise RuntimeError(
            "Training launch rejected: unexpected project path in kwargs "
            f"({project_path}). Expected: {model_dir}"
        )
    if str(train_kw.get("data", "")) != str(data_yaml):
        raise RuntimeError(
            "Training launch rejected: unexpected data yaml in kwargs "
            f"({train_kw.get('data')}). Expected: {data_yaml}"
        )
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(f"Runtime data.yaml not found: {data_yaml}")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Run directory is missing: {model_dir}")
    if not os.access(model_dir, os.W_OK | os.X_OK):
        raise PermissionError(
            f"Run directory is not writable: {model_dir}. "
            "Check permissions and remove path-like keys from external args.yaml (for example save_dir)."
        )


def _materialize_preferred_run_model(run_dir: str, source_path: str | None = None) -> str | None:
    target = materialize_preferred_run_model(
        run_dir,
        ext=".pt",
        source_path=source_path,
        move=True,
        normalize_metadata=True,
    )
    return str(target) if target is not None else None


def ensure_confidence_recommendations(
    *,
    trained_model: Any,
    primary_test_result: Any,
    model_dir: str,
    data_yaml: str,
    imgsz: int | None,
    val_conf: float | None,
    val_iou: float | None,
    val_batch: int | None,
    beta_recall: float,
    beta_precision: float,
    fallback_confidence: float,
) -> None:
    test_path = recommendation_file_path(model_dir, "test")
    val_path = recommendation_file_path(model_dir, "val")
    has_test = recommendations_complete(read_recommendation_file(test_path))
    has_val = recommendations_complete(read_recommendation_file(val_path))
    if has_test and has_val:
        print("[INFO] Confidence recommendations already exist (val/test), skipping recompute.")
        return

    if not has_test:
        test_payload = compute_confidence_recommendations(
            primary_test_result,
            split="test",
            beta_recall=float(beta_recall),
            beta_precision=float(beta_precision),
            fallback_confidence=float(fallback_confidence),
        )
        write_recommendation_file(test_path, test_payload)
        print(f"[OK] Confidence recommendations (test): {test_path}")

    if has_val:
        return

    val_kwargs: dict[str, Any] = {
        "data": data_yaml,
        "split": "val",
        "plots": False,
        "save": False,
        "verbose": False,
        "project": str(run_tests_dir(model_dir)),
        "name": "val-ultralytics-recs",
        "exist_ok": True,
    }
    if imgsz is not None:
        val_kwargs["imgsz"] = imgsz
    if val_conf is not None:
        val_kwargs["conf"] = val_conf
    if val_iou is not None:
        val_kwargs["iou"] = val_iou
    if val_batch is not None:
        val_kwargs["batch"] = int(val_batch)
    try:
        val_result = trained_model.val(**val_kwargs)
        try:
            with open(format_metrics_path_for_split(model_dir, "val", "pt"), "w", encoding="utf-8") as f:
                f.write(val_result.to_csv())
        except Exception:
            pass
        val_payload = compute_confidence_recommendations(
            val_result,
            split="val",
            beta_recall=float(beta_recall),
            beta_precision=float(beta_precision),
            fallback_confidence=float(fallback_confidence),
        )
        write_recommendation_file(val_path, val_payload)
        print(f"[OK] Confidence recommendations (val): {val_path}")
    except Exception as exc:
        write_not_available_recommendations(
            model_dir=model_dir,
            split="val",
            reason=f"val_split_failed: {exc}",
            beta_recall=float(beta_recall),
            beta_precision=float(beta_precision),
            fallback_confidence=float(fallback_confidence),
        )
        print(f"[WARN] Failed to compute val confidence recommendations: {exc}")
    try:
        normalize_ultralytics_run_layout(model_dir)
    except Exception:
        pass


def train_yolo(
    dataset_path: str,
    target_dir: str,
    non_interactive: bool = False,
    workspace_root: str | None = None,
    ultralytics_cfg: dict[str, Any] | None = None,
    smartrain_opts: dict[str, Any] | None = None,
    *,
    hooks: TrainYoloHooks | None = None,
):
    mpl_rt = ensure_matplotlib_training_runtime(non_interactive=non_interactive)
    ultralytics_cfg = dict(ultralytics_cfg or {})
    smartrain_opts = dict(smartrain_opts or {})
    hooks = hooks or TrainYoloHooks()

    training_start_time = datetime.now()
    validate_dataset_dir(dataset_path)

    dataset_name = os.path.basename(os.path.normpath(dataset_path))

    model_version = normalize_model_spec(
        ultralytics_cfg.get("model", DEFAULT_MODEL_VERSION),
        default_model=DEFAULT_MODEL_VERSION,
        add_pt_when_missing=True,
    )
    ultralytics_cfg["model"] = model_version
    epochs = int(ultralytics_cfg.get("epochs", DEFAULT_EPOCHS))

    try:
        dataset_hash = calculate_dataset_hash(dataset_path)
        print(f"[INFO] Dataset hash: {dataset_hash}")
    except Exception as e:
        print(f"[WARNING] Failed to calculate dataset hash: {e}")
        dataset_hash = None

    batch = int(ultralytics_cfg.get("batch", DEFAULT_BATCH))
    img_size = int(ultralytics_cfg.get("imgsz", DEFAULT_IMG_SIZE))
    folder_name = build_run_name(
        "ultralytics",
        model_version,
        epochs,
        batch,
        dataset_hash,
        timestamp=training_start_time,
    )

    model_dir = os.path.join(target_dir, dataset_name, folder_name)

    if os.path.exists(model_dir):
        if non_interactive:
            print(f"[INFO] The folder already exists, continue without prompting: {model_dir}")
        else:
            while True:
                answer = input(
                    f"[WARNING] A folder with the same name already exists: {model_dir}. Continue training? (y/n): \n"
                ).strip().lower()
                if answer == "y":
                    break
                elif answer == "n":
                    sys.exit(1)
                else:
                    print("Please enter 'y' or 'n' only.\n")
    else:
        os.makedirs(model_dir, exist_ok=True)

    data_yaml = build_runtime_data_yaml(
        dataset_path,
        model_dir,
        stage="train",
        ensure_run_layout_cb=ensure_run_layout,
        run_tmp_dir_cb=run_tmp_dir,
    )
    train_kw = finalize_train_kwargs(ultralytics_cfg, data_yaml, model_dir)
    if non_interactive or mpl_rt.force_ultralytics_plots_false:
        train_kw.setdefault("plots", False)
    validate_train_launch_paths(train_kw=train_kw, model_dir=model_dir, data_yaml=data_yaml)
    ensure_initial_training_metadata(
        model_dir=model_dir,
        dataset_path=dataset_path,
        model_version=model_version.replace(".pt", ""),
        epochs=epochs,
        batch=batch,
        img_size=img_size,
        training_start_time=training_start_time,
        dataset_hash=dataset_hash,
        workspace_root=workspace_root,
        task_type=task_to_metadata_task_type(train_kw.get("task")),
    )

    clearml_task = None
    if smartrain_opts.get("clearml"):
        try:
            from clearml import Task
        except ImportError as e:
            raise ImportError(
                "For --clearml, install: pip install 'smartrain[clearml]' or pip install clearml"
            ) from e
        cm_proj = (
            smartrain_opts.get("clearml_project")
            or os.environ.get("CLEARML_PROJECT")
            or "smartrain"
        )
        clearml_task = Task.init(
            project_name=cm_proj,
            task_name=os.path.basename(model_dir),
            task_type=Task.TaskTypes.training,
        )
        train_kw = clearml_task.connect(train_kw)

    if smartrain_opts.get("weighted_sampling") and hooks.setup_weighted_sampling_env is not None:
        hooks.setup_weighted_sampling_env()

    print("\n" + "=" * 60)
    print(f"[INFO] Training models: {model_kw_model(train_kw)}")
    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Task: {train_kw.get('task', 'detect')}")
    print(f"[INFO] Configuration: {data_yaml}")
    print(f"[INFO] Saving results in {model_dir}")
    print("=" * 60 + "\n")

    requested_model = normalize_model_spec(
        train_kw.get("model", model_version),
        default_model=DEFAULT_MODEL_VERSION,
        add_pt_when_missing=True,
    )
    train_kw["model"] = requested_model
    model = YOLO(requested_model)
    loaded_model = extract_effective_loaded_model(model, fallback=requested_model)
    print(f"[INFO] Requested model: {requested_model}")
    print(f"[INFO] Loaded model: {loaded_model}")

    req_fs = extract_model_family_scale(requested_model)
    loaded_fs = extract_model_family_scale(loaded_model)
    if req_fs and loaded_fs and req_fs != loaded_fs:
        mismatch_msg = (
            "[ERROR] Model family/scale mismatch: "
            f"requested {req_fs[0]}{req_fs[1]}, loaded {loaded_fs[0]}{loaded_fs[1]}. "
            "Silent model replacement is blocked."
        )
        if non_interactive:
            raise RuntimeError(mismatch_msg)
        while True:
            answer = input(f"{mismatch_msg} Continue anyway? (y/n): ").strip().lower()
            if answer == "y":
                print("[WARNING] User confirmed training with mismatched model.")
                break
            if answer == "n":
                raise RuntimeError("Training aborted by user due to model mismatch.")
            print("Please enter 'y' or 'n' only.\n")

    if smartrain_opts.get("weighted_sampling") and hooks.register_weighted_sampling_callback is not None:
        hooks.register_weighted_sampling_callback(model)

    training_end_time = None
    canonical_best_path = preferred_run_model_path(model_dir, ".pt")
    model_path = canonical_best_path
    try:
        model.train(**train_kw)
        training_end_time = datetime.now()
    except Exception as e:
        training_end_time = datetime.now()
        print(
            f"[ERROR] Training failed ({model_version}, dataset {dataset_name}, {epochs} epochs): {e}"
        )
    finally:
        if clearml_task is not None:
            try:
                clearml_task.close()
            except Exception:
                pass
        try:
            normalize_ultralytics_run_layout(model_dir)
        except Exception:
            pass

    try:
        best_src = resolve_run_model(model_dir)
        materialized = _materialize_preferred_run_model(
            model_dir,
            source_path=str(best_src) if best_src is not None else None,
        )
        model_path = str(materialized) if materialized is not None else canonical_best_path
        print("\n" + "-" * 60)
        if os.path.exists(model_path):
            print("[OK] Training complete.")
            print(f"[INFO] Model saved at path:\n{model_path}")
    except Exception:
        pass

    weights_dest = resolve_run_model(model_dir)
    training_weights_ok = weights_dest is not None and weights_dest.is_file()

    meta_extras = {
        "train_kw": {k: v for k, v in train_kw.items() if k != "data"},
        "task_type": task_to_metadata_task_type(train_kw.get("task")),
        "training_ok": training_weights_ok,
        "mpl_runtime": mpl_rt.as_dict(),
    }
    return model_dir, training_start_time, training_end_time, dataset_hash, workspace_root, meta_extras


def test_yolo(
    model_dir: str,
    dataset_path: str,
    training_start_time: datetime | None = None,
    training_end_time: datetime | None = None,
    train_img_size: int | None = None,
    val_imgsz: int | None = None,
    val_conf: float | None = None,
    val_iou: float | None = None,
    val_batch: int | None = None,
    conf_rec_disable: bool = False,
    conf_rec_beta_recall: float = 2.0,
    conf_rec_beta_precision: float = 0.5,
    conf_rec_fallback: float = 0.25,
    *,
    non_interactive: bool = False,
):
    del training_start_time, training_end_time
    mpl_rt = ensure_matplotlib_training_runtime(non_interactive=non_interactive)
    test_start_time = datetime.now()

    data_yaml = build_runtime_data_yaml(
        dataset_path,
        model_dir,
        stage="test",
        ensure_run_layout_cb=ensure_run_layout,
        run_tmp_dir_cb=run_tmp_dir,
    )
    imgsz = val_imgsz if val_imgsz is not None else train_img_size

    inference_record: dict[str, Any] = {
        "imgsz": imgsz,
        "conf": val_conf,
        "iou": val_iou,
        "batch": val_batch,
        "matplotlib_runtime": mpl_rt.as_dict(),
    }

    model_path = preferred_run_model_path(model_dir, ".pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"canonical run model is missing: {model_path}")
    trained_model = YOLO(model_path)

    val_kwargs: dict[str, Any] = {
        "data": data_yaml,
        "split": "test",
        "project": str(run_tests_dir(model_dir)),
        "name": "test-ultralytics",
        "exist_ok": True,
        "plots": False,
        "save": False,
    }
    if imgsz is not None:
        val_kwargs["imgsz"] = imgsz
    if val_conf is not None:
        val_kwargs["conf"] = val_conf
    if val_iou is not None:
        val_kwargs["iou"] = val_iou
    if val_batch is not None:
        val_kwargs["batch"] = int(val_batch)

    print("\n" + "=" * 60)
    print(f"[INFO] Model testing: {model_dir}")
    print(f"[INFO] Dataset: {dataset_path}")
    print(f"[INFO] Configuration: {data_yaml}")
    print(f"[INFO] Saving results in {model_dir}")
    if imgsz is not None:
        print(f"[INFO] val imgsz={imgsz}, batch={val_batch}, conf={val_conf}, iou={val_iou}")
    print("=" * 60 + "\n")

    test_end_time = None
    try:
        result = trained_model.val(**val_kwargs)

        if not conf_rec_disable:
            ensure_confidence_recommendations(
                trained_model=trained_model,
                primary_test_result=result,
                model_dir=model_dir,
                data_yaml=data_yaml,
                imgsz=imgsz,
                val_conf=val_conf,
                val_iou=val_iou,
                val_batch=val_batch,
                beta_recall=conf_rec_beta_recall,
                beta_precision=conf_rec_beta_precision,
                fallback_confidence=conf_rec_fallback,
            )

        test_end_time = datetime.now()
        csv_file = save_metrics_csv(result, model_dir)
        target_pt = model_path if os.path.isfile(model_path) else None
        persist_target_test_artifacts_state(
            model_dir,
            format_name="pt",
            target_path=target_pt,
            backend="ultralytics",
            status="ok" if os.path.exists(csv_file) else "incomplete",
        )

        print("\n" + "-" * 60)
        if os.path.exists(csv_file):
            print("[OK] Testing complete.")
            print(f"[INFO] Results saved at path:\n{csv_file}")
        else:
            print("[ERROR].csv file not found. Check Ultralytics log.")
        print("-" * 60 + "\n")
    except Exception as e:
        test_end_time = datetime.now()
        print(f"[ERROR] Failed to test {model_dir} on dataset {dataset_path}: {e}")
        raise
    finally:
        try:
            normalize_ultralytics_run_layout(model_dir)
        except Exception:
            pass

    return test_start_time, test_end_time, inference_record
