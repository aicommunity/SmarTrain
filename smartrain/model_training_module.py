import copy
import json
import os
import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from smartrain.cli_argparse import CliArgumentParser
from smartrain.dataset_hash import calculate_dataset_hash
from smartrain.train_profile import (
    apply_cli_smartrain_overrides,
    dataset_root_from_data_yaml,
    extract_smartrain_options,
    load_train_profile,
    merge_cli_into_ultralytics_cfg,
    resolve_profile_data_path,
    task_to_metadata_task_type,
)
from smartrain.workspace_paths import (
    WORKSPACE_ENV_VAR,
    WorkspaceLayout,
    resolve_workspace_root,
    resolve_dataset_root,
    DATASETS_INFO_FILE,
)


MODEL_VERSION = "yolov8n"
EPOCHS = 50
BATCH = 16
IMG_SIZE = 640


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(description="Обучение моделей")

    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Корень workspace (иначе {WORKSPACE_ENV_VAR}); прогоны в runs/, разрешение --data по work_datasets",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="YAML-профиль гиперпараметров Ultralytics (формат как у model.train()); CLI переопределяет model/epochs/batch/img-size/task",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Каталог с data.yaml (абсолютный/относительный) или имя записи из work_datasets/datasets_info.json; "
        "если задан --config, можно опустить при наличии data: в YAML",
    )

    parser.add_argument(
        "--task",
        type=str,
        default=argparse.SUPPRESS,
        help="Задача Ultralytics: detect, segment, classify, pose, obb (по умолчанию из профиля или detect)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=argparse.SUPPRESS,
        help=f"Модель (по умолчанию {MODEL_VERSION} или из профиля --config)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Эпохи (по умолчанию {EPOCHS} или из профиля)",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=argparse.SUPPRESS,
        help=f"Batch (по умолчанию {BATCH} или из профиля)",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=argparse.SUPPRESS,
        help=f"imgsz (по умолчанию {IMG_SIZE} или из профиля)",
    )

    parser.add_argument(
        "--target-path",
        type=str,
        default=None,
        help="Базовый каталог для прогонов (по умолчанию workspace/runs при использовании workspace)",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Путь к папке с моделью",
    )

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Выполнить только тестирование без обучения",
    )

    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        dest="non_interactive",
        help="Не спрашивать подтверждение при существующей папке результатов (для очереди и CI)",
    )

    parser.add_argument(
        "--val-imgsz",
        type=int,
        default=None,
        help="Размер изображения для val/test (по умолчанию как --img-size при обучении)",
    )
    parser.add_argument(
        "--val-conf",
        type=float,
        default=None,
        help="Порог conf для val() (Ultralytics)",
    )
    parser.add_argument(
        "--val-iou",
        type=float,
        default=None,
        help="Порог IoU для val() (Ultralytics)",
    )

    parser.add_argument(
        "--weighted-sampling",
        action="store_true",
        help="Взвешенная выборка изображений (классы с меньшим числом объектов чаще); патч ultralytics",
    )

    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="После успешного обучения экспорт best.pt в ONNX",
    )
    parser.add_argument(
        "--export-onnx-fp32",
        action="store_true",
        help="При --export-onnx не использовать half=True",
    )

    parser.add_argument(
        "--clearml",
        action="store_true",
        help="Логирование гиперпараметров в ClearML (нужен pip install clearml)",
    )
    parser.add_argument(
        "--clearml-project",
        type=str,
        default=None,
        help="Имя проекта ClearML (иначе CLEARML_PROJECT или smartrain)",
    )

    return parser


def parse_args(argv=None):
    return build_train_arg_parser().parse_args(argv)


def resolve_training_data_path(layout: WorkspaceLayout, data_arg: str) -> str:
    expanded = os.path.abspath(os.path.expanduser(data_arg))
    yaml_here = os.path.join(expanded, "data.yaml")
    if os.path.isdir(expanded) and os.path.isfile(yaml_here):
        return expanded
    info_path = layout.work_datasets_info_path()
    if not os.path.isfile(info_path):
        raise FileNotFoundError(
            f"Каталог с data.yaml для {data_arg!r} не найден и отсутствует {info_path}."
        )
    with open(info_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    if not isinstance(catalog, dict):
        raise ValueError(f"{info_path}: ожидается объект JSON.")
    if data_arg not in catalog:
        names = ", ".join(sorted(catalog.keys()))
        hint = f" Известные имена: {names}." if names else ""
        raise ValueError(
            f"Имя датасета {data_arg!r} отсутствует в work_datasets/{DATASETS_INFO_FILE}.{hint}"
        )
    entry = catalog[data_arg]
    if not isinstance(entry, dict):
        raise ValueError(f"Запись {data_arg!r} должна быть объектом JSON.")
    return resolve_dataset_root(layout.root, data_arg, entry, layout.work_datasets)


def _validate_dataset_dir(dataset_path: str) -> None:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Папка с датасетом не найдена: {dataset_path}")
    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Не найден yaml файл: {data_yaml}")


def _resolve_cli_paths_with_profile(args, u_cfg: dict) -> tuple[str | None, str, str]:
    """
    workspace, dataset_root (каталог с data.yaml), target_base.
    """
    try:
        ws = resolve_workspace_root(args.workspace)
    except ValueError:
        ws = None

    if ws is not None:
        layout = WorkspaceLayout(ws)
        os.makedirs(layout.runs, exist_ok=True)
        if args.data is not None:
            dataset_path = resolve_training_data_path(layout, args.data)
        elif u_cfg.get("data"):
            yp = resolve_profile_data_path(str(u_cfg["data"]))
            dataset_path = dataset_root_from_data_yaml(yp)
        else:
            raise ValueError(
                "При использовании workspace укажите --data или поле data: в профиле --config."
            )
        if args.target_path is not None:
            target_base = os.path.abspath(os.path.expanduser(args.target_path))
        else:
            target_base = layout.runs
        return ws, dataset_path, target_base

    if args.data is not None:
        dataset_path = os.path.abspath(os.path.expanduser(args.data))
    elif u_cfg.get("data"):
        yp = resolve_profile_data_path(str(u_cfg["data"]))
        dataset_path = dataset_root_from_data_yaml(yp)
    else:
        raise ValueError(
            f"Задайте --workspace (или {WORKSPACE_ENV_VAR}) и --data (или data в YAML), "
            "либо без workspace — --data и --target-path."
        )

    if args.target_path is None:
        raise ValueError(
            f"Без workspace укажите --target-path (базовый каталог прогонов) или задайте {WORKSPACE_ENV_VAR}."
        )
    target_base = os.path.abspath(os.path.expanduser(args.target_path))
    return None, dataset_path, target_base


def _finalize_train_kwargs(ultralytics_cfg: dict[str, Any], data_yaml: str, model_dir: str) -> dict[str, Any]:
    k = copy.deepcopy(ultralytics_cfg)
    k.pop("data", None)
    k["data"] = data_yaml
    k["project"] = model_dir
    k["name"] = "train"
    k["exist_ok"] = False
    k.setdefault("mode", "train")
    return k


def train_yolo(
    dataset_path: str,
    target_dir: str,
    non_interactive: bool = False,
    workspace_root: str | None = None,
    ultralytics_cfg: dict[str, Any] | None = None,
    smartrain_opts: dict[str, Any] | None = None,
):
    ultralytics_cfg = ultralytics_cfg or {}
    smartrain_opts = smartrain_opts or {}

    training_start_time = datetime.now()
    _validate_dataset_dir(dataset_path)

    data_yaml = os.path.join(dataset_path, "data.yaml")
    dataset_name = os.path.basename(os.path.normpath(dataset_path))

    model_version = str(ultralytics_cfg.get("model", MODEL_VERSION))
    epochs = int(ultralytics_cfg.get("epochs", EPOCHS))

    try:
        dataset_hash = calculate_dataset_hash(dataset_path)
        print(f"[INFO] Хеш датасета: {dataset_hash}")
    except Exception as e:
        print(f"[WARNING] Не удалось вычислить хеш датасета: {e}")
        dataset_hash = None

    timestamp_str = training_start_time.strftime("%Y-%m-%d_%H-%M")
    folder_name = f"{timestamp_str}_{model_version.replace('.pt', '')}_{epochs}epochs"
    if dataset_hash:
        folder_name = f"{folder_name}-{dataset_hash}"

    model_dir = os.path.join(target_dir, dataset_name, folder_name)

    if os.path.exists(model_dir):
        if non_interactive:
            print(f"[INFO] Папка уже существует, продолжаем без запроса: {model_dir}")
        else:
            while True:
                answer = input(
                    f"[WARNING] Папка с таким названием уже существует: {model_dir}. Продолжить обучение? (y/n): \n"
                ).strip().lower()
                if answer == "y":
                    break
                elif answer == "n":
                    sys.exit(1)
                else:
                    print("Пожалуйста, введите только 'y' или 'n'.\n")
    else:
        os.makedirs(model_dir, exist_ok=True)

    train_kw = _finalize_train_kwargs(ultralytics_cfg, data_yaml, model_dir)

    clearml_task = None
    if smartrain_opts.get("clearml"):
        try:
            from clearml import Task
        except ImportError as e:
            raise ImportError(
                "Для --clearml установите: pip install 'smartrain[clearml]' или pip install clearml"
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

    if smartrain_opts.get("weighted_sampling"):
        from smartrain.weighted_yolo_dataset import setup_weighted_sampling_env

        setup_weighted_sampling_env()

    print("\n" + "=" * 60)
    print(f"[INFO] Обучение модели: {model_kw_model(train_kw)}")
    print(f"[INFO] Датасет: {dataset_name}")
    print(f"[INFO] Задача (task): {train_kw.get('task', 'detect')}")
    print(f"[INFO] Конфигурация: {data_yaml}")
    print(f"[INFO] Сохранение результатов в {model_dir}")
    print("=" * 60 + "\n")

    _, model_ext = os.path.splitext(str(train_kw.get("model", "")))
    spec = train_kw["model"]
    if model_ext == "":
        model = YOLO(str(spec) + ".pt")
    else:
        model = YOLO(str(spec))

    if smartrain_opts.get("weighted_sampling"):
        from smartrain.weighted_yolo_dataset import register_weighted_sampling_callback

        register_weighted_sampling_callback(model)

    training_end_time = None
    onnx_rel = None
    best_path = os.path.join(model_dir, "train", "weights", "best.pt")
    try:
        model.train(**train_kw)
        training_end_time = datetime.now()
        model_path = best_path
        print("\n" + "-" * 60)
        if os.path.exists(model_path):
            print("[OK] Обучение завершено.")
            print(f"[INFO] Модель сохранена по пути:\n{model_path}")
        if smartrain_opts.get("export_onnx") and os.path.exists(model_path):
            half = bool(smartrain_opts.get("export_onnx_half", True))
            simplify = bool(smartrain_opts.get("export_onnx_simplify", True))
            opset = smartrain_opts.get("export_onnx_opset", 17)
            dynamic = bool(smartrain_opts.get("export_onnx_dynamic", False))
            try:
                ex = model.export(
                    format="onnx",
                    dynamic=dynamic,
                    simplify=simplify,
                    opset=int(opset),
                    half=half,
                )
                if ex:
                    onnx_abs = str(ex) if isinstance(ex, (str, Path)) else str(getattr(ex, "path", ex))
                    if os.path.isfile(onnx_abs):
                        onnx_rel = os.path.relpath(onnx_abs, model_dir)
                    else:
                        cand = os.path.join(model_dir, "train", "weights", "best.onnx")
                        if os.path.isfile(cand):
                            onnx_rel = os.path.relpath(cand, model_dir)
                print(f"[INFO] ONNX экспорт выполнен: {onnx_rel or '(см. каталог weights)'}")
            except Exception as ex_err:
                print(f"[WARNING] ONNX экспорт не удался: {ex_err}")
    except Exception as e:
        training_end_time = datetime.now()
        print(
            f"[ERROR] Не удалось запустить обучение {model_version} на датасете {dataset_name} "
            f"на {epochs} эпох: {e}"
        )
    finally:
        if clearml_task is not None:
            try:
                clearml_task.close()
            except Exception:
                pass

    meta_extras = {
        "train_kw": {k: v for k, v in train_kw.items() if k != "data"},
        "task_type": task_to_metadata_task_type(train_kw.get("task")),
        "onnx_relative": onnx_rel,
        "training_ok": os.path.isfile(best_path),
    }
    return model_dir, training_start_time, training_end_time, dataset_hash, workspace_root, meta_extras


def model_kw_model(train_kw: dict) -> str:
    return str(train_kw.get("model", ""))


def test_yolo(
    model_dir,
    dataset_path,
    training_start_time=None,
    training_end_time=None,
    train_img_size=None,
    val_imgsz=None,
    val_conf=None,
    val_iou=None,
):
    test_start_time = datetime.now()

    data_yaml = os.path.join(dataset_path, "data.yaml")
    imgsz = val_imgsz if val_imgsz is not None else train_img_size

    inference_record = {
        "imgsz": imgsz,
        "conf": val_conf,
        "iou": val_iou,
    }

    model_path = os.path.join(model_dir, "train", "weights", "best.pt")
    trained_model = YOLO(model_path)

    val_kwargs = {
        "data": data_yaml,
        "split": "test",
        "project": model_dir,
        "name": "test",
        "exist_ok": False,
    }
    if imgsz is not None:
        val_kwargs["imgsz"] = imgsz
    if val_conf is not None:
        val_kwargs["conf"] = val_conf
    if val_iou is not None:
        val_kwargs["iou"] = val_iou

    print("\n" + "=" * 60)
    print(f"[INFO] Тестирование модели: {model_dir}")
    print(f"[INFO] Датасет: {dataset_path}")
    print(f"[INFO] Конфигурация: {data_yaml}")
    print(f"[INFO] Сохранение результатов в {model_dir}")
    if imgsz is not None:
        print(f"[INFO] val imgsz={imgsz}, conf={val_conf}, iou={val_iou}")
    print("=" * 60 + "\n")

    test_end_time = None
    try:
        result = trained_model.val(**val_kwargs)

        test_end_time = datetime.now()
        csv_file = save_metrics_csv(result, model_dir)

        print("\n" + "-" * 60)
        if os.path.exists(csv_file):
            print("[OK] Тестирование завершено.")
            print(f"[INFO] Результаты сохранены по пути:\n{csv_file}")
        else:
            print("[ERROR] .csv файл не найден. Проверьте лог Ultralytics.")
        print("-" * 60 + "\n")
    except Exception as e:
        test_end_time = datetime.now()
        print(f"[ERROR] Не удалось протестировать {model_dir} на датасете {dataset_path}: {e}")

    return test_start_time, test_end_time, inference_record


def save_metrics_csv(test_result, model_dir):
    base_name = "test_metrics"
    ext = ".csv"
    csv_file = os.path.join(model_dir, base_name + ext)

    counter = 1
    while os.path.exists(csv_file):
        csv_file = os.path.join(model_dir, f"{base_name}_{counter}{ext}")
        counter += 1

    csv_data = test_result.to_csv()
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(csv_data)

    return csv_file


def _relative_to_workspace(path: str, workspace_root: str) -> str:
    ap = os.path.abspath(path)
    wr = os.path.abspath(workspace_root)
    try:
        return os.path.relpath(ap, wr)
    except ValueError:
        return ap


def save_training_metadata(
    model_dir,
    dataset_path,
    model_version=None,
    training_start_time=None,
    training_end_time=None,
    test_start_time=None,
    test_end_time=None,
    epochs=None,
    batch=None,
    img_size=None,
    training_success=True,
    training_error=None,
    test_success=True,
    test_error=None,
    dataset_hash=None,
    inference=None,
    workspace_root=None,
    task_type=None,
    ultralytics_train_summary=None,
    onnx_relative=None,
):
    metadata = {
        "training_info": {
            "framework": "ultralytics",
            "task_type": task_type or "detection",
            "model": model_version,
            "dataset": {
                "name": os.path.basename(os.path.normpath(dataset_path)),
                "path_absolute": os.path.abspath(dataset_path),
                "path_relative": _get_relative_path(dataset_path, model_dir),
                "hash": dataset_hash,
            },
            "hyperparameters": {
                "epochs": epochs,
                "batch_size": batch,
                "image_size": img_size,
            },
        },
        "timestamps": {
            "training": {
                "start": training_start_time.isoformat() if training_start_time else None,
                "end": training_end_time.isoformat() if training_end_time else None,
                "duration_seconds": (training_end_time - training_start_time).total_seconds()
                if training_start_time and training_end_time
                else None,
            },
            "testing": {
                "start": test_start_time.isoformat() if test_start_time else None,
                "end": test_end_time.isoformat() if test_end_time else None,
                "duration_seconds": (test_end_time - test_start_time).total_seconds()
                if test_start_time and test_end_time
                else None,
            },
        },
        "status": {
            "training": {
                "success": training_success,
                "error": training_error,
            },
            "testing": {
                "success": test_success,
                "error": test_error,
            },
        },
        "paths": {
            "model_directory": ".",
            "best_model": "train/weights/best.pt"
            if os.path.exists(os.path.join(model_dir, "train", "weights", "best.pt"))
            else None,
        },
    }

    if ultralytics_train_summary:
        metadata["training_info"]["ultralytics_train"] = ultralytics_train_summary
    if onnx_relative:
        metadata["paths"]["onnx"] = onnx_relative

    if workspace_root is not None:
        metadata["workspace"] = {
            "root": os.path.abspath(workspace_root),
            "dataset_path_relative": _relative_to_workspace(dataset_path, workspace_root),
            "run_directory_relative": _relative_to_workspace(model_dir, workspace_root),
        }

    if inference:
        metadata["inference"] = {k: v for k, v in inference.items() if v is not None}

    metadata_file = os.path.join(model_dir, "training_metadata.json")

    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Метаданные обучения сохранены: {metadata_file}")
    except Exception as e:
        print(f"[WARNING] Не удалось сохранить метаданные: {e}")


def _get_relative_path(target_path, base_path):
    try:
        target = Path(os.path.abspath(target_path))
        base = Path(os.path.abspath(base_path))

        try:
            relative = os.path.relpath(target, base)
            return relative
        except ValueError:
            return target.as_posix()
    except Exception:
        return os.path.abspath(target_path)


def _json_safe_train_summary(train_kw: dict[str, Any] | None) -> dict[str, Any] | None:
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


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    profile = load_train_profile(args.config) if args.config else {}
    u_cfg, sm_opts = extract_smartrain_options(profile)
    merge_cli_into_ultralytics_cfg(
        u_cfg,
        model=getattr(args, "model", None),
        epochs=getattr(args, "epochs", None),
        batch=getattr(args, "batch", None),
        imgsz=getattr(args, "img_size", None),
        task=getattr(args, "task", None),
        defaults={
            "model": MODEL_VERSION,
            "epochs": EPOCHS,
            "batch": BATCH,
            "imgsz": IMG_SIZE,
            "task": "detect",
        },
    )
    apply_cli_smartrain_overrides(sm_opts, args)

    if args.export_onnx_fp32:
        sm_opts["export_onnx_half"] = False

    try:
        workspace_root, data, target_dir = _resolve_cli_paths_with_profile(args, u_cfg)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    u_cfg.pop("data", None)

    model_version = str(u_cfg.get("model", MODEL_VERSION))
    epochs = int(u_cfg.get("epochs", EPOCHS))
    batch = int(u_cfg.get("batch", BATCH))
    img_size = u_cfg.get("imgsz", IMG_SIZE)
    try:
        img_size = int(img_size) if img_size is not None else IMG_SIZE
    except (TypeError, ValueError):
        img_size = IMG_SIZE

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

    if not args.test_only:
        try:
            (
                model_dir,
                training_start_time,
                training_end_time,
                dataset_hash,
                _,
                meta_extras,
            ) = train_yolo(
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
            print(f"[ERROR] Ошибка при обучении: {e}")
            training_error = f"{str(e)}\n{traceback.format_exc()}"
            try:
                dataset_hash = calculate_dataset_hash(data)
            except Exception:
                dataset_hash = None
            if not model_dir:
                dataset_name = os.path.basename(os.path.normpath(data))
                timestamp_str = (
                    training_start_time.strftime("%Y-%m-%d_%H-%M")
                    if training_start_time
                    else datetime.now().strftime("%Y-%m-%d_%H-%M")
                )
                folder_name = f"{timestamp_str}_{model_version.replace('.pt', '')}_{epochs}epochs"
                if dataset_hash:
                    folder_name = f"{folder_name}-{dataset_hash}"
                model_dir = os.path.join(target_dir, dataset_name, folder_name)
                os.makedirs(model_dir, exist_ok=True)
            meta_extras = {
                "task_type": task_to_metadata_task_type(u_cfg.get("task")),
                "train_kw": {k: v for k, v in u_cfg.items() if k != "data"},
                "training_ok": False,
            }

        if training_success and model_dir:
            try:
                test_start_time, test_end_time, inference_info = test_yolo(
                    model_dir,
                    data,
                    training_start_time=training_start_time,
                    training_end_time=training_end_time,
                    train_img_size=img_size,
                    val_imgsz=args.val_imgsz,
                    val_conf=args.val_conf,
                    val_iou=args.val_iou,
                )
            except Exception as e:
                test_success = False
                test_error = str(e)
                test_end_time = datetime.now()
                print(f"[ERROR] Ошибка при тестировании: {e}")
                test_error = f"{str(e)}\n{traceback.format_exc()}"

        if model_dir:
            save_training_metadata(
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
                ultralytics_train_summary=_json_safe_train_summary(meta_extras.get("train_kw")),
                onnx_relative=meta_extras.get("onnx_relative"),
            )
    else:
        model_dir = args.model_dir
        if model_dir:
            try:
                test_start_time, test_end_time, inference_info = test_yolo(
                    model_dir,
                    data,
                    train_img_size=img_size,
                    val_imgsz=args.val_imgsz,
                    val_conf=args.val_conf,
                    val_iou=args.val_iou,
                )
            except Exception as e:
                test_success = False
                test_error = str(e)
                test_end_time = datetime.now()
                print(f"[ERROR] Ошибка при тестировании: {e}")
                test_error = f"{str(e)}\n{traceback.format_exc()}"

            save_training_metadata(
                model_dir=model_dir,
                dataset_path=data,
                test_start_time=test_start_time,
                test_end_time=test_end_time,
                test_success=test_success,
                test_error=test_error,
                inference=inference_info,
                workspace_root=workspace_root,
                task_type=task_to_metadata_task_type(u_cfg.get("task")),
            )
        else:
            print("[ERROR] Не указан путь к модели")


if __name__ == "__main__":
    main()
