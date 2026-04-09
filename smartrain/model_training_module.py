import copy
import json
import os
import argparse
import sys
import traceback
import gc
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from smartrain.cli_argparse import CliArgumentParser
from smartrain.cli_replay import build_non_interactive_command, print_replay_command
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
_ULTRALYTICS_YAML_IGNORED_KEYS = frozenset(
    {
        "data",
        "project",
        "name",
        "exist_ok",
        # В Ultralytics `cfg` может указывать на внешний YAML с гиперпараметрами,
        # но smart-train уже читает заданный пользователем `--ultralytics_yaml`,
        # поэтому `cfg` часто бывает "остатком" и может ссылаться на файл,
        # которого нет на текущей машине.
        "cfg",
        # device мы задаём через окружение/CLI; значения из сохранённых args.yaml
        # (например '0,1,2') часто не соответствуют доступным GPU на машине.
        "device",
        "model_dir",
        "target_path",
        "workspace",
    }
)


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        description="Обучение моделей (без аргументов запускается интерактивный режим)"
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Корень workspace (иначе {WORKSPACE_ENV_VAR}); прогоны в runs/, разрешение --data по datasets",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="YAML-профиль smart-train (базовый конфиг). Можно смешивать с --ultralytics_yaml; приоритет CLI > --ultralytics_yaml > --config",
    )
    parser.add_argument(
        "--ultralytics_yaml",
        type=str,
        default=None,
        help="Внешний Ultralytics args.yaml; несовместимые ключи (data/project/name/exist_ok/...) игнорируются с предупреждением",
    )
    parser.add_argument(
        "--base-run-args-yaml",
        type=str,
        default=None,
        help="Путь к args.yaml базового прогона (используется как источник дефолтов в интерактивном режиме)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Каталог с data.yaml (абсолютный/относительный) или имя записи из datasets/datasets_info.json; "
        "при --workspace обычно задаётся явно (значение data из --ultralytics_yaml не используется)",
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
        "--val-batch",
        type=int,
        default=None,
        help="Batch для val/test (по умолчанию: как batch обучения; для --test-only берётся из training_metadata.json при наличии)",
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


def _prompt_input(label: str, default: str = "", completer=None, show_default_hint: bool = True) -> str:
    from prompt_toolkit import prompt

    prompt_label = f"{label} [default: {default}]: " if (default != "" and show_default_hint) else label
    value = str(prompt(prompt_label, default="", completer=completer, complete_while_typing=True)).strip()
    if value:
        return value
    if default != "":
        if sys.stdin.isatty():
            try:
                sys.stdout.write("\x1b[1A\r")
                sys.stdout.write(f"{prompt_label}{default}\n")
                sys.stdout.flush()
            except Exception:
                print(default)
        else:
            print(default)
    return str(default)


def _prompt_yes_no(label: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    default_text = "y" if default else "n"
    raw = _prompt_input(f"{label} [{suffix}]: ", default=default_text, show_default_hint=False).strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true", "да", "д")


def _prompt_int(label: str, default: int) -> int:
    while True:
        raw = _prompt_input(f"{label}: ", default=str(default)).strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"[ERROR] Ожидается целое число, получено: {raw!r}")


def _prompt_optional_int(label: str, default: int | None = None) -> int | None:
    default_text = "" if default is None else str(default)
    while True:
        raw = _prompt_input(f"{label}: ", default=default_text).strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"[ERROR] Ожидается целое число или пустое значение, получено: {raw!r}")


def _prompt_optional_float(label: str, default: float | None = None) -> float | None:
    default_text = "" if default is None else str(default)
    while True:
        raw = _prompt_input(f"{label}: ", default=default_text).strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print(f"[ERROR] Ожидается число или пустое значение, получено: {raw!r}")


def _load_available_datasets(layout: WorkspaceLayout) -> list[str]:
    info_path = layout.work_datasets_info_path()
    if not os.path.isfile(info_path):
        return []
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    return sorted(str(k) for k in data.keys())


def _prompt_dataset_name(available: list[str]) -> str:
    from prompt_toolkit.completion import WordCompleter

    completer = WordCompleter(available, ignore_case=True)
    while True:
        raw = _prompt_input(
            "Датасет (имя из datasets/datasets_info.json): ",
            default="",
            completer=completer,
        ).strip()
        if raw in available:
            return raw
        print("[ERROR] Неизвестное имя датасета. Доступные:", ", ".join(available))


def _collect_available_base_runs(layout: WorkspaceLayout, selected_dataset: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    runs_root = Path(layout.runs)
    if not runs_root.is_dir():
        return out
    for ds_dir in sorted(runs_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds_name = ds_dir.name
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            args_train = run_dir / "train" / "args.yaml"
            args_root = run_dir / "args.yaml"
            args_path: Path | None = None
            if args_train.is_file():
                args_path = args_train
            elif args_root.is_file():
                args_path = args_root
            if args_path is None:
                continue
            out.append(
                {
                    "dataset": ds_name,
                    "run_dir": str(run_dir),
                    "args_yaml": str(args_path),
                }
            )
    out.sort(key=lambda x: (x["dataset"] != selected_dataset, x["dataset"], x["run_dir"]))
    return out


def _print_available_base_runs(selected_dataset: str, runs: list[dict[str, str]]) -> None:
    if not runs:
        print("[INFO] Базовые прогоны в runs/ не найдены.")
        return
    print("[INFO] Доступные базовые прогоны (сверху — для выбранного датасета):")
    for i, r in enumerate(runs, start=1):
        mark = " [selected-dataset]" if r["dataset"] == selected_dataset else ""
        print(f"  {i:>3}. {r['dataset']} :: {r['run_dir']}{mark}")


def _prompt_base_run_args_yaml(runs: list[dict[str, str]], default_path: str | None = None) -> str | None:
    if not runs:
        return default_path
    while True:
        raw = _prompt_input(
            "Базовый прогон (номер или путь к args.yaml, пусто=без базового): ",
            default=str(default_path or ""),
        ).strip()
        if not raw:
            return default_path
        if os.path.isfile(raw):
            return raw
        try:
            idx = int(raw)
        except ValueError:
            print(f"[ERROR] Ожидается номер прогона или путь к args.yaml, получено: {raw!r}")
            continue
        if 1 <= idx <= len(runs):
            return runs[idx - 1]["args_yaml"]
        print(f"[ERROR] Номер вне диапазона 1..{len(runs)}")


def _get_interactive_default(args, attr: str, fallback, baseline_cfg: dict[str, Any], baseline_key: str):
    if hasattr(args, attr):
        val = getattr(args, attr)
        if val is not None and (fallback is None or val != fallback):
            return val
    if baseline_key in baseline_cfg:
        return baseline_cfg[baseline_key]
    return fallback


def _run_interactive_train_setup(args) -> bool:
    from prompt_toolkit.completion import WordCompleter

    print("[INFO] Интерактивный режим train (Enter = значение по умолчанию).")

    try:
        ws = resolve_workspace_root(getattr(args, "workspace", None))
    except ValueError:
        ws_raw = _prompt_input("Путь workspace: ", default=os.getcwd()).strip()
        if not ws_raw:
            print("[ERROR] Workspace не задан.")
            return False
        ws = os.path.abspath(os.path.expanduser(ws_raw))
        args.workspace = ws

    layout = WorkspaceLayout(ws)
    dataset_names = _load_available_datasets(layout)
    if not dataset_names:
        print(
            "[ERROR] В datasets/datasets_info.json нет доступных датасетов. "
            "Сначала выполните scan."
        )
        return False
    print("[INFO] Доступные датасеты:")
    for name in dataset_names:
        print(f"  - {name}")
    args.data = _prompt_dataset_name(dataset_names)
    baseline_u_cfg: dict[str, Any] = {}
    baseline_sm_opts: dict[str, Any] = {}
    available_runs = _collect_available_base_runs(layout, args.data)
    _print_available_base_runs(args.data, available_runs)
    baseline_args_yaml = _prompt_base_run_args_yaml(
        available_runs,
        default_path=str(getattr(args, "base_run_args_yaml", "") or "") or None,
    )
    args.base_run_args_yaml = baseline_args_yaml
    if baseline_args_yaml:
        try:
            baseline_profile = _load_ultralytics_yaml(baseline_args_yaml)
            baseline_filtered = {
                k: v for k, v in baseline_profile.items() if k not in _ULTRALYTICS_YAML_IGNORED_KEYS
            }
            baseline_u_cfg, baseline_sm_opts = extract_smartrain_options(baseline_filtered)
            print(f"[INFO] Используется базовый прогон: {baseline_args_yaml}")
        except Exception as e:
            print(f"[WARNING] Не удалось прочитать args.yaml базового прогона: {e}")
            baseline_u_cfg, baseline_sm_opts = {}, {}

    args.ultralytics_yaml = (
        _prompt_input(
            "Путь к внешнему Ultralytics args.yaml (--ultralytics_yaml, пусто=не использовать): ",
            default=str(getattr(args, "ultralytics_yaml", "") or ""),
        ).strip()
        or None
    )
    if args.ultralytics_yaml:
        print(
            "[INFO] Для --ultralytics_yaml: data/project/name/exist_ok и служебные path-ключи "
            "игнорируются; data всегда берётся из выбранного датасета."
        )
    ultra_u_cfg: dict[str, Any] = {}
    ultra_sm_opts: dict[str, Any] = {}
    if args.ultralytics_yaml:
        try:
            ultra_profile = _load_ultralytics_yaml(args.ultralytics_yaml)
        except Exception as e:
            print(f"[ERROR] Не удалось прочитать --ultralytics_yaml: {e}")
            return False
        filtered = {
            k: v for k, v in ultra_profile.items() if k not in _ULTRALYTICS_YAML_IGNORED_KEYS
        }
        ultra_u_cfg, ultra_sm_opts = extract_smartrain_options(filtered)

    task_choices = ["detect", "segment", "classify", "pose", "obb"]
    if "task" in ultra_u_cfg:
        args.task = str(ultra_u_cfg["task"])
        print(f"[INFO] Task взят из --ultralytics_yaml: {args.task}")
    else:
        task_default = str(
            _get_interactive_default(args, "task", "detect", baseline_u_cfg, "task")
        )
        task_completer = WordCompleter(task_choices, ignore_case=True)
        args.task = (
            _prompt_input(
                "Task (detect/segment/classify/pose/obb): ",
                default=task_default,
                completer=task_completer,
            ).strip()
            or task_default
        )

    if "model" in ultra_u_cfg:
        args.model = str(ultra_u_cfg["model"])
        print(f"[INFO] Модель взята из --ultralytics_yaml: {args.model}")
    else:
        model_default = str(_get_interactive_default(args, "model", MODEL_VERSION, baseline_u_cfg, "model"))
        args.model = (
            _prompt_input(
                "Модель (--model): ",
                default=model_default,
            ).strip()
            or model_default
        )
    if "epochs" in ultra_u_cfg:
        args.epochs = int(ultra_u_cfg["epochs"])
        print(f"[INFO] Эпохи взяты из --ultralytics_yaml: {args.epochs}")
    else:
        args.epochs = _prompt_int(
            "Эпохи (--epochs)",
            int(_get_interactive_default(args, "epochs", EPOCHS, baseline_u_cfg, "epochs")),
        )
    if "batch" in ultra_u_cfg:
        args.batch = int(ultra_u_cfg["batch"])
        print(f"[INFO] Batch взят из --ultralytics_yaml: {args.batch}")
    else:
        args.batch = _prompt_int(
            "Batch (--batch)",
            int(_get_interactive_default(args, "batch", BATCH, baseline_u_cfg, "batch")),
        )
    if "imgsz" in ultra_u_cfg:
        args.img_size = int(ultra_u_cfg["imgsz"])
        print(f"[INFO] Размер изображения взят из --ultralytics_yaml: {args.img_size}")
    else:
        args.img_size = _prompt_int(
            "Размер изображения (--img-size)",
            int(_get_interactive_default(args, "img_size", IMG_SIZE, baseline_u_cfg, "imgsz")),
        )

    default_target = str(getattr(args, "target_path", None) or layout.runs)
    args.target_path = (_prompt_input("Каталог прогонов (--target-path): ", default=default_target).strip()
                        or default_target)

    args.test_only = _prompt_yes_no("Только тест без обучения (--test-only)?", default=bool(getattr(args, "test_only", False)))
    if args.test_only:
        model_dir_default = str(getattr(args, "model_dir", "") or "")
        while True:
            model_dir = _prompt_input("Путь к модели (--model-dir): ", default=model_dir_default).strip()
            if model_dir:
                args.model_dir = model_dir
                break
            print("[ERROR] Для --test-only необходимо указать --model-dir.")
    else:
        args.model_dir = getattr(args, "model_dir", None)

    args.val_imgsz = _prompt_optional_int(
        "Размер val/test (--val-imgsz, пусто=как train)",
        _get_interactive_default(args, "val_imgsz", None, baseline_u_cfg, "imgsz"),
    )
    args.val_conf = _prompt_optional_float(
        "Порог conf (--val-conf, пусто=по умолчанию Ultralytics)",
        _get_interactive_default(args, "val_conf", None, baseline_u_cfg, "conf"),
    )
    args.val_iou = _prompt_optional_float(
        "Порог IoU (--val-iou, пусто=по умолчанию Ultralytics)",
        _get_interactive_default(args, "val_iou", None, baseline_u_cfg, "iou"),
    )

    if "weighted_sampling" in ultra_sm_opts:
        args.weighted_sampling = bool(ultra_sm_opts["weighted_sampling"])
    else:
        args.weighted_sampling = _prompt_yes_no(
            "Включить weighted sampling (--weighted-sampling)?",
            default=bool(_get_interactive_default(args, "weighted_sampling", False, baseline_sm_opts, "weighted_sampling")),
        )
    if "export_onnx" in ultra_sm_opts:
        args.export_onnx = bool(ultra_sm_opts["export_onnx"])
    else:
        args.export_onnx = _prompt_yes_no(
            "Экспортировать ONNX после обучения (--export-onnx)?",
            default=bool(_get_interactive_default(args, "export_onnx", False, baseline_sm_opts, "export_onnx")),
        )
    if "export_onnx_half" in ultra_sm_opts:
        args.export_onnx_fp32 = not bool(ultra_sm_opts["export_onnx_half"])
    else:
        default_fp32 = bool(getattr(args, "export_onnx_fp32", False))
        if "export_onnx_half" in baseline_sm_opts:
            default_fp32 = not bool(baseline_sm_opts["export_onnx_half"])
        args.export_onnx_fp32 = _prompt_yes_no(
            "Использовать FP32 для ONNX (--export-onnx-fp32)?",
            default=default_fp32,
        )
    if "clearml" in ultra_sm_opts:
        args.clearml = bool(ultra_sm_opts["clearml"])
    else:
        args.clearml = _prompt_yes_no(
            "Логировать в ClearML (--clearml)?",
            default=bool(_get_interactive_default(args, "clearml", False, baseline_sm_opts, "clearml")),
        )
    if args.clearml:
        if "clearml_project" in ultra_sm_opts:
            args.clearml_project = str(ultra_sm_opts["clearml_project"]).strip() or None
        else:
            default_cm_project = str(getattr(args, "clearml_project", "") or "")
            if "clearml_project" in baseline_sm_opts:
                default_cm_project = str(baseline_sm_opts["clearml_project"] or "")
            args.clearml_project = (
                _prompt_input(
                    "Проект ClearML (--clearml-project): ",
                    default=default_cm_project,
                ).strip()
                or None
            )
    args.non_interactive = _prompt_yes_no(
        "Не спрашивать подтверждения при существующей папке (--yes)?",
        default=bool(getattr(args, "non_interactive", False)),
    )
    return True


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
            f"Имя датасета {data_arg!r} отсутствует в datasets/{DATASETS_INFO_FILE}.{hint}"
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


def _pick_split_relative_dir(dataset_path: str, split_aliases: tuple[str, ...]) -> str | None:
    """
    Ищет директорию split внутри выбранного dataset_path.
    Возвращает относительный путь (предпочтительно с images/) или None.
    """
    candidates: list[str] = []
    for split in split_aliases:
        candidates.extend([f"{split}/images", f"images/{split}", split])
    for rel in candidates:
        abs_p = os.path.join(dataset_path, rel)
        if os.path.isdir(abs_p):
            return rel
    return None


def _build_runtime_data_yaml(dataset_path: str, run_dir: str, *, stage: str) -> str:
    """
    Создаёт служебный data.yaml для Ultralytics с привязкой к текущему dataset_path.
    Это защищает от старых абсолютных путей в исходном data.yaml (другая машина).
    """
    src_yaml = os.path.join(dataset_path, "data.yaml")
    with open(src_yaml, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Некорректный YAML формата data.yaml: {src_yaml}")

    train_rel = _pick_split_relative_dir(dataset_path, ("train",))
    val_rel = _pick_split_relative_dir(dataset_path, ("val", "valid"))
    test_rel = _pick_split_relative_dir(dataset_path, ("test",))
    if train_rel is None or val_rel is None:
        raise FileNotFoundError(
            f"Не найдены обязательные split-папки train/val внутри {dataset_path}."
        )

    runtime_cfg = dict(raw)
    runtime_cfg["path"] = dataset_path
    runtime_cfg["train"] = train_rel
    runtime_cfg["val"] = val_rel
    if test_rel is not None:
        runtime_cfg["test"] = test_rel

    out_yaml = os.path.join(run_dir, f"_runtime_data_{stage}.yaml")
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_cfg, f, allow_unicode=True, sort_keys=False)
    print(
        f"[INFO] Runtime data.yaml ({stage}) сформирован для выбранного датасета: {out_yaml}"
    )
    return out_yaml


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
    overwritten: list[str] = []
    if "data" in k:
        overwritten.append("data")
    if "project" in k:
        overwritten.append("project")
    if "name" in k:
        overwritten.append("name")
    if "exist_ok" in k:
        overwritten.append("exist_ok")
    k.pop("data", None)
    k["data"] = data_yaml
    k["project"] = model_dir
    k["name"] = "train"
    k["exist_ok"] = False
    k.setdefault("mode", "train")
    if overwritten:
        print(
            "[WARNING] Принудительно переопределены служебные ключи train: "
            + ", ".join(sorted(set(overwritten)))
        )
    return k


def _load_ultralytics_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = load_train_profile(path)
    if not isinstance(raw, dict):
        return {}
    return raw


def _merge_sources_with_priority(
    *,
    config_profile: dict[str, Any],
    ultralytics_profile: dict[str, Any],
    args: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Base: --config
    u_cfg, sm_opts = extract_smartrain_options(config_profile)

    # Overlay: --ultralytics_yaml (minus ignored keys)
    if ultralytics_profile:
        ignored = sorted(k for k in ultralytics_profile.keys() if k in _ULTRALYTICS_YAML_IGNORED_KEYS)
        if ignored:
            print(
                "[WARNING] --ultralytics_yaml: проигнорированы ключи: "
                + ", ".join(ignored)
            )
        filtered = {k: v for k, v in ultralytics_profile.items() if k not in _ULTRALYTICS_YAML_IGNORED_KEYS}
        u_from_ultra, sm_from_ultra = extract_smartrain_options(filtered)
        cli_key_map = {
            "model": "model",
            "epochs": "epochs",
            "batch": "batch",
            "imgsz": "img_size",
            "task": "task",
        }
        overridden_by_cli: list[str] = []
        for yaml_key, cli_attr in cli_key_map.items():
            if yaml_key in u_from_ultra and hasattr(args, cli_attr):
                overridden_by_cli.append(yaml_key)
        if overridden_by_cli:
            print(
                "[WARNING] --ultralytics_yaml: следующие ключи будут переопределены CLI: "
                + ", ".join(sorted(overridden_by_cli))
            )
        u_cfg.update(u_from_ultra)
        sm_opts.update(sm_from_ultra)
    return u_cfg, sm_opts


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

    data_yaml = _build_runtime_data_yaml(dataset_path, target_dir, stage="train")
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
    val_batch=None,
):
    test_start_time = datetime.now()

    data_yaml = _build_runtime_data_yaml(dataset_path, model_dir, stage="test")
    imgsz = val_imgsz if val_imgsz is not None else train_img_size

    inference_record = {
        "imgsz": imgsz,
        "conf": val_conf,
        "iou": val_iou,
        "batch": val_batch,
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
    if val_batch is not None:
        val_kwargs["batch"] = int(val_batch)

    print("\n" + "=" * 60)
    print(f"[INFO] Тестирование модели: {model_dir}")
    print(f"[INFO] Датасет: {dataset_path}")
    print(f"[INFO] Конфигурация: {data_yaml}")
    print(f"[INFO] Сохранение результатов в {model_dir}")
    if imgsz is not None:
        print(f"[INFO] val imgsz={imgsz}, batch={val_batch}, conf={val_conf}, iou={val_iou}")
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


def _load_batch_from_training_metadata(model_dir: str) -> int | None:
    """
    В режиме --test-only хотим тестировать с тем же batch, что был при обучении.
    Берём из training_metadata.json если файл есть и формат ожидаемый.
    """
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


def _maybe_free_cuda_memory() -> None:
    """
    Смягчение OOM между train и val/test в одном процессе.
    """
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Иногда помогает собрать IPC кэш, но не обязателен.
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        # torch может быть недоступен в окружениях без GPU/torch; это не критично.
        pass


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    parser = build_train_arg_parser()
    interactive_mode = len(argv) == 0
    replay_cmd = None
    if interactive_mode:
        if not sys.stdin.isatty():
            print(
                "[ERROR] Интерактивный режим train требует терминал (TTY). "
                "Либо запустите в терминале, либо передайте аргументы."
            )
            return
        try:
            ok = _run_interactive_train_setup(args)
        except Exception as e:
            print(f"[ERROR] Ошибка интерактивного режима train: {e}")
            return
        if not ok:
            return
        replay_cmd = build_non_interactive_command("train", parser, args)
        print_replay_command("перед запуском", replay_cmd)

    profile = load_train_profile(args.config) if args.config else {}
    ultra_profile = _load_ultralytics_yaml(getattr(args, "ultralytics_yaml", None))
    u_cfg, sm_opts = _merge_sources_with_priority(
        config_profile=profile,
        ultralytics_profile=ultra_profile,
        args=args,
    )
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

    if getattr(args, "export_onnx_fp32", False):
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
                _maybe_free_cuda_memory()
                val_batch = args.val_batch if args.val_batch is not None else batch
                test_start_time, test_end_time, inference_info = test_yolo(
                    model_dir,
                    data,
                    training_start_time=training_start_time,
                    training_end_time=training_end_time,
                    train_img_size=img_size,
                    val_imgsz=args.val_imgsz,
                    val_conf=args.val_conf,
                    val_iou=args.val_iou,
                    val_batch=val_batch,
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
                val_batch = (
                    args.val_batch
                    if args.val_batch is not None
                    else (_load_batch_from_training_metadata(model_dir) or batch)
                )
                test_start_time, test_end_time, inference_info = test_yolo(
                    model_dir,
                    data,
                    train_img_size=img_size,
                    val_imgsz=args.val_imgsz,
                    val_conf=args.val_conf,
                    val_iou=args.val_iou,
                    val_batch=val_batch,
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
    if replay_cmd:
        print_replay_command("после выполнения", replay_cmd)


if __name__ == "__main__":
    main()
