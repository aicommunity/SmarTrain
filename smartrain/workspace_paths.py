"""
Единый корень workspace: подкаталоги и резолв путей к датасетам (data_path / каталог по ключу).
"""
from __future__ import annotations

import json
import os
import hashlib
import shutil
import zipfile
from typing import Any

WORKSPACE_ENV_VAR = "SMART_TRAIN_WORKSPACE"

DATASETS_INFO_FILE = "datasets_info.json"
CLASS_NAMES_FILE = "class_names.json"
WORKSPACE_QUEUE_BASENAME = "queue.txt"


class WorkspaceLayout:
    """Все стандартные пути внутри workspace; root — абсолютный нормализованный путь."""

    def __init__(self, root: str):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.source_datasets = os.path.join(self.root, "source_datasets")
        self.work_datasets = os.path.join(self.root, "work_datasets")
        self.runs = os.path.join(self.root, "runs")
        self.analytics = os.path.join(self.root, "analytics")
        self.models = os.path.join(self.root, "models")
        self.extracted_datasets = os.path.join(self.root, "tmp", "extracted_datasets")

    def source_datasets_info_path(self) -> str:
        return os.path.join(self.source_datasets, DATASETS_INFO_FILE)

    def source_class_names_path(self) -> str:
        return os.path.join(self.source_datasets, CLASS_NAMES_FILE)

    def work_datasets_info_path(self) -> str:
        return os.path.join(self.work_datasets, DATASETS_INFO_FILE)

    def work_class_names_path(self) -> str:
        return os.path.join(self.work_datasets, CLASS_NAMES_FILE)


def resolve_workspace_root(cli_workspace: str | None) -> str:
    """
    Источник корня: аргумент CLI (непустой) перекрывает переменную окружения SMART_TRAIN_WORKSPACE.
    Иначе — явная ошибка.
    """
    if cli_workspace is not None:
        w = cli_workspace.strip()
        if w:
            return os.path.abspath(os.path.expanduser(w))
    env_val = os.environ.get(WORKSPACE_ENV_VAR)
    if env_val is not None:
        e = env_val.strip()
        if e:
            return os.path.abspath(os.path.expanduser(e))
    raise ValueError(
        "Не задан корень workspace: укажите --workspace или переменную окружения "
        f"{WORKSPACE_ENV_VAR}."
    )


def resolve_path_under_workspace(workspace_root: str, relative_or_absolute: str) -> str:
    """Абсолютный путь как есть; иначе путь относительно workspace_root."""
    p = relative_or_absolute.strip()
    if not p:
        raise ValueError("Пустой data_path.")
    if os.path.isabs(p):
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(workspace_root, os.path.normpath(p)))


def resolve_dataset_root(
    workspace_root: str,
    entry_key: str,
    entry_dict: dict,
    catalog_dir: str,
) -> str:
    """
    Если в записи есть ключ data_path — резолвим от workspace или абсолют.
    Иначе корень данных: catalog_dir / entry_key.
    """
    if "data_path" in entry_dict:
        raw = entry_dict["data_path"]
        if not isinstance(raw, str):
            raise TypeError(f"data_path для {entry_key!r} должен быть строкой.")
        return resolve_path_under_workspace(workspace_root, raw)
    return os.path.join(catalog_dir, entry_key)


def _safe_extract_zip(zip_path: str, target_dir: str) -> None:
    """
    Безопасная распаковка zip в target_dir с защитой от path traversal.
    """
    abs_target = os.path.abspath(target_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_name = member.filename
            out_path = os.path.abspath(os.path.join(abs_target, member_name))
            if not out_path.startswith(abs_target + os.sep) and out_path != abs_target:
                raise ValueError(f"Архив содержит небезопасный путь: {member_name!r}")
        zf.extractall(abs_target)


def _choose_extracted_dataset_root(extract_dir: str) -> str:
    """
    Если в архиве один верхнеуровневый каталог — используем его как корень датасета.
    Иначе используем сам каталог распаковки.
    """
    try:
        entries = [name for name in os.listdir(extract_dir) if name != "__meta__.json"]
    except FileNotFoundError:
        return extract_dir
    dirs = [name for name in entries if os.path.isdir(os.path.join(extract_dir, name))]
    files = [name for name in entries if os.path.isfile(os.path.join(extract_dir, name))]
    if len(dirs) == 1 and not files:
        return os.path.join(extract_dir, dirs[0])
    return extract_dir


def extract_dataset_zip_to_cache(workspace_root: str, zip_path: str) -> str:
    """
    Распаковывает zip датасет в кэш workspace/tmp/extracted_datasets с инвалидацией
    по размеру и mtime архива. Возвращает путь к корню распакованного датасета.
    """
    abs_zip = os.path.abspath(os.path.expanduser(zip_path))
    if not os.path.isfile(abs_zip):
        raise FileNotFoundError(f"Zip-архив не найден: {abs_zip}")
    stat = os.stat(abs_zip)
    key_src = f"{abs_zip}|{stat.st_size}|{stat.st_mtime_ns}"
    cache_key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:16]

    layout = WorkspaceLayout(workspace_root)
    cache_root = layout.extracted_datasets
    cache_dir = os.path.join(cache_root, cache_key)
    meta_path = os.path.join(cache_dir, "__meta__.json")
    os.makedirs(cache_root, exist_ok=True)

    if os.path.isdir(cache_dir) and os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if (
                meta.get("zip_path") == abs_zip
                and meta.get("size") == stat.st_size
                and meta.get("mtime_ns") == stat.st_mtime_ns
            ):
                root_rel = meta.get("dataset_root_rel", "")
                root = os.path.join(cache_dir, root_rel) if root_rel else cache_dir
                if os.path.isdir(root):
                    return root
        except Exception:
            pass

    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    _safe_extract_zip(abs_zip, cache_dir)
    dataset_root = _choose_extracted_dataset_root(cache_dir)
    rel_root = os.path.relpath(dataset_root, cache_dir)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "zip_path": abs_zip,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "dataset_root_rel": "" if rel_root == "." else rel_root,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return dataset_root


def resolve_or_extract_dataset_root(
    workspace_root: str,
    entry_key: str,
    entry_dict: dict,
    catalog_dir: str,
) -> str:
    """
    Как resolve_dataset_root, но если путь указывает на zip-архив, возвращает
    корень распакованного датасета из workspace-кэша.
    """
    root = resolve_dataset_root(workspace_root, entry_key, entry_dict, catalog_dir)
    if root.lower().endswith(".zip"):
        return extract_dataset_zip_to_cache(workspace_root, root)
    return root


def workspace_queue_path(workspace_root: str) -> str:
    """Файл очереди обучения в корне workspace (`queue.txt`)."""
    root = os.path.abspath(os.path.expanduser(workspace_root))
    return os.path.join(root, WORKSPACE_QUEUE_BASENAME)


def workspace_queue_status_path(workspace_root: str) -> str:
    """Статусы исполнителя очереди: `workspace/tmp/status.txt`."""
    root = os.path.abspath(os.path.expanduser(workspace_root))
    return os.path.join(root, "tmp", "status.txt")


def deploy_workspace(target_root: str | None = None) -> dict[str, Any]:
    """
    Создаёт каталоги workspace и пустые datasets_info.json при отсутствии.
    target_root по умолчанию — текущий каталог (как у пользовательского workspace).
    """
    root = os.path.abspath(os.path.expanduser(target_root or os.getcwd()))
    layout = WorkspaceLayout(root)
    created_dirs: list[str] = []
    created_files: list[str] = []
    skipped: list[str] = []

    dir_specs = [
        ("source_datasets", layout.source_datasets),
        ("work_datasets", layout.work_datasets),
        ("runs", layout.runs),
        ("analytics", layout.analytics),
        ("models", layout.models),
        ("tmp", os.path.join(root, "tmp")),
        ("extracted_datasets", layout.extracted_datasets),
    ]
    for name, dpath in dir_specs:
        if os.path.isdir(dpath):
            skipped.append(f"dir:{name}")
        else:
            os.makedirs(dpath, exist_ok=True)
            created_dirs.append(name)

    file_specs = [
        ("source_datasets_info", layout.source_datasets_info_path()),
        ("work_datasets_info", layout.work_datasets_info_path()),
    ]
    for label, fpath in file_specs:
        if os.path.isfile(fpath):
            skipped.append(f"file:{label}")
        else:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
            created_files.append(label)

    return {
        "root": root,
        "created_dirs": created_dirs,
        "created_files": created_files,
        "skipped": skipped,
    }
