"""
Единый корень workspace: подкаталоги и резолв путей к датасетам (data_path / каталог по ключу).
"""
import os

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


def workspace_queue_path(workspace_root: str) -> str:
    """Файл очереди обучения в корне workspace (`queue.txt`)."""
    root = os.path.abspath(os.path.expanduser(workspace_root))
    return os.path.join(root, WORKSPACE_QUEUE_BASENAME)


def workspace_queue_status_path(workspace_root: str) -> str:
    """Статусы исполнителя очереди: `workspace/tmp/status.txt`."""
    root = os.path.abspath(os.path.expanduser(workspace_root))
    return os.path.join(root, "tmp", "status.txt")
