import argparse
import hashlib
import json
import os
import sys

from workspace_paths import (
    DATASETS_INFO_FILE,
    WorkspaceLayout,
    resolve_dataset_root,
    resolve_workspace_root,
)


def calculate_dataset_hash(dataset_path):
    """
    Вычисляет хеш датасета на основе структуры папок, имен файлов и их размеров.
    
    Хеш не зависит от даты/времени изменения файлов, но зависит от:
    - Структуры папок в датасете
    - Имен файлов (изображения и разметка)
    - Размеров файлов
    
    Args:
        dataset_path: Путь к папке с датасетом
        
    Returns:
        str: Первые 8 символов MD5 хеша (hex)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Папка с датасетом не найдена: {dataset_path}")
    
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Указанный путь не является папкой: {dataset_path}")
    
    hasher = hashlib.md5()
    
    # Список служебных файлов, которые нужно игнорировать
    ignored_files = {'.DS_Store', 'Thumbs.db', '.gitkeep', '.gitignore'}
    
    # Получаем абсолютный путь для нормализации
    dataset_path = os.path.abspath(dataset_path)
    dataset_path_len = len(dataset_path) + 1  # +1 для слеша после пути
    
    # Собираем информацию о всех файлах и папках в отсортированном порядке
    items = []
    
    for root, dirs, files in os.walk(dataset_path):
        # Сортируем для детерминированности
        dirs.sort()
        files.sort()
        
        # Получаем относительный путь от корня датасета
        rel_root = root[dataset_path_len:] if len(root) > dataset_path_len else ""
        
        # Добавляем информацию о папках
        for dir_name in dirs:
            rel_path = os.path.join(rel_root, dir_name) if rel_root else dir_name
            items.append(('dir', rel_path))
        
        # Добавляем информацию о файлах
        for file_name in files:
            # Пропускаем служебные файлы
            if file_name in ignored_files:
                continue
            
            rel_path = os.path.join(rel_root, file_name) if rel_root else file_name
            file_path = os.path.join(root, file_name)
            
            try:
                file_size = os.path.getsize(file_path)
                items.append(('file', rel_path, file_size))
            except (OSError, IOError):
                # Пропускаем файлы, к которым нет доступа
                continue
    
    # Сортируем все элементы для детерминированности
    items.sort()
    
    # Вычисляем хеш на основе собранной информации
    for item in items:
        if item[0] == 'dir':
            # Для папки: добавляем тип и относительный путь
            hasher.update(b'dir:')
            hasher.update(item[1].encode('utf-8'))
            hasher.update(b'\n')
        elif item[0] == 'file':
            # Для файла: добавляем тип, относительный путь и размер
            hasher.update(b'file:')
            hasher.update(item[1].encode('utf-8'))
            hasher.update(b':')
            hasher.update(str(item[2]).encode('utf-8'))
            hasher.update(b'\n')
    
    # Возвращаем первые 8 символов хеша
    return hasher.hexdigest()[:8]


def resolve_hash_dataset_root(
    workspace_cli: str | None,
    dataset_path_pos: str | None,
    work_dataset: str | None,
) -> str:
    """
    Корень датасета для хеша: явный путь или резолв по имени записи work_datasets
    (как resolve_training_data_path в model_training_module, без импорта Ultralytics).
    """
    if work_dataset is not None and str(work_dataset).strip():
        name = str(work_dataset).strip()
        root = resolve_workspace_root(workspace_cli)
        layout = WorkspaceLayout(root)
        expanded = os.path.abspath(os.path.expanduser(name))
        yaml_here = os.path.join(expanded, "data.yaml")
        if os.path.isdir(expanded) and os.path.isfile(yaml_here):
            return expanded
        info_path = layout.work_datasets_info_path()
        if not os.path.isfile(info_path):
            raise FileNotFoundError(
                f"Каталог с data.yaml для {name!r} не найден и отсутствует {info_path}."
            )
        with open(info_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        if not isinstance(catalog, dict):
            raise ValueError(f"{info_path}: ожидается объект JSON.")
        if name not in catalog:
            raise KeyError(
                f"Имя {name!r} отсутствует в work_datasets/{DATASETS_INFO_FILE}."
            )
        entry = catalog[name]
        if not isinstance(entry, dict):
            raise TypeError(f"Запись {name!r} должна быть объектом JSON.")
        return resolve_dataset_root(layout.root, name, entry, layout.work_datasets)

    if dataset_path_pos is None or not str(dataset_path_pos).strip():
        raise ValueError(
            "Укажите путь к папке датасета или --work-dataset <имя> с --workspace "
            "(или SMART_TRAIN_WORKSPACE)."
        )
    return os.path.abspath(os.path.expanduser(str(dataset_path_pos).strip()))


def main():
    parser = argparse.ArgumentParser(
        description="Вычисление хеша датасета на основе структуры, имен файлов и их размеров"
    )
    
    parser.add_argument(
        "dataset_path",
        type=str,
        nargs="?",
        default=None,
        help="Путь к папке с датасетом (не нужен при --work-dataset)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Корень workspace для --work-dataset (иначе SMART_TRAIN_WORKSPACE)",
    )
    parser.add_argument(
        "--work-dataset",
        type=str,
        default=None,
        metavar="NAME",
        help="Имя записи из work_datasets/datasets_info.json (или каталог с data.yaml)",
    )

    parser.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Ожидаемое значение хеша для валидации"
    )
    
    args = parser.parse_args()
    if args.work_dataset and args.dataset_path:
        print(
            "[ERROR] Укажите либо путь к датасету, либо --work-dataset, не оба сразу.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        root = resolve_hash_dataset_root(
            args.workspace, args.dataset_path, args.work_dataset
        )
        computed_hash = calculate_dataset_hash(root)
        
        if args.validate:
            if computed_hash.lower() == args.validate.lower():
                print(f"Валидация успешна. Хеш совпадает: {computed_hash}")
                sys.exit(0)
            else:
                print(f"Валидация не пройдена.")
                print(f"Ожидалось: {args.validate}")
                print(f"Получено: {computed_hash}")
                sys.exit(1)
        else:
            print(computed_hash)
            sys.exit(0)

    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except KeyError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[ERROR] Неожиданная ошибка: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

