import os
import hashlib
import argparse
import sys


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


def main():
    parser = argparse.ArgumentParser(
        description="Вычисление хеша датасета на основе структуры, имен файлов и их размеров"
    )
    
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Путь к папке с датасетом"
    )
    
    parser.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Ожидаемое значение хеша для валидации"
    )
    
    args = parser.parse_args()
    
    try:
        computed_hash = calculate_dataset_hash(args.dataset_path)
        
        if args.validate:
            # Режим валидации
            if computed_hash.lower() == args.validate.lower():
                print(f"Валидация успешна. Хеш совпадает: {computed_hash}")
                sys.exit(1)  # Успех
            else:
                print(f"Валидация не пройдена.")
                print(f"Ожидалось: {args.validate}")
                print(f"Получено: {computed_hash}")
                sys.exit(0)  # Ошибка
        else:
            # Режим вычисления хеша
            print(computed_hash)
            sys.exit(0)
            
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(0)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Неожиданная ошибка: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()

