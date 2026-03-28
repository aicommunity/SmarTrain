import sys
import os
import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from dataset_hash import calculate_dataset_hash


DATASET_PATH = "/media/user/Data/IndustrialSafety/Datasets/HardHatSkz"
MODELS_BASE_DIR = "/media/user/Data/IndustrialSafety/Models"
MODEL_VERSION = "yolov8n"
EPOCHS = 50
BATCH = 16
IMG_SIZE = 640


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение моделей")

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Путь к папке с датасетом (должен содержать файл data.yaml)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Модель (например: yolov8n.pt, yolov8s.pt, yolov11n.pt)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Количество эпох обучения"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Размер batch"
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Размер изображения"
    )

    parser.add_argument(
        "--target-path",
        type=str,
        default=None,
        help="Путь к папке с результатами обучения"
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Путь к папке с моделью"
    )

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Выполнить только тестирование без обучения"
    )

    return parser.parse_args()


def train_yolo(dataset_path, model_version, epochs, batch, img_size, target_dir):
    training_start_time = datetime.now()
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Папка с датасетом не найдена: {dataset_path}")
    
    data_yaml = os.path.join(dataset_path, "data.yaml")

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Не найден yaml файл: {data_yaml}")

    dataset_name = os.path.basename(os.path.normpath(dataset_path))

    # Вычисляем хеш датасета
    try:
        dataset_hash = calculate_dataset_hash(dataset_path)
        print(f"[INFO] Хеш датасета: {dataset_hash}")
    except Exception as e:
        print(f"[WARNING] Не удалось вычислить хеш датасета: {e}")
        dataset_hash = None

    # Форматируем дату и время в формате YYYY-MM-DD_HH-MM
    timestamp_str = training_start_time.strftime("%Y-%m-%d_%H-%M")
    
    # Формируем имя папки с хешем
    folder_name = f"{timestamp_str}_{model_version.replace('.pt', '')}_{epochs}epochs"
    if dataset_hash:
        folder_name = f"{folder_name}-{dataset_hash}"
    
    model_dir = os.path.join(
        target_dir, 
        dataset_name, 
        folder_name
        )
    
    if os.path.exists(model_dir):
        while True:
            answer = input(f"[WARNING] Папка с таким названием уже существует: {model_dir}. Продолжить обучение? (y/n): \n").strip().lower()
            if answer == 'y':
                break
            elif answer == 'n':
                sys.exit(1)
            else:
                print("Пожалуйста, введите только 'y' или 'n'.\n")
    else:
        os.makedirs(model_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"[INFO] Обучение модели: {model_version}")
    print(f"[INFO] Датасет: {dataset_name}")
    print(f"[INFO] Конфигурация: {data_yaml}")
    print(f"[INFO] Сохранение результатов в {model_dir}")
    print("=" * 60 + "\n")

    _, model_ext = os.path.splitext(model_version)
    if model_ext == '':
        model = YOLO(model_version + ".pt")
    else:
        model = YOLO(model_version)

    training_end_time = None
    try:
        model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=img_size,
            project=model_dir,
            name="train",
            exist_ok=False
        )

        training_end_time = datetime.now()
        model_path = os.path.join(model_dir, "train", "weights", "best.pt")

        print("\n" + "-" * 60)
        if os.path.exists(model_path):
            print(f"[OK] Обучение завершено.")
            print(f"[INFO] Модель сохранена по пути:\n{model_path}")
    except Exception as e:
        training_end_time = datetime.now()
        print(f"[ERROR] Не удалось запустить обучение {model_version} на датасете {dataset_name} на {epochs} эпох: {e}")
    
    return model_dir, training_start_time, training_end_time, dataset_hash


def test_yolo(model_dir, dataset_path, training_start_time=None, training_end_time=None):
    test_start_time = datetime.now()
    
    model_path = os.path.join(model_dir, "train", "weights", "best.pt")
    trained_model = YOLO(model_path)

    data_yaml = os.path.join(dataset_path, "data.yaml")

    print("\n" + "=" * 60)
    print(f"[INFO] Тестирование модели: {model_dir}")
    print(f"[INFO] Датасет: {dataset_path}")
    print(f"[INFO] Конфигурация: {data_yaml}")
    print(f"[INFO] Сохранение результатов в {model_dir}")
    print("=" * 60 + "\n")
    
    test_end_time = None
    try:
        result = trained_model.val(
            data=data_yaml, 
            split='test', 
            project=model_dir, 
            name="test",
            exist_ok = False
            )

        test_end_time = datetime.now()
        csv_file = save_metrics_csv(result, model_dir)

        print("\n" + "-" * 60)
        if os.path.exists(csv_file):
            print(f"[OK] Тестирование завершено.")
            print(f"[INFO] Результаты сохранены по пути:\n{csv_file}")
        else:
            print("[ERROR] .csv файл не найден. Проверьте лог Ultralytics.")
        print("-" * 60 + "\n")
    except Exception as e:
        test_end_time = datetime.now()
        print(f"[ERROR] Не удалось протестировать {model_dir} на датасете {dataset_path}: {e}")
    
    return test_start_time, test_end_time


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


def save_training_metadata(model_dir, dataset_path, model_version=None, training_start_time=None, 
                          training_end_time=None, test_start_time=None, test_end_time=None,
                          epochs=None, batch=None, img_size=None, training_success=True, 
                          training_error=None, test_success=True, test_error=None, dataset_hash=None):
    """
    Сохраняет метаданные обучения в JSON файл рядом с test_metrics.csv
    
    Args:
        model_dir: Директория с результатами обучения
        dataset_path: Полный путь к датасету
        model_version: Версия модели (например, 'yolo11n')
        training_start_time: Время начала обучения
        training_end_time: Время окончания обучения
        test_start_time: Время начала тестирования
        test_end_time: Время окончания тестирования
        epochs: Количество эпох
        batch: Размер batch
        img_size: Размер изображения
        training_success: Успешность обучения (True/False)
        training_error: Сообщение об ошибке обучения (если было)
        test_success: Успешность тестирования (True/False)
        test_error: Сообщение об ошибке тестирования (если было)
        dataset_hash: Хеш датасета (8 символов)
    """
    metadata = {
        "training_info": {
            "framework": "ultralytics",
            "task_type": "detection",
            "model": model_version,
            "dataset": {
                "name": os.path.basename(os.path.normpath(dataset_path)),
                "path_absolute": os.path.abspath(dataset_path),
                "path_relative": _get_relative_path(dataset_path, model_dir),
                "hash": dataset_hash
            },
            "hyperparameters": {
                "epochs": epochs,
                "batch_size": batch,
                "image_size": img_size
            }
        },
        "timestamps": {
            "training": {
                "start": training_start_time.isoformat() if training_start_time else None,
                "end": training_end_time.isoformat() if training_end_time else None,
                "duration_seconds": (training_end_time - training_start_time).total_seconds() 
                    if training_start_time and training_end_time else None
            },
            "testing": {
                "start": test_start_time.isoformat() if test_start_time else None,
                "end": test_end_time.isoformat() if test_end_time else None,
                "duration_seconds": (test_end_time - test_start_time).total_seconds() 
                    if test_start_time and test_end_time else None
            }
        },
        "status": {
            "training": {
                "success": training_success,
                "error": training_error
            },
            "testing": {
                "success": test_success,
                "error": test_error
            }
        },
        "paths": {
            "model_directory": ".",
            "best_model": "train/weights/best.pt" if os.path.exists(
                os.path.join(model_dir, "train", "weights", "best.pt")) else None
        }
    }
    
    metadata_file = os.path.join(model_dir, "training_metadata.json")
    
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Метаданные обучения сохранены: {metadata_file}")
    except Exception as e:
        print(f"[WARNING] Не удалось сохранить метаданные: {e}")


def _get_relative_path(target_path, base_path):
    """
    Вычисляет относительный путь от base_path к target_path.
    Если пути на разных дисках или не удается вычислить относительный путь,
    возвращает абсолютный путь.
    """
    try:
        target = Path(os.path.abspath(target_path))
        base = Path(os.path.abspath(base_path))
        
        # Пытаемся вычислить относительный путь
        try:
            relative = os.path.relpath(target, base)
            # Если относительный путь содержит много '..', все равно возвращаем его
            # (это нормально для датасетов, которые находятся далеко от модели)
            return relative
        except ValueError:
            # Пути на разных дисках или другие проблемы - возвращаем абсолютный
            return target.as_posix()
    except Exception:
        # В случае любой ошибки возвращаем абсолютный путь
        return os.path.abspath(target_path)
    

def main():
    args = parse_args()
    
    data = args.data if args.data else DATASET_PATH
    model_version = args.model if args.model else MODEL_VERSION
    epochs = args.epochs if args.epochs else EPOCHS
    batch = args.batch if args.batch else BATCH
    img_size = args.img_size if args.img_size else IMG_SIZE
    target_dir = args.target_path if args.target_path else MODELS_BASE_DIR

    # Переменные для отслеживания статуса
    training_success = True
    training_error = None
    test_success = True
    test_error = None
    training_start_time = None
    training_end_time = None
    test_start_time = None
    test_end_time = None
    model_dir = None

    # Переменная для хранения хеша датасета
    dataset_hash = None
    
    if not args.test_only:
        # Обучение с обработкой ошибок
        try:
            model_dir, training_start_time, training_end_time, dataset_hash = train_yolo(
                dataset_path=data,
                model_version=model_version,
                epochs=epochs,
                batch=batch,
                img_size=img_size,
                target_dir=target_dir
            )
        except Exception as e:
            training_success = False
            training_error = str(e)
            training_end_time = datetime.now()
            print(f"[ERROR] Ошибка при обучении: {e}")
            training_error = f"{str(e)}\n{traceback.format_exc()}"
            # Вычисляем хеш для случая ошибки
            try:
                dataset_hash = calculate_dataset_hash(data)
            except Exception:
                dataset_hash = None
            # Создаем директорию для сохранения метаданных об ошибке
            if not model_dir:
                dataset_name = os.path.basename(os.path.normpath(data))
                timestamp_str = training_start_time.strftime("%Y-%m-%d_%H-%M") if training_start_time else datetime.now().strftime("%Y-%m-%d_%H-%M")
                folder_name = f"{timestamp_str}_{model_version.replace('.pt', '')}_{epochs}epochs"
                if dataset_hash:
                    folder_name = f"{folder_name}-{dataset_hash}"
                model_dir = os.path.join(
                    target_dir,
                    dataset_name,
                    folder_name
                )
                os.makedirs(model_dir, exist_ok=True)

        # Тестирование с обработкой ошибок (только если обучение прошло успешно)
        if training_success and model_dir:
            try:
                test_start_time, test_end_time = test_yolo(
                    model_dir=model_dir,
                    dataset_path=data,
                    training_start_time=training_start_time,
                    training_end_time=training_end_time
                )
            except Exception as e:
                test_success = False
                test_error = str(e)
                test_end_time = datetime.now()
                print(f"[ERROR] Ошибка при тестировании: {e}")
                test_error = f"{str(e)}\n{traceback.format_exc()}"
        
        # Сохраняем метаданные с параметрами обучения и тестирования
        if model_dir:
            save_training_metadata(
                model_dir=model_dir,
                dataset_path=data,
                model_version=model_version.replace('.pt', ''),
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
                dataset_hash=dataset_hash
            )
    else:
        model_dir = args.model_dir
        if model_dir:
            try:
                test_start_time, test_end_time = test_yolo(model_dir, data)
            except Exception as e:
                test_success = False
                test_error = str(e)
                test_end_time = datetime.now()
                print(f"[ERROR] Ошибка при тестировании: {e}")
                test_error = f"{str(e)}\n{traceback.format_exc()}"
            
            # Для режима test-only сохраняем только метаданные тестирования
            save_training_metadata(
                model_dir=model_dir,
                dataset_path=data,
                test_start_time=test_start_time,
                test_end_time=test_end_time,
                test_success=test_success,
                test_error=test_error
            )
        else:
            print(f"[ERROR] Не указан путь к модели")

    
if __name__ == "__main__":
    main()
