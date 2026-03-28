import os
import json
import shutil
import random
import argparse
from tqdm import tqdm

from datasets_json_former import yolo_flat_image_label_buckets


BASE_DIR = "/media/user/Data/IndustrialSafety/Datasets"
JSON_FILE = "datasets_info.json"
CLASS_NAMES_FILE = "class_names.json"
OUTPUT_DATASET_NAME = "merged_dataset"
OUTPUT_DIR = os.path.join("/media/user/Data/IndustrialSafety/Datasets", OUTPUT_DATASET_NAME)
TRAIN_PART = 0.8  # 80%
VAL_PART = 0.1    # 10%
TEST_PART = 0.1   # 10%
RANDOM_SEED = random.seed(12345)


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def _unique_merge_stem(dataset_name, src_image_path, used_stems):
    """
    Имя без расширения для пары image/label: «имя_датасета-исходный_файл».
    used_stems — множество уже занятых имён в целевом split (коллизии: суффиксы __2, __3, …).
    """
    base = os.path.splitext(os.path.basename(src_image_path))[0]
    safe_ds = dataset_name.replace(os.sep, "_").replace("/", "_")
    safe_base = base.replace(os.sep, "_").replace("/", "_")
    stem = f"{safe_ds}-{safe_base}"
    if stem not in used_stems:
        used_stems.add(stem)
        return stem
    n = 2
    while True:
        cand = f"{stem}__{n}"
        if cand not in used_stems:
            used_stems.add(cand)
            return cand
        n += 1


def find_dataset_paths(dataset_path, structure, arg=False):
    paths = []
    dataset_splitting = ["train", "val"] if arg else ["train", "val", "test"]
    if structure == "split":
        for subset in dataset_splitting:
            subdir = os.path.join(dataset_path, subset)
            if os.path.exists(os.path.join(subdir, "images")) and os.path.exists(os.path.join(subdir, "labels")):
                paths.append((os.path.join(subdir, "images"), os.path.join(subdir, "labels")))
    elif structure in ("flat", "subset_flat"):
        buckets = yolo_flat_image_label_buckets(dataset_path)
        if buckets:
            paths.extend(buckets)
        elif os.path.exists(os.path.join(dataset_path, "images")) and os.path.exists(
            os.path.join(dataset_path, "labels")
        ):
            paths.append((os.path.join(dataset_path, "images"), os.path.join(dataset_path, "labels")))
    elif structure == "nested_split":
        for subset in dataset_splitting:
            img_dir = os.path.join(dataset_path, "images", subset)
            lbl_dir = os.path.join(dataset_path, "labels", subset)
            if os.path.exists(img_dir) and os.path.exists(lbl_dir):
                paths.append((img_dir, lbl_dir))
    elif structure == "darknet":
        obj_train_data_path = os.path.join(dataset_path, "obj_train_data")
        if os.path.exists(obj_train_data_path):
            # Для Darknet формата изображения и аннотации в одной папке
            paths.append((obj_train_data_path, obj_train_data_path))
    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Объединение и фильтрация датасетов по выбранным классам"
    )

    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help="Путь к исходным датасетам (если не указан, используется значение BASE_DIR)"
    )

    parser.add_argument(
        "--target-path",
        type=str,
        default=None,
        help="Путь к новому создаваемому датасету (если не указан, используется значение OUTPUT_DIR)"
    )

    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Имена классов через запятую; если не задано — объединение всех классов из всех датасетов в datasets_info.json (кроме выходного)"
    )

    parser.add_argument(
        "--datasets-info-path",
        type=str,
        default=None,
        help="Путь к JSON файлам с информацией о датасетах (если не указан, используется source-path)"
    )
    
    parser.add_argument(
        "--exclude-test",
        action="store_true",
        help="Исключить тестовые данные из выбранных датасетов"
    )

    parser.add_argument(
        "--merge-classes",
        nargs=2,
        metavar=("SOURCES", "TARGET"),
        action="append",
        default=None,
        help='Слияние классов: строка имён через запятую и целевое имя в --classes. Повторяйте флаг для нескольких групп.',
    )

    parser.add_argument(
        "--common-classes-only",
        action="store_true",
        help="Оставить только классы из набора (--classes или авто-объединение), присутствующие в каждом датасете группы пересечения; остальные отбрасываются с предупреждением",
    )

    return parser.parse_args()


def _normalize_name(name, class_names_map):
    return class_names_map.get(name, name)


def dataset_normalized_keys(info, class_names_map):
    return {_normalize_name(k, class_names_map) for k in info["classes"].keys()}


def all_classes_union_from_datasets(datasets_info, output_dataset_name, class_names_map):
    """
    Объединение нормализованных имён классов по всем датасетам из datasets_info,
    кроме выходного. Порядок — лексикографический по нормализованным именам.
    """
    normalized = set()
    for name, info in datasets_info.items():
        if name == output_dataset_name:
            continue
        for k in info.get("classes", {}).keys():
            normalized.add(_normalize_name(k, class_names_map))
    return sorted(normalized)


def request_normalized_tokens(selected_classes, merge_args, class_names_map):
    """Нормализованные имена из --classes и из всех --merge-classes."""
    tokens = {_normalize_name(c, class_names_map) for c in selected_classes}
    if merge_args:
        for sources_csv, target in merge_args:
            tokens.add(_normalize_name(target.strip(), class_names_map))
            for part in sources_csv.split(","):
                p = part.strip()
                if p:
                    tokens.add(_normalize_name(p, class_names_map))
    return tokens


def candidate_datasets_for_common_mode(datasets_info, output_dataset_name, request_tokens, class_names_map):
    """Датасеты (кроме выходного), у которых есть хотя бы один класс из запроса."""
    out = []
    for name, info in datasets_info.items():
        if name == output_dataset_name:
            continue
        if dataset_normalized_keys(info, class_names_map) & request_tokens:
            out.append((name, info))
    return out


def reduce_selected_to_common_in_candidates(
    selected_classes, class_names_map, candidates, merge_targets_to_sources
):
    """
    Оставляет классы из selected_classes по порядку, которые удовлетворяют
    dataset_matches_selection(..., [cls], ...) для каждого датасета из candidates.
    """
    effective = []
    for out in selected_classes:
        if all(
            dataset_matches_selection(info, class_names_map, [out], merge_targets_to_sources)
            for _, info in candidates
        ):
            effective.append(out)
    return effective


def _canonical_class_label(name_n, selected_classes, class_names_map):
    """Имя из --classes`, совпадающее с name_n после нормализации."""
    for sc in selected_classes:
        if _normalize_name(sc, class_names_map) == name_n:
            return sc
    return None


def build_merge_config(merge_args, class_names_map, selected_classes):
    """
    merge_args: список пар [sources_csv, target] или None.
    Возвращает (normalized_to_output_name, merge_targets_to_sources).
    Ключи merge_targets_to_sources — как в списке --classes.
    """
    if not merge_args:
        return None, {}

    merge_targets_to_sources = {}
    used_sources = set()

    for sources_csv, target in merge_args:
        target_n = _normalize_name(target.strip(), class_names_map)
        canonical_target = _canonical_class_label(target_n, selected_classes, class_names_map)
        if canonical_target is None:
            raise ValueError(
                f"Целевой класс слияния {target.strip()!r} не найден в --classes: {selected_classes}"
            )
        sources = [_normalize_name(s.strip(), class_names_map) for s in sources_csv.split(",") if s.strip()]
        if not sources:
            raise ValueError(f"Пустой список исходных классов для цели {target!r}")
        if canonical_target in merge_targets_to_sources:
            raise ValueError(f"Цель слияния {canonical_target!r} указана дважды")
        merge_targets_to_sources[canonical_target] = set(sources)
        for s in sources:
            if s in used_sources:
                raise ValueError(f"Исходный класс {s!r} участвует в более чем одной группе --merge-classes")
            used_sources.add(s)

    normalized_to_output_name = {}
    for canonical_target, src_set in merge_targets_to_sources.items():
        for s in src_set:
            normalized_to_output_name[s] = canonical_target

    for out in selected_classes:
        out_n = _normalize_name(out, class_names_map)
        if out in merge_targets_to_sources:
            continue
        if out_n in used_sources:
            raise ValueError(
                f"Класс {out!r} только как источник слияния; уберите из --classes или задайте отдельную цель"
            )
        if out_n in normalized_to_output_name:
            raise ValueError(f"Конфликт имён для {out!r}")
        normalized_to_output_name[out_n] = out

    return normalized_to_output_name, merge_targets_to_sources


def dataset_matches_selection(
    info, class_names_map, selected_classes, merge_targets_to_sources
):
    normalized_in_ds = {_normalize_name(k, class_names_map) for k in info["classes"].keys()}

    for out in selected_classes:
        out_n = _normalize_name(out, class_names_map)
        sources = merge_targets_to_sources.get(out)
        if sources:
            if not (normalized_in_ds & sources):
                return False
        else:
            if out_n not in normalized_in_ds:
                return False
    return True


def filter_label_file(
    src_label_path,
    dst_label_path,
    class_map,
    class_names_map,
    selected_classes,
    normalized_to_output_name=None,
):
    id_to_normalized = {}
    for name, idx in class_map.items():
        normalized = class_names_map.get(name, name)
        id_to_normalized[idx] = normalized

    new_id_map = {cls: i for i, cls in enumerate(selected_classes)}

    filtered_lines = []

    with open(src_label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue

        normalized_name = id_to_normalized.get(class_id)
        if normalized_name is None:
            continue

        if normalized_to_output_name is not None:
            output_name = normalized_to_output_name.get(normalized_name)
            if output_name is None:
                continue
        else:
            output_name = normalized_name
            if output_name not in selected_classes:
                continue

        if output_name not in new_id_map:
            continue
        new_id = new_id_map[output_name]
        parts[0] = str(new_id)
        filtered_lines.append(" ".join(parts) + "\n")

    if filtered_lines:
        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.writelines(filtered_lines)
        return True
    return False


def main():
    args = parse_args()
    
    # Определяем пути
    source_dir = args.source_path if args.source_path else BASE_DIR
    target_dir = args.target_path if args.target_path else OUTPUT_DIR

    # Определяем путь к JSON файлам
    if args.datasets_info_path:
        info_dir = args.datasets_info_path
    else:
        info_dir = source_dir

    json_file = os.path.join(info_dir, JSON_FILE)
    class_names_file = os.path.join(info_dir, CLASS_NAMES_FILE)

    with open(json_file, "r", encoding="utf-8") as f:
        datasets_info = json.load(f)

    with open(class_names_file, "r", encoding="utf-8") as f:
        class_names_map = json.load(f)

    output_dataset_name = os.path.basename(target_dir)

    # Выбранные классы
    if args.classes:
        selected_classes = [cls.strip() for cls in args.classes.split(",") if cls.strip()]
        if not selected_classes:
            print("[ERROR] Параметр --classes задан, но список имён пуст.")
            return
        classes_auto = False
    else:
        selected_classes = all_classes_union_from_datasets(
            datasets_info, output_dataset_name, class_names_map
        )
        classes_auto = True
        if not selected_classes:
            print(
                "[ERROR] --classes не задан: не найдено ни одного класса в датасетах "
                "(проверьте datasets_info.json и секции classes)."
            )
            return
        print(
            f"[INFO] --classes не задан: используется объединение классов из всех датасетов "
            f"({len(selected_classes)}): {', '.join(selected_classes)}"
        )

    try:
        normalized_to_output_name, merge_targets_to_sources = build_merge_config(
            args.merge_classes, class_names_map, selected_classes
        )
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    requested_classes = list(selected_classes)

    if args.common_classes_only:
        request_tokens = request_normalized_tokens(
            requested_classes, args.merge_classes, class_names_map
        )
        candidates = candidate_datasets_for_common_mode(
            datasets_info, output_dataset_name, request_tokens, class_names_map
        )
        if not candidates:
            req_label = "классами из --classes" if not classes_auto else "автособранным набором классов"
            print(
                f"[ERROR] Ни один датасет не пересекается с {req_label} "
                "(и при необходимости из --merge-classes). Проверьте имена и datasets_info.json."
            )
            return

        effective_classes = reduce_selected_to_common_in_candidates(
            requested_classes, class_names_map, candidates, merge_targets_to_sources
        )
        if not effective_classes:
            req_label = "классов из --classes" if not classes_auto else "автособранных классов"
            print(
                f"[ERROR] Ни один из {req_label} не присутствует одновременно "
                "во всех датасетах группы (есть пересечение с запросом)."
            )
            return

        if effective_classes != requested_classes:
            dropped = [c for c in requested_classes if c not in effective_classes]
            print("[WARN] Режим --common-classes-only: итоговый набор классов сужен.")
            src_label = "Запрошено в --classes" if not classes_auto else "Исходный набор (авто)"
            print(f"[WARN] {src_label}: {', '.join(requested_classes)}")
            print(f"[WARN] Будет использовано: {', '.join(effective_classes)}")
            print(
                f"[WARN] Исключено (нет покрытия хотя бы в одном датасете из группы пересечения): "
                f"{', '.join(dropped)}"
            )
            print(
                f"[INFO] Группа датасетов для проверки пересечения: "
                f"{len(candidates)} шт. ({', '.join(n for n, _ in candidates)})"
            )

        selected_classes = effective_classes
        try:
            normalized_to_output_name, merge_targets_to_sources = build_merge_config(
                args.merge_classes, class_names_map, selected_classes
            )
        except ValueError as e:
            print(
                f"[ERROR] После сокращения классов --merge-classes несогласован с итоговым списком: {e}"
            )
            return

    for split in ["train", "valid", "test"]:
        safe_mkdir(os.path.join(target_dir, split, "images"))
        safe_mkdir(os.path.join(target_dir, split, "labels"))

    matching_datasets = []
    for dataset_name, info in datasets_info.items():
        if dataset_name == output_dataset_name:
            continue
        if dataset_matches_selection(
            info, class_names_map, selected_classes, merge_targets_to_sources
        ):
            matching_datasets.append((dataset_name, info))

    if not matching_datasets:
        print("[ERROR] Ни один датасет не содержит все выбранные классы.")
        return

    print(f"[INFO] Найдено {len(matching_datasets)} подходящих датасета:")
    for name, _ in matching_datasets:
        print(f"   - {name}")

    total_labels = 0
    for dataset_name, info in matching_datasets:
        dataset_path = os.path.join(source_dir, dataset_name)
        for _, labels_path in find_dataset_paths(dataset_path, info["structure"], args.exclude_test):
            total_labels += len([f for f in os.listdir(labels_path) if f.endswith(".txt")])

    used_stems = {split: set() for split in ("train", "valid", "test")}
    copied_count = 0

    with tqdm(total=total_labels, desc="Обработка датасетов", unit="файл") as pbar:
        for dataset_name, info in matching_datasets:
            dataset_path = os.path.join(source_dir, dataset_name)
            for images_path, labels_path in find_dataset_paths(dataset_path, info["structure"], args.exclude_test):

                pairs = []
                for label_file in os.listdir(labels_path):
                    if not label_file.endswith(".txt"):
                        continue
                    image_name = os.path.splitext(label_file)[0]
                    for ext in [".jpg", ".jpeg", ".png"]:
                        candidate = os.path.join(images_path, image_name + ext)
                        if os.path.exists(candidate):
                            pairs.append((candidate, os.path.join(labels_path, label_file)))
                            break

                if not pairs:
                    continue

                random.shuffle(pairs)
                n = len(pairs)
                train_split = pairs[:int(n * TRAIN_PART)]
                val_split = pairs[int(n * TRAIN_PART):int(n * (TRAIN_PART + VAL_PART))]
                test_split = pairs[int(n * (VAL_PART + TRAIN_PART)):]

                splits_data = {"train": train_split, "valid": val_split, "test": test_split}

                for split_name, split_pairs in splits_data.items():
                    for image_src, label_src in split_pairs:
                        image_ext = os.path.splitext(image_src)[1]
                        stem = _unique_merge_stem(
                            dataset_name, image_src, used_stems[split_name]
                        )
                        image_dst = os.path.join(
                            target_dir, split_name, "images", f"{stem}{image_ext}"
                        )
                        label_dst = os.path.join(
                            target_dir, split_name, "labels", f"{stem}.txt"
                        )

                        ok = filter_label_file(
                            label_src,
                            label_dst,
                            info["classes"],
                            class_names_map,
                            selected_classes,
                            normalized_to_output_name,
                        )
                        if ok:
                            shutil.copy2(image_src, image_dst)
                            copied_count += 1
                        pbar.update(1)

    print(f"\n[DEBUG] Всего label-файлов: {total_labels}")
    print(f"[DEBUG] Отфильтровано и скопировано: {copied_count}")
    print(f"[DEBUG] Процент используемых файлов: {copied_count / total_labels * 100:.2f}%")

    print(f"\n[OK] Скопировано {copied_count} изображений с фильтрованными аннотациями.")

    yaml_path = os.path.join(target_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("train: ./train/images\n")
        f.write("val: ./valid/images\n")
        f.write("test: ./test/images\n\n")
        f.write(f"nc: {len(selected_classes)}\n")
        f.write(f"names: {selected_classes}\n")

    print(f"[OK] Итоговый YAML создан: {yaml_path}")


if __name__ == "__main__":
    main()
