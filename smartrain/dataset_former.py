import os
import json
import shutil
import random
import argparse
import tempfile
import sys
from datetime import datetime
from tqdm import tqdm

from smartrain.cli_argparse import CliArgumentParser
from smartrain.cvat11_converter import YOLO_IMAGE_EXTS
from smartrain.dataset_access import (
    find_dataset_paths,
    iter_image_label_buckets,
    resolve_dataset_root_for_entry,
)
from smartrain.dataset_passport import write_dataset_passport
from smartrain.workspace_paths import (
    WORKSPACE_ENV_VAR,
    WorkspaceLayout,
    resolve_workspace_root,
    DATASETS_INFO_FILE,
    CLASS_NAMES_FILE,
)

# Суффикс имени каталога по умолчанию в workspace (префикс — дата-время см. main).
FUSION_DEFAULT_DIR_SUFFIX = "merged"
TRAIN_PART = 0.8  # 80%
VAL_PART = 0.1    # 10%
TEST_PART = 0.1   # 10%
RANDOM_SEED = random.seed(12345)

_SPLIT_SUM_EPS = 1e-5


def parse_fusion_split_arg(value: str | None) -> tuple[float, float, float]:
    """
    Три доли train, val, test для переразбиения кадров внутри каждого bucket при fusion.
    Сумма должна быть 1.0 (с допуском). При value is None — константы модуля.
    """
    if value is None or not str(value).strip():
        return TRAIN_PART, VAL_PART, TEST_PART
    raw = [x.strip() for x in str(value).split(",")]
    if len(raw) != 3:
        raise ValueError(
            "Ожидается ровно три числа через запятую: train,val,test (например 0.8,0.1,0.1)."
        )
    try:
        tr, va, te = (float(x) for x in raw)
    except ValueError as e:
        raise ValueError(f"Некорректные числа в --fusion-split: {value!r}") from e
    if tr < 0 or va < 0 or te < 0:
        raise ValueError("Доли в --fusion-split не могут быть отрицательными.")
    s = tr + va + te
    if abs(s - 1.0) > _SPLIT_SUM_EPS:
        raise ValueError(
            f"Сумма долей --fusion-split должна быть 1.0 (сейчас {s:.6f}): {value!r}"
        )
    return tr, va, te


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


def build_dataset_former_arg_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        description="Объединение и фильтрация датасетов по выбранным классам"
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Корень workspace (иначе {WORKSPACE_ENV_VAR}); работа только с datasets/",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Имя выходного датасета (подкаталог datasets/) в workspace; "
        "если не задано — YYYY-MM-DD_HH-MM-SS-merged",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Имя входного датасета для объединения (можно повторять).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="CSV-список входных датасетов для объединения (например ds1,ds2).",
    )

    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help="Legacy: родительский каталог датасетов (вместе с --target-path и --datasets-info-path)",
    )

    parser.add_argument(
        "--target-path",
        type=str,
        default=None,
        help="Legacy: полный путь к выходному датасету; в workspace — переопределяет datasets/<output-name>",
    )

    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Имена классов через запятую; если не задано — объединение всех классов из всех датасетов в datasets_info.json (кроме выходного)",
    )

    parser.add_argument(
        "--datasets-info-path",
        type=str,
        default=None,
        help="Legacy: каталог с datasets_info.json; в workspace не нужен (всегда datasets/)",
    )

    parser.add_argument(
        "--exclude-test",
        action="store_true",
        help="Исключить тестовые данные из выбранных датасетов",
    )

    parser.add_argument(
        "--merge-classes",
        nargs=2,
        metavar=("SOURCES", "TARGET"),
        action="append",
        default=None,
        help="Слияние классов: строка имён через запятую и целевое имя в --classes. Повторяйте флаг для нескольких групп.",
    )

    parser.add_argument(
        "--common-classes-only",
        action="store_true",
        help="Оставить только классы из набора (--classes или авто-объединение), присутствующие в каждом датасете группы пересечения; остальные отбрасываются с предупреждением",
    )
    parser.add_argument(
        "--include-partial-datasets",
        action="store_true",
        help="Брать в слияние датасеты, в которых есть хотя бы один из выбранных классов "
        "(по умолчанию требуется наличие всех выбранных классов в каждом датасете)",
    )
    parser.add_argument(
        "--drop-empty-images",
        action="store_true",
        help="После слияния удалить в выходном каталоге пары image+label без ни одной валидной строки YOLO в .txt",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default=None,
        help="Каталог для временных файлов (по умолчанию: <workspace>/tmp или <source-path>/tmp в legacy-режиме)",
    )

    parser.add_argument(
        "--fusion-split",
        type=str,
        default=None,
        help="Только fusion: доли train,val,test при случайном переразбиении кадров внутри каждого "
        "bucket исходного датасета (три числа через запятую, сумма 1.0). По умолчанию "
        f"{TRAIN_PART},{VAL_PART},{TEST_PART}. Не влияет на scan, train, roi и др.",
    )

    return parser


def parse_args(argv=None):
    return build_dataset_former_arg_parser().parse_args(argv)


def _parse_selected_datasets(args) -> list[str]:
    out: list[str] = []
    if args.dataset:
        for item in args.dataset:
            name = str(item).strip()
            if name:
                out.append(name)
    if args.datasets:
        for part in str(args.datasets).split(","):
            name = part.strip()
            if name:
                out.append(name)
    uniq: list[str] = []
    seen: set[str] = set()
    for name in out:
        if name not in seen:
            seen.add(name)
            uniq.append(name)
    return uniq


def _prompt_dataset_selection(available: list[str]) -> list[str]:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import WordCompleter

    print("[INFO] Не указаны --dataset/--datasets: интерактивный выбор входных датасетов.")
    print("[INFO] Доступные датасеты:")
    for name in available:
        print(f"  - {name}")
    completer = WordCompleter(available, ignore_case=True)
    value = prompt(
        "Введите датасеты через запятую: ",
        completer=completer,
        complete_while_typing=True,
    )
    parsed = [x.strip() for x in str(value).split(",") if x.strip()]
    uniq: list[str] = []
    seen: set[str] = set()
    for name in parsed:
        if name not in seen:
            seen.add(name)
            uniq.append(name)
    return uniq


def _prompt_yes_no(label: str, default: bool = False) -> bool:
    from prompt_toolkit import prompt

    suffix = "Y/n" if default else "y/N"
    default_text = "y" if default else "n"
    raw = prompt(f"{label} [{suffix}]: ", default=default_text)
    val = str(raw).strip().lower()
    if not val:
        return default
    return val in ("y", "yes", "1", "true", "да", "д")


def _prompt_interactive_options(
    args,
    *,
    default_output_name: str,
    class_candidates: list[str],
) -> None:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import WordCompleter

    print("[INFO] Интерактивная настройка параметров fusion (Enter = значение по умолчанию).")
    if class_candidates:
        print(
            "[INFO] Доступные классы выбранных датасетов: "
            + ", ".join(class_candidates)
        )
    else:
        print("[WARN] В выбранных датасетах не найдено классов в метаданных.")
    out_name = prompt("Имя выходного датасета: ", default=default_output_name).strip()
    args.output_name = out_name or default_output_name

    class_completer = WordCompleter(class_candidates, ignore_case=True)
    classes_raw = prompt(
        "Классы через запятую (пусто = авто-объединение): ",
        default=(args.classes or ""),
        completer=class_completer,
        complete_while_typing=True,
    ).strip()
    args.classes = classes_raw or None

    split_default = args.fusion_split or f"{TRAIN_PART},{VAL_PART},{TEST_PART}"
    args.fusion_split = prompt(
        "Fusion split train,val,test (сумма=1.0): ",
        default=split_default,
    ).strip()

    args.include_partial_datasets = _prompt_yes_no(
        "Включать частичные датасеты (--include-partial-datasets)",
        default=bool(args.include_partial_datasets),
    )
    args.common_classes_only = _prompt_yes_no(
        "Оставить только общие классы (--common-classes-only)",
        default=bool(args.common_classes_only),
    )
    args.exclude_test = _prompt_yes_no(
        "Исключить test части источников (--exclude-test)",
        default=bool(args.exclude_test),
    )
    args.drop_empty_images = _prompt_yes_no(
        "Удалять пары без валидных объектов (--drop-empty-images)",
        default=bool(args.drop_empty_images),
    )

    tmp_default = args.tmp_dir or ""
    tmp_value = prompt("Каталог tmp (пусто = по умолчанию): ", default=tmp_default).strip()
    args.tmp_dir = tmp_value or None


def _validate_requested_classes(
    selected_classes: list[str],
    class_candidates: list[str],
    class_names_map: dict,
) -> tuple[bool, list[str]]:
    """
    Проверка, что пользовательские классы доступны среди выбранных датасетов
    с учетом нормализации class_names.
    """
    if not selected_classes:
        return True, []
    available_norm = {_normalize_name(c, class_names_map) for c in class_candidates}
    missing = [
        cls
        for cls in selected_classes
        if _normalize_name(cls, class_names_map) not in available_norm
    ]
    return (len(missing) == 0), missing


def _normalize_name(name, class_names_map):
    return class_names_map.get(name, name)


def dataset_normalized_keys(info, class_names_map):
    if "classes" not in info:
        return set()
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
        if "classes" not in info:
            continue
        for k in info["classes"].keys():
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
    info,
    class_names_map,
    selected_classes,
    merge_targets_to_sources,
    *,
    require_all_classes: bool = True,
):
    """
    require_all_classes=True (по умолчанию): в датасете должны быть все выбранные классы
    (для группы --merge-classes — хотя бы один источник из каждой группы).
    require_all_classes=False: достаточно пересечения с любым из выбранных классов.
    """
    if "classes" not in info:
        return False
    normalized_in_ds = {_normalize_name(k, class_names_map) for k in info["classes"].keys()}

    if require_all_classes:
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

    for out in selected_classes:
        out_n = _normalize_name(out, class_names_map)
        sources = merge_targets_to_sources.get(out)
        if sources:
            if normalized_in_ds & sources:
                return True
        else:
            if out_n in normalized_in_ds:
                return True
    return False


def _label_file_has_valid_yolo_annotation(path: str) -> bool:
    """Есть ли в файле хотя бы одна строка с целочисленным class_id (как в filter_label_file)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    int(parts[0])
                except ValueError:
                    continue
                return True
    except OSError:
        return False
    return False


def prune_output_empty_label_pairs(target_dir: str) -> int:
    """
    Удаляет в train/valid/test пары label+image, где .txt пустой или без валидных аннотаций.
    Возвращает число удалённых label-файлов.
    """
    removed = 0
    for split in ("train", "valid", "test"):
        labels_dir = os.path.join(target_dir, split, "labels")
        images_dir = os.path.join(target_dir, split, "images")
        if not os.path.isdir(labels_dir):
            continue
        for name in os.listdir(labels_dir):
            if not name.endswith(".txt"):
                continue
            lp = os.path.join(labels_dir, name)
            if _label_file_has_valid_yolo_annotation(lp):
                continue
            stem = os.path.splitext(name)[0]
            try:
                os.remove(lp)
            except OSError:
                pass
            if os.path.isdir(images_dir):
                for fn in os.listdir(images_dir):
                    if os.path.splitext(fn)[0] == stem:
                        ip = os.path.join(images_dir, fn)
                        try:
                            os.remove(ip)
                        except OSError:
                            pass
            removed += 1
    return removed


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


def _update_datasets_sidecar(
    layout: WorkspaceLayout,
    output_key: str,
    selected_classes: list,
    target_dir: str,
) -> None:
    os.makedirs(layout.datasets, exist_ok=True)
    rel = os.path.relpath(os.path.abspath(target_dir), layout.root)
    entry = {
        "classes": {name: idx for idx, name in enumerate(selected_classes)},
        "structure": "split",
        "elements_count": None,
        "data_path": rel,
    }
    info_path = layout.work_datasets_info_path()
    previous: dict = {}
    if os.path.isfile(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            previous = json.load(f)
        if not isinstance(previous, dict):
            previous = {}
    previous[output_key] = entry
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(previous, f, ensure_ascii=False, indent=4)

    cn_path = layout.work_class_names_path()
    class_names_out: dict = {}
    if os.path.isfile(cn_path):
        with open(cn_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            class_names_out = dict(loaded)
    for c in selected_classes:
        class_names_out[c] = c
    with open(cn_path, "w", encoding="utf-8") as f:
        json.dump(class_names_out, f, ensure_ascii=False, indent=4)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    legacy = (
        args.source_path is not None
        and args.target_path is not None
        and args.datasets_info_path is not None
    )
    layout: WorkspaceLayout | None = None
    workspace_root: str | None = None

    if legacy:
        source_dir = os.path.abspath(os.path.expanduser(args.source_path))
        target_dir = os.path.abspath(os.path.expanduser(args.target_path))
        info_dir = os.path.abspath(os.path.expanduser(args.datasets_info_path))
    else:
        try:
            workspace_root = resolve_workspace_root(args.workspace)
        except ValueError as e:
            print(f"[ERROR] {e}")
            print(
                "[ERROR] Либо задайте workspace, либо все три флага: "
                "--source-path, --target-path, --datasets-info-path."
            )
            return
        layout = WorkspaceLayout(workspace_root)
        os.makedirs(layout.datasets, exist_ok=True)
        info_dir = layout.datasets
        source_dir = layout.datasets
        if args.target_path:
            target_dir = os.path.abspath(os.path.expanduser(args.target_path))
        else:
            raw_out = (args.output_name or "").strip()
            if raw_out:
                workspace_out = raw_out
            else:
                workspace_out = (
                    f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{FUSION_DEFAULT_DIR_SUFFIX}"
                )
            target_dir = os.path.join(layout.datasets, workspace_out)

    json_file = os.path.join(info_dir, DATASETS_INFO_FILE)
    class_names_file = os.path.join(info_dir, CLASS_NAMES_FILE)

    with open(json_file, "r", encoding="utf-8") as f:
        datasets_info = json.load(f)

    with open(class_names_file, "r", encoding="utf-8") as f:
        class_names_map = json.load(f)

    output_dataset_name = os.path.basename(target_dir)
    selected_dataset_names = _parse_selected_datasets(args)
    interactive_mode = not selected_dataset_names
    available_dataset_names = sorted(
        [
            n
            for n in datasets_info.keys()
            if n != output_dataset_name and isinstance(datasets_info.get(n), dict)
        ]
    )

    if not selected_dataset_names:
        if not sys.stdin.isatty():
            print(
                "[ERROR] Не указаны входные датасеты. Используйте --dataset/--datasets "
                "или запустите в интерактивном терминале."
            )
            return
        try:
            selected_dataset_names = _prompt_dataset_selection(available_dataset_names)
        except Exception as e:
            print(f"[ERROR] Не удалось запустить интерактивный выбор датасетов: {e}")
            return

    if not selected_dataset_names:
        print("[ERROR] Не выбран ни один датасет для слияния.")
        return

    unknown = [n for n in selected_dataset_names if n not in datasets_info]
    if unknown:
        known = ", ".join(available_dataset_names)
        print(
            f"[ERROR] Неизвестные датасеты: {', '.join(unknown)}. "
            f"Доступные: {known}"
        )
        return

    if interactive_mode:
        class_candidates = all_classes_union_from_datasets(
            {k: v for k, v in datasets_info.items() if k in set(selected_dataset_names)},
            output_dataset_name,
            class_names_map,
        )
        try:
            _prompt_interactive_options(
                args,
                default_output_name=output_dataset_name,
                class_candidates=class_candidates,
            )
        except Exception as e:
            print(f"[ERROR] Ошибка интерактивного ввода параметров fusion: {e}")
            return
        if layout is not None and not args.target_path:
            out_key = (args.output_name or "").strip() or output_dataset_name
            target_dir = os.path.join(layout.datasets, out_key)
            output_dataset_name = out_key

    try:
        train_part, val_part, test_part = parse_fusion_split_arg(args.fusion_split)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    if args.fusion_split and str(args.fusion_split).strip():
        print(
            f"[INFO] --fusion-split: train={train_part}, val={val_part}, test={test_part} "
            "(переразбиение внутри каждого bucket исходного датасета)"
        )

    class_candidates_for_selected = all_classes_union_from_datasets(
        {k: v for k, v in datasets_info.items() if k in set(selected_dataset_names)},
        output_dataset_name,
        class_names_map,
    )

    # Выбранные классы
    if args.classes:
        selected_classes = [cls.strip() for cls in args.classes.split(",") if cls.strip()]
        if not selected_classes:
            print("[ERROR] Параметр --classes задан, но список имён пуст.")
            return
        is_valid, missing_classes = _validate_requested_classes(
            selected_classes,
            class_candidates_for_selected,
            class_names_map,
        )
        if not is_valid:
            print(
                "[ERROR] В --classes указаны неизвестные для выбранных датасетов классы: "
                f"{', '.join(missing_classes)}"
            )
            if class_candidates_for_selected:
                print(
                    "[INFO] Доступные классы выбранных датасетов: "
                    f"{', '.join(class_candidates_for_selected)}"
                )
            return
        classes_auto = False
    else:
        selected_classes = class_candidates_for_selected
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

    require_all_classes = not args.include_partial_datasets
    if args.include_partial_datasets:
        print(
            "[INFO] --include-partial-datasets: в слияние входят датасеты, "
            "у которых есть хотя бы один из выбранных классов."
        )

    matching_datasets = []
    for dataset_name, info in datasets_info.items():
        if dataset_name == output_dataset_name:
            continue
        if dataset_name not in selected_dataset_names:
            continue
        if dataset_matches_selection(
            info,
            class_names_map,
            selected_classes,
            merge_targets_to_sources,
            require_all_classes=require_all_classes,
        ):
            matching_datasets.append((dataset_name, info))

    if not matching_datasets:
        if require_all_classes:
            print("[ERROR] Ни один датасет не содержит все выбранные классы.")
            print(
                "[INFO] Подсказка: --include-partial-datasets — брать датасеты с любым "
                "подмножеством выбранных классов и объединять кадры со всех таких источников."
            )
        else:
            print("[ERROR] Ни один датасет не пересекается с выбранными классами.")
        return

    print(f"[INFO] Найдено {len(matching_datasets)} подходящих датасета:")
    for name, _ in matching_datasets:
        print(f"   - {name}")

    temp_ctx = None
    buckets_by_dataset: dict[str, list[tuple[str, str]]] = {}
    if args.tmp_dir:
        temp_root = os.path.abspath(os.path.expanduser(args.tmp_dir))
        os.makedirs(temp_root, exist_ok=True)
    elif layout is not None:
        temp_root = os.path.join(layout.root, "tmp")
        os.makedirs(temp_root, exist_ok=True)
    else:
        # Legacy mode: временные файлы только рядом с рабочими данными, не в системном /tmp.
        legacy_tmp_parent = os.path.join(source_dir, "tmp")
        os.makedirs(legacy_tmp_parent, exist_ok=True)
        temp_ctx = tempfile.TemporaryDirectory(prefix="smartrain_cvat11_", dir=legacy_tmp_parent)
        temp_root = temp_ctx.name

    total_labels = 0
    try:
        for dataset_name, info in matching_datasets:
            if "structure" not in info:
                print(f"[ERROR] В записи {dataset_name!r} нет поля structure.")
                return
            dataset_path = resolve_dataset_root_for_entry(
                dataset_name,
                info,
                workspace_root=workspace_root,
                source_catalog_dir=layout.datasets if layout else source_dir,
                legacy_source_parent=source_dir,
            )
            buckets = iter_image_label_buckets(
                dataset_path,
                info["structure"],
                info,
                dataset_name=dataset_name,
                temp_root=temp_root,
                exclude_test=args.exclude_test,
            )
            buckets_by_dataset[dataset_name] = buckets
            for _, labels_path in buckets:
                total_labels += len([f for f in os.listdir(labels_path) if f.endswith(".txt")])

        used_stems = {split: set() for split in ("train", "valid", "test")}
        copied_count = 0

        with tqdm(total=total_labels, desc="Обработка датасетов", unit="файл") as pbar:
            for dataset_name, info in matching_datasets:
                buckets = buckets_by_dataset[dataset_name]
                for images_path, labels_path in buckets:

                    pairs = []
                    for label_file in os.listdir(labels_path):
                        if not label_file.endswith(".txt"):
                            continue
                        image_name = os.path.splitext(label_file)[0]
                        for ext in list(YOLO_IMAGE_EXTS):
                            candidate = os.path.join(images_path, image_name + ext)
                            if os.path.exists(candidate):
                                pairs.append((candidate, os.path.join(labels_path, label_file)))
                                break

                    if not pairs:
                        continue

                    random.shuffle(pairs)
                    n = len(pairs)
                    train_split = pairs[: int(n * train_part)]
                    val_split = pairs[
                        int(n * train_part) : int(n * (train_part + val_part))
                    ]
                    test_split = pairs[int(n * (val_part + train_part)) :]

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
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

    if args.drop_empty_images:
        pruned = prune_output_empty_label_pairs(target_dir)
        if pruned:
            print(f"[INFO] --drop-empty-images: удалено пар без объектов: {pruned}")
            copied_count = max(0, copied_count - pruned)

    print(f"\n[DEBUG] Всего label-файлов: {total_labels}")
    print(f"[DEBUG] Отфильтровано и скопировано: {copied_count}")
    pct = (copied_count / total_labels * 100) if total_labels else 0.0
    print(f"[DEBUG] Процент используемых файлов: {pct:.2f}%")

    print(f"\n[OK] Скопировано {copied_count} изображений с фильтрованными аннотациями.")

    yaml_path = os.path.join(target_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("train: ./train/images\n")
        f.write("val: ./valid/images\n")
        f.write("test: ./test/images\n\n")
        f.write(f"nc: {len(selected_classes)}\n")
        f.write(f"names: {selected_classes}\n")

    print(f"[OK] Итоговый YAML создан: {yaml_path}")

    if layout is not None:
        out_key = os.path.basename(os.path.normpath(target_dir))
        _update_datasets_sidecar(layout, out_key, selected_classes, target_dir)
        print(f"[OK] Обновлены {layout.work_datasets_info_path()} и class_names.json в datasets/")
        try:
            source_datasets = []
            for ds_name, info in matching_datasets:
                source_datasets.append(
                    {
                        "name": ds_name,
                        "path": resolve_dataset_root_for_entry(
                            ds_name,
                            info,
                            workspace_root=workspace_root,
                            source_catalog_dir=layout.datasets,
                            legacy_source_parent=source_dir,
                        ),
                        "dataset_hash": info.get("dataset_hash"),
                    }
                )
            passport_path = write_dataset_passport(
                output_dataset_dir=target_dir,
                command="fusion",
                source_datasets=source_datasets,
                parameters=vars(args),
                transformations=[
                    {
                        "selected_classes": list(selected_classes),
                        "merge_classes": args.merge_classes or [],
                        "fusion_split": [train_part, val_part, test_part],
                        "include_partial_datasets": bool(args.include_partial_datasets),
                        "common_classes_only": bool(args.common_classes_only),
                        "exclude_test": bool(args.exclude_test),
                        "drop_empty_images": bool(args.drop_empty_images),
                    }
                ],
                random_seed=12345,
                stats_before={"total_labels": total_labels},
                stats_after={"copied_images": copied_count},
            )
            print(f"[OK] Passport: {passport_path}")
        except Exception as e:
            print(f"[WARNING] Не удалось записать dataset_passport.json: {e}")


if __name__ == "__main__":
    main()
