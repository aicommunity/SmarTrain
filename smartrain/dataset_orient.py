from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm

from smartrain.cli_argparse import CliArgumentParser
from smartrain.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.dataset_hash import calculate_dataset_hash
from smartrain.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.yolo_labels import read_yolo_labels, rotate_yolo_labels_90cw_k, write_yolo_labels


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ON_UNCERTAIN = ("keep", "skip", "fail")


@dataclass(frozen=True)
class _RefSig:
    desc: np.ndarray | None  # ORB descriptors
    grad_vec: tuple[float, float]  # normalized (ex, ey)
    tmpl: np.ndarray  # normalized template vector (flattened)


def build_orient_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Исправление поворотов 0/90/180/270 в датасете YOLO (в новый datasets/<name>)")
    p.add_argument("--workspace", type=str, default=None, help=f"Корень workspace (иначе {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Имя исходного датасета из datasets_info.json")
    p.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Имя выходного датасета (по умолчанию <dataset>_oriented)",
    )
    p.add_argument(
        "--reference",
        action="append",
        default=None,
        help="Путь к эталонному изображению в правильной ориентации (можно повторять).",
    )
    p.add_argument(
        "--min-score",
        type=int,
        default=8,
        help="Минимальный score уверенности (после объединения ORB+градиентного fallback).",
    )
    p.add_argument(
        "--on-uncertain",
        choices=ON_UNCERTAIN,
        default="keep",
        help="Что делать, если уверенность ниже --min-score: keep|skip|fail.",
    )
    p.add_argument("--report-only", action="store_true", help="Только отчёт по распределению углов, без записи датасета.")
    p.add_argument("--dry-run", action="store_true", help="Не писать файлы, но выполнять расчёт и печатать итог.")
    p.add_argument("--no-legend", action="store_true", help="Отключить прогресс-бар.")
    return p


def _load_catalog(layout: WorkspaceLayout) -> dict:
    p = layout.work_datasets_info_path()
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _interactive_fill(args, *, dataset_names: list[str]) -> None:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import WordCompleter

    print("[INFO] Интерактивный режим orient")
    print("[INFO] Доступные датасеты:")
    for n in dataset_names:
        print(f"  - {n}")
    args.dataset = prompt("Датасет: ", completer=WordCompleter(dataset_names, ignore_case=True)).strip()
    args.output_name = prompt("Имя выходного датасета (пусто=авто): ", default=(args.output_name or "")).strip() or None

    refs: list[str] = []
    print("[INFO] Эталоны (правильная ориентация). Введите пути, пустая строка = конец.")
    while True:
        r = prompt("reference: ", default="").strip()
        if not r:
            break
        refs.append(r)
    if refs:
        args.reference = refs

    raw = prompt("min_score: ", default=str(getattr(args, "min_score", 8))).strip()
    if raw:
        try:
            args.min_score = int(raw)
        except ValueError:
            pass
    args.on_uncertain = (
        prompt(
            "on_uncertain (keep/skip/fail): ",
            default=str(getattr(args, "on_uncertain", "keep")),
            completer=WordCompleter(list(ON_UNCERTAIN), ignore_case=True),
        ).strip()
        or getattr(args, "on_uncertain", "keep")
    )
    args.report_only = (prompt("report_only? [y/N]: ", default=("y" if args.report_only else "n")).strip().lower() in ("y", "yes", "1", "true", "да", "д"))
    args.dry_run = (prompt("dry_run? [y/N]: ", default=("y" if args.dry_run else "n")).strip().lower() in ("y", "yes", "1", "true", "да", "д"))


def _read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return img


def _rot_img_k(img: np.ndarray, k: int) -> np.ndarray:
    kk = int(k) % 4
    if kk == 0:
        return img
    if kk == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if kk == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _orb() -> cv2.ORB:
    # nfeatures повышаем: сцены с сеткой/шумом могут давать мало ключевых точек при дефолтах
    return cv2.ORB_create(nfeatures=1200, fastThreshold=7)


def _compute_grad_vec(gray: np.ndarray) -> tuple[float, float]:
    g = gray.astype(np.float32) / 255.0
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    ex = float(np.mean(np.abs(sx)))
    ey = float(np.mean(np.abs(sy)))
    s = ex + ey + 1e-9
    return ex / s, ey / s


def _template_vec(gray: np.ndarray, *, size: int = 64) -> np.ndarray:
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    v = small.reshape(-1)
    v = v - float(np.mean(v))
    n = float(np.linalg.norm(v) + 1e-9)
    return v / n


def _load_references(paths: list[str]) -> list[_RefSig]:
    o = _orb()
    out: list[_RefSig] = []
    for p in paths:
        gray = _read_gray(p)
        kp, desc = o.detectAndCompute(gray, None)
        _ = kp  # for symmetry, unused
        out.append(_RefSig(desc=desc, grad_vec=_compute_grad_vec(gray), tmpl=_template_vec(gray)))
    return out


def _score_orb(desc: np.ndarray | None, refs: list[_RefSig]) -> int:
    if desc is None or len(desc) == 0:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    score = 0
    for r in refs:
        if r.desc is None or len(r.desc) == 0:
            continue
        try:
            matches = bf.match(desc, r.desc)
        except cv2.error:
            continue
        if not matches:
            continue
        # Good matches by distance threshold
        good = [m for m in matches if m.distance <= 48.0]
        score += int(len(good))
    return int(score)


def _grad_distance(vec: tuple[float, float], refs: list[_RefSig]) -> float:
    if not refs:
        return 1.0
    rv = np.array([np.mean([r.grad_vec[0] for r in refs]), np.mean([r.grad_vec[1] for r in refs])], dtype=np.float32)
    v = np.array([vec[0], vec[1]], dtype=np.float32)
    return float(np.linalg.norm(v - rv, ord=2))


def _template_similarity(vec: np.ndarray, refs: list[_RefSig]) -> float:
    if not refs:
        return 0.0
    sims = [float(np.dot(vec, r.tmpl)) for r in refs]
    return float(max(sims)) if sims else 0.0


def _combined_score(orb_score: int, grad_dist: float, tmpl_sim: float) -> int:
    # grad_dist in [0..~1.4], map to [0..100] where 0 dist => 100
    grad_score = int(max(0.0, 100.0 * (1.0 - min(1.0, grad_dist))))
    tmpl_score = int(max(0.0, min(1.0, (tmpl_sim + 1.0) / 2.0)) * 120.0)  # [-1..1] -> [0..120]
    return int(orb_score * 2 + grad_score + tmpl_score)


def _choose_rotation(gray: np.ndarray, refs: list[_RefSig]) -> tuple[int, dict[int, int]]:
    o = _orb()
    scores: dict[int, int] = {}
    for k in (0, 1, 2, 3):
        g = _rot_img_k(gray, k)
        _kp, desc = o.detectAndCompute(g, None)
        orb_s = _score_orb(desc, refs)
        gd = _grad_distance(_compute_grad_vec(g), refs)
        ts = _template_similarity(_template_vec(g), refs)
        scores[k] = _combined_score(orb_s, gd, ts)
    best_k = max(scores.items(), key=lambda kv: kv[1])[0]
    return int(best_k), {int(k): int(v) for k, v in scores.items()}


def _copy_tree_structure_if_exists(src_root: str, dst_root: str) -> None:
    # copy data.yaml if present at top-level
    for name in ("data.yaml", "data.yml"):
        sp = os.path.join(src_root, name)
        if os.path.isfile(sp):
            os.makedirs(dst_root, exist_ok=True)
            shutil.copy2(sp, os.path.join(dst_root, name))


def _update_datasets_sidecar(
    layout: WorkspaceLayout,
    output_key: str,
    entry: dict,
    target_dir: str,
    output_hash: str,
) -> None:
    os.makedirs(layout.datasets, exist_ok=True)
    rel = os.path.relpath(os.path.abspath(target_dir), layout.root)
    prev: dict = {}
    info_path = layout.work_datasets_info_path()
    if os.path.isfile(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            prev = loaded
    new_entry = dict(entry) if isinstance(entry, dict) else {}
    new_entry["data_path"] = rel
    new_entry["dataset_hash"] = output_hash
    new_entry["modified"] = False
    prev[output_key] = new_entry
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=4)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args = build_orient_arg_parser().parse_args(argv)
    root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(root)
    catalog = _load_catalog(layout)
    if not catalog:
        print("[ERROR] Не найдено datasets_info.json или он пуст.")
        return

    if args.dataset is None and sys.stdin.isatty():
        _interactive_fill(args, dataset_names=sorted(catalog.keys()))
    if not args.dataset:
        print("[ERROR] Укажите --dataset или используйте интерактивный режим.")
        return
    if args.dataset not in catalog:
        print(f"[ERROR] Неизвестный датасет: {args.dataset}")
        return
    if not args.reference:
        print("[ERROR] Нужен хотя бы один --reference (эталон правильной ориентации).")
        return

    entry = catalog[args.dataset]
    structure = str(entry.get("structure", "split"))
    if structure == "cvat11":
        print("[ERROR] structure=cvat11 не поддерживается для orient (по договорённости работаем только с datasets/ без cvat11).")
        return

    refs = _load_references([str(p) for p in args.reference])

    src_root = resolve_dataset_root_for_entry(
        args.dataset,
        entry,
        workspace_root=layout.root,
        source_catalog_dir=layout.datasets,
        legacy_source_parent=layout.datasets,
    )
    buckets = iter_image_label_buckets(
        src_root,
        structure,
        entry,
        dataset_name=args.dataset,
        temp_root=os.path.join(layout.root, "tmp"),
        exclude_test=False,
    )
    if not buckets:
        print("[ERROR] Не найдено ни одной пары images/labels в датасете.")
        return

    out_base = args.output_name or f"{args.dataset}_oriented"
    out_name = next_dataset_name(layout.datasets, out_base)
    out_dir = os.path.join(layout.datasets, out_name)

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    uncertain = 0
    total = 0

    progress = tqdm(buckets, total=len(buckets), desc="orient:buckets", disable=bool(args.no_legend))
    for images_path, labels_path in progress:
        # Preserve bucket-relative layout
        rel_images = os.path.relpath(os.path.abspath(images_path), os.path.abspath(src_root))
        rel_labels = os.path.relpath(os.path.abspath(labels_path), os.path.abspath(src_root))
        # If layout is split-like, keep split folders under output root too.
        dst_images = os.path.join(out_dir, rel_images)
        dst_labels = os.path.join(out_dir, rel_labels)
        os.makedirs(dst_images, exist_ok=True)
        os.makedirs(dst_labels, exist_ok=True)

        for name in os.listdir(images_path):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            img_path = os.path.join(images_path, name)
            lbl_path = os.path.join(labels_path, f"{stem}.txt")
            try:
                gray = _read_gray(img_path)
            except Exception as e:
                print(f"[WARNING] Пропуск изображения: {img_path} ({e})")
                continue

            k, scores = _choose_rotation(gray, refs)
            score_best = scores.get(int(k), 0)
            total += 1
            counts[int(k)] = counts.get(int(k), 0) + 1

            is_uncertain = score_best < int(args.min_score)
            if is_uncertain:
                uncertain += 1
                if args.on_uncertain == "skip":
                    continue
                if args.on_uncertain == "fail":
                    raise RuntimeError(f"Неуверенный поворот для {img_path}: scores={scores}, chosen={k}")

            if args.report_only or args.dry_run:
                continue

            # rotate image and labels
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[WARNING] Не удалось прочитать изображение (color): {img_path}")
                continue
            rbgr = _rot_img_k(bgr, k)
            labels = read_yolo_labels(lbl_path)
            rotated_labels, new_w, new_h = rotate_yolo_labels_90cw_k(labels, w=int(bgr.shape[1]), h=int(bgr.shape[0]), k=int(k))
            # sanity: dimensions should match rotated image
            ih, iw = int(rbgr.shape[0]), int(rbgr.shape[1])
            if (iw, ih) != (new_w, new_h):
                # Should not happen, but keep consistent with actual image
                rotated_labels, _, _ = rotate_yolo_labels_90cw_k(
                    labels,
                    w=int(bgr.shape[1]),
                    h=int(bgr.shape[0]),
                    k=int(k),
                )

            out_img = os.path.join(dst_images, name)
            out_lbl = os.path.join(dst_labels, f"{stem}.txt")
            cv2.imwrite(out_img, rbgr)
            write_yolo_labels(out_lbl, rotated_labels)

    progress.close()

    print("[OK] orient report")
    print(f"  dataset: {args.dataset}")
    print(f"  output: {out_name}" if not args.report_only else "  output: <report-only>")
    print(f"  total_images: {total}")
    print(f"  rotations(k*90cw): {counts}")
    print(f"  uncertain: {uncertain} (min_score={int(args.min_score)}, on_uncertain={args.on_uncertain})")

    if args.report_only or args.dry_run:
        return

    _copy_tree_structure_if_exists(src_root, out_dir)
    out_hash = calculate_dataset_hash(out_dir)
    _update_datasets_sidecar(layout, out_name, entry if isinstance(entry, dict) else {}, out_dir, out_hash)
    write_dataset_passport(
        output_dataset_dir=out_dir,
        command="orient",
        source_datasets=[{"name": args.dataset, "dataset_hash": entry.get("dataset_hash") if isinstance(entry, dict) else None}],
        parameters={"dataset": args.dataset, "output_name": out_name, "reference": list(args.reference), "min_score": int(args.min_score), "on_uncertain": str(args.on_uncertain)},
        transformations=[{"type": "discrete_rotation_90", "angles": [0, 90, 180, 270], "method": "orb+grad_reference"}],
        random_seed=None,
    )

