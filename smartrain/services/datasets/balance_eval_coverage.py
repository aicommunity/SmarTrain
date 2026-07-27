from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

def _ensure_non_empty_eval_splits(
    balanced_train: list[tuple[str, str, str, list[str]]],
    passthrough_items: list[tuple[str, str, str, list[str]]],
    *,
    seed: int,
    eval_min_class_count: int = 0,
) -> list[tuple[str, str, str, list[str]]]:
    """
    Ensure val/test are not empty in output when possible.
    We keep the current logic as-is unless one of eval splits is empty.
    If needed, move a deterministic subset of balanced-train items to val/test
    targeting roughly 80/10/10 split.
    """
    out_train = list(balanced_train)
    passthrough_keys = {_source_image_key(img) for _s, img, _lbl, _cls in passthrough_items}

    def train_groups() -> dict[str, list[int]]:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, (s, img, _lbl, _cls) in enumerate(out_train):
            if s != "train":
                continue
            groups[_source_image_key(img)].append(i)
        return groups

    def movable_train_keys() -> list[str]:
        groups = train_groups()
        return [key for key in groups.keys() if key not in passthrough_keys]

    def move_key_to_split(key: str, target_split: str) -> int:
        moved = 0
        for i in train_groups().get(key, []):
            s, img, lbl, cls = out_train[i]
            if s != target_split:
                out_train[i] = (target_split, img, lbl, cls)
                moved += 1
        return moved

    val_count = sum(1 for s, *_ in passthrough_items if s == "val")
    test_count = sum(1 for s, *_ in passthrough_items if s == "test")
    total = len(out_train) + len(passthrough_items)
    if total < 3:
        return out_train
    have_non_empty_eval = val_count > 0 and test_count > 0

    target_val = max(1, int(round(total * 0.1)))
    target_test = max(1, int(round(total * 0.1)))
    need_val = max(0, target_val - val_count)
    need_test = max(0, target_test - test_count)
    # Keep at least one source key in train and move only split-safe keys.
    can_move = max(0, len(movable_train_keys()) - 1)
    if need_val + need_test > can_move:
        # Prioritize making both splits non-empty first.
        min_need_val = 1 if val_count == 0 and can_move > 0 else 0
        min_need_test = 1 if test_count == 0 and can_move > min_need_val else 0
        left = max(0, can_move - min_need_val - min_need_test)
        need_val = min_need_val
        need_test = min_need_test
        # Distribute remaining budget approximately evenly.
        add_val = min(left // 2 + left % 2, max(0, target_val - val_count - need_val))
        need_val += add_val
        left -= add_val
        add_test = min(left, max(0, target_test - test_count - need_test))
        need_test += add_test

    if (not have_non_empty_eval) and (need_val > 0 or need_test > 0):
        rng = random.Random(seed)
        keys = movable_train_keys()
        rng.shuffle(keys)

        pos = 0
        for _ in range(need_val):
            if pos >= len(keys):
                break
            key = keys[pos]
            pos += 1
            move_key_to_split(key, "val")
        for _ in range(need_test):
            if pos >= len(keys):
                break
            key = keys[pos]
            pos += 1
            move_key_to_split(key, "test")

        cur_val = sum(1 for s, *_ in out_train if s == "val") + val_count
        cur_test = sum(1 for s, *_ in out_train if s == "test") + test_count
        if val_count == 0 and cur_val == 0:
            print(
                "[WARN] eval_coverage: val is empty and could not be seeded; "
                "no split-safe train source keys are available."
            )
        if test_count == 0 and cur_test == 0:
            print(
                "[WARN] eval_coverage: test is empty and could not be seeded; "
                "no split-safe train source keys are available."
            )

    # Optional class-coverage enrichment for eval splits:
    # if a class exists globally but is absent in val/test, move a minimal
    # number of train items containing that class to the target split.
    global_classes = {
        c
        for _s, _img, _lbl, cls_names in (out_train + passthrough_items)
        for c in cls_names
    }
    if not global_classes:
        return out_train

    rng_cov = random.Random(seed + 17)

    def present_classes(split_name: str) -> set[str]:
        out = set()
        for s, _img, _lbl, cls_names in out_train:
            if s == split_name:
                out.update(cls_names)
        for s, _img, _lbl, cls_names in passthrough_items:
            if s == split_name:
                out.update(cls_names)
        return out

    def train_count() -> int:
        return sum(1 for s, *_ in out_train if s == "train")

    for target_split in ("val", "test"):
        missing = set(global_classes) - present_classes(target_split)
        # keep at least one sample in train
        while missing and train_count() > 1:
            groups = train_groups()
            movable = [k for k in groups.keys() if k not in passthrough_keys]
            candidates: list[tuple[int, str]] = []
            for key in movable:
                idxs = groups.get(key, [])
                group_classes: set[str] = set()
                for i in idxs:
                    s, _img, _lbl, cls_names = out_train[i]
                    if s != "train":
                        continue
                    group_classes.update(cls_names)
                cover = len(group_classes & missing)
                if cover > 0:
                    candidates.append((cover, key))
            if not candidates:
                missing_preview = ", ".join(sorted(missing)[:8])
                movable_count = len(movable)
                print(
                    f"[WARN] eval_coverage: cannot cover missing classes in '{target_split}' "
                    f"without cross-split duplicate risk. Missing ({len(missing)}): {missing_preview}"
                    f"{' ...' if len(missing) > 8 else ''}; movable source keys: {movable_count}."
                )
                break
            # maximize coverage; tie-break with deterministic random jitter.
            max_cover = max(c for c, _ in candidates)
            best_keys = [k for c, k in candidates if c == max_cover]
            chosen_key = best_keys[rng_cov.randrange(len(best_keys))]
            covered: set[str] = set()
            for i in train_groups().get(chosen_key, []):
                s, _img, _lbl, cls_names = out_train[i]
                if s != "train":
                    continue
                covered.update(cls_names)
            move_key_to_split(chosen_key, target_split)
            missing -= covered

    # Optional eval-tail strengthening:
    # raise per-class bbox minimum in val/test by moving split-safe train groups.
    target_min = max(0, int(eval_min_class_count))
    if target_min <= 0:
        return out_train

    def split_bbox_counts(split_name: str) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for s, _img, _lbl, cls_names in out_train:
            if s != split_name:
                continue
            for c in cls_names:
                counts[c] += 1
        for s, _img, _lbl, cls_names in passthrough_items:
            if s != split_name:
                continue
            for c in cls_names:
                counts[c] += 1
        return counts

    for target_split in ("val", "test"):
        while train_count() > 1:
            current = split_bbox_counts(target_split)
            shortage = {c: target_min - int(current.get(c, 0)) for c in global_classes if int(current.get(c, 0)) < target_min}
            if not shortage:
                break
            groups = train_groups()
            movable = [k for k in groups.keys() if k not in passthrough_keys]
            best_key = None
            best_gain = 0
            for key in movable:
                idxs = groups.get(key, [])
                key_box_counts: dict[str, int] = defaultdict(int)
                for i in idxs:
                    s, _img, _lbl, cls_names = out_train[i]
                    if s != "train":
                        continue
                    for c in cls_names:
                        key_box_counts[c] += 1
                gain = 0
                for c, miss in shortage.items():
                    gain += min(int(miss), int(key_box_counts.get(c, 0)))
                if gain > best_gain:
                    best_gain = gain
                    best_key = key
            if best_key is None or best_gain <= 0:
                break
            move_key_to_split(best_key, target_split)
    return out_train



def _enforce_no_cross_split_duplicates(
    balanced_train: list[tuple[str, str, str, list[str]]],
    passthrough_items: list[tuple[str, str, str, list[str]]],
) -> tuple[list[tuple[str, str, str, list[str]]], list[tuple[str, str, str, list[str]]], int]:
    """
    Force each source image key to belong to exactly one split globally.
    Priority: train > val > test.
    """
    split_priority = {"train": 0, "val": 1, "test": 2}
    grouped: dict[str, list[str]] = defaultdict(list)
    for s, img, _lbl, _cls in balanced_train:
        grouped[_source_image_key(img)].append(s)
    for s, img, _lbl, _cls in passthrough_items:
        grouped[_source_image_key(img)].append(s)

    winner_by_key: dict[str, str] = {}
    for key, splits in grouped.items():
        winner_by_key[key] = min(splits, key=lambda s: split_priority.get(s, 99))

    changed = 0
    new_balanced: list[tuple[str, str, str, list[str]]] = []
    for s, img, lbl, cls in balanced_train:
        win = winner_by_key[_source_image_key(img)]
        if s != win:
            changed += 1
        new_balanced.append((win, img, lbl, cls))

    new_passthrough: list[tuple[str, str, str, list[str]]] = []
    for s, img, lbl, cls in passthrough_items:
        win = winner_by_key[_source_image_key(img)]
        if s != win:
            changed += 1
        new_passthrough.append((win, img, lbl, cls))

    return new_balanced, new_passthrough, changed



def _head_bbox_undersample_balanced_train(
    balanced_train: list[tuple[str, str, str, list[str]]],
    *,
    id_to_name: dict[int, str],
    cap_mult: float,
    seed: int,
    selected_classes: set[str],
) -> tuple[list[tuple[str, str, str, list[str]]], dict[str, object], list[set[int]]]:
    """Stratified removal of excess bbox lines for head classes (plan §7)."""
    empty_skips = [set() for _ in balanced_train]
    counts: Counter[str] = Counter()
    for _s, _img, _lbl, cls_names in balanced_train:
        for c in cls_names:
            if selected_classes and c not in selected_classes:
                continue
            counts[c] += 1
    if not counts:
        return balanced_train, {}, empty_skips

    vals_sorted = sorted(int(v) for v in counts.values())
    median_bbox = int(_quantile(vals_sorted, 0.5))
    if median_bbox <= 0:
        return balanced_train, {}, empty_skips

    cap_target = max(0, int(math.floor(float(cap_mult) * float(median_bbox))))
    omit: dict[int, set[int]] = {}

    for cls_name, n_raw in counts.items():
        if int(n_raw) <= cap_target:
            continue
        excess = int(n_raw) - cap_target
        pool: list[tuple[int, int]] = []
        for ti, (_sp, img, lbl, _cn) in enumerate(balanced_train):
            lines = _read_label_text_lines(lbl)
            for li, raw in enumerate(lines):
                cid = _line_class_id(raw)
                if cid is None:
                    continue
                name = id_to_name.get(cid, f"id_{cid}")
                if selected_classes and name not in selected_classes:
                    continue
                if name != cls_name:
                    continue
                pool.append((ti, li))
        pool.sort(key=lambda p: (Path(balanced_train[p[0]][1]).stem, p[0], p[1]))
        if not pool or excess <= 0:
            continue
        g_sz = min(32, max(1, len(pool)))
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for ti, li in pool:
            stem = Path(balanced_train[ti][1]).stem
            h = int(hashlib.md5(f"{seed}:{stem}".encode()).hexdigest(), 16)
            groups[h % g_sz].append((ti, li))
        removed_pairs: set[tuple[int, int]] = set()
        while len(removed_pairs) < excess and groups:
            best_g: int | None = None
            best_sz = -1
            for g in sorted(groups.keys()):
                sz = sum(1 for pair in groups[g] if pair not in removed_pairs)
                if sz > best_sz:
                    best_sz = sz
                    best_g = g
            if best_g is None or best_sz <= 0:
                break
            for pair in groups[best_g]:
                if pair not in removed_pairs:
                    removed_pairs.add(pair)
                    ti, li = pair
                    omit.setdefault(ti, set()).add(li)
                    break

    dropped_indices = set()
    for ti, it in enumerate(balanced_train):
        lines = _read_label_text_lines(it[2])
        kept_is = [j for j in range(len(lines)) if j not in omit.get(ti, set())]
        if not kept_is:
            dropped_indices.add(ti)

    after_counts: Counter[str] = Counter()
    for ti, (_s, _img, lbl, _cn) in enumerate(balanced_train):
        if ti in dropped_indices:
            continue
        lines = _read_label_text_lines(lbl)
        kept_lines = [lines[j] for j in range(len(lines)) if j not in omit.get(ti, set())]
        for raw in kept_lines:
            cid = _line_class_id(raw)
            if cid is None:
                continue
            name = id_to_name.get(cid, f"id_{cid}")
            if selected_classes and name not in selected_classes:
                continue
            after_counts[name] += 1

    per_class: dict[str, dict[str, int]] = {}
    for cls_name, before in counts.items():
        after = int(after_counts.get(cls_name, 0))
        per_class[str(cls_name)] = {"before": int(before), "after": after, "removed": max(0, int(before) - after)}

    new_train: list[tuple[str, str, str, list[str]]] = []
    for ti, it in enumerate(balanced_train):
        if ti in dropped_indices:
            continue
        lines = _read_label_text_lines(it[2])
        kept_lines = [lines[j] for j in range(len(lines)) if j not in omit.get(ti, set())]
        new_cls: list[str] = []
        for raw in kept_lines:
            cid = _line_class_id(raw)
            if cid is None:
                continue
            new_cls.append(id_to_name.get(cid, f"id_{cid}"))
        new_train.append((it[0], it[1], it[2], new_cls))

    stats: dict[str, object] = {
        "mode": "median-factor",
        "cap_mult": float(cap_mult),
        "median_bbox_per_class": median_bbox,
        "cap_target": cap_target,
        "per_class": per_class,
    }
    label_skips: list[set[int]] = []
    for ti, _it in enumerate(balanced_train):
        if ti in dropped_indices:
            continue
        label_skips.append(set(omit.get(ti, set())))
    return new_train, stats, label_skips


