#!/usr/bin/env python3
"""Dry-run balance preset harness (no YOLO train).

Compares weights-safe / rfs-aggressive / hybrid-default / irfs-default pools
and writes CSV+JSON under analytics/balance-harness/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Allow running as scripts/balance_preset_harness.py without install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from smartrain.core.runtime.workspace_paths import WorkspaceLayout, resolve_workspace_root
from smartrain.services.datasets.balance_presets import BALANCE_PRESETS
from smartrain.services.datasets.balance_strategies import _irfs_expand_pool, _rfs_expand_pool
from smartrain.services.datasets.dataset_access import (
    iter_image_label_buckets,
    resolve_dataset_root_for_entry,
)
from smartrain.services.datasets.dataset_cli_catalog import load_datasets_catalog

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_PRESETS = ("weights-safe", "rfs-aggressive", "hybrid-default", "irfs-default")


def _detect_split(images_path: str) -> str:
    low = images_path.replace("\\", "/").lower()
    for name in ("train", "val", "valid", "test"):
        if f"/{name}/" in f"/{low}/" or f"/{name}/images" in low:
            return "val" if name == "valid" else name
    return "train"


def _read_label_classes(lbl: str) -> list[int]:
    if not os.path.isfile(lbl):
        return []
    out: list[int] = []
    with open(lbl, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                try:
                    out.append(int(float(parts[0])))
                except ValueError:
                    continue
    return out


def _load_train_pool(layout: WorkspaceLayout, dataset: str) -> tuple[list[tuple], dict[str, int], dict[str, set[str]], dict[str, int]]:
    catalog = load_datasets_catalog(layout)
    if dataset not in catalog:
        raise KeyError(f"Unknown dataset: {dataset}")
    entry = catalog[dataset]
    class_map = entry.get("classes", {}) if isinstance(entry.get("classes"), dict) else {}
    id_to_name = {int(v): str(k) for k, v in class_map.items()}
    src_root = resolve_dataset_root_for_entry(
        dataset,
        entry,
        workspace_root=layout.root,
        source_catalog_dir=layout.datasets,
        legacy_source_parent=layout.datasets,
    )
    buckets = iter_image_label_buckets(
        src_root,
        str(entry.get("structure", "split")),
        entry,
        dataset_name=dataset,
        temp_root=os.path.join(layout.root, "tmp"),
        exclude_test=False,
    )
    pool: list[tuple[str, str, str, list[str]]] = []
    for images_path, labels_path in buckets:
        split = _detect_split(images_path)
        if split != "train":
            continue
        for name in sorted(os.listdir(images_path)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            img = os.path.join(images_path, name)
            lbl = os.path.join(labels_path, f"{stem}.txt")
            cls_ids = _read_label_classes(lbl)
            cls_names = [id_to_name.get(i, f"id_{i}") for i in cls_ids]
            pool.append((split, img, lbl, cls_names))
    bbox_count: dict[str, int] = defaultdict(int)
    image_presence: dict[str, int] = defaultdict(int)
    img_to_classes: dict[str, set[str]] = {}
    for _s, img, _l, cls_names in pool:
        classes = set(cls_names)
        img_to_classes[img] = classes
        for c in classes:
            image_presence[c] += 1
        for c in cls_names:
            bbox_count[c] += 1
    return pool, dict(bbox_count), img_to_classes, dict(image_presence)


def _bbox_counts(pool: list[tuple]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for item in pool:
        c.update(item[3])
    return dict(c)


def _head_tail_ratio(counts: dict[str, int]) -> float | None:
    if not counts:
        return None
    vals = sorted(counts.values())
    if vals[0] <= 0:
        return None
    return float(vals[-1]) / float(vals[0])


def _expand_for_preset(
    name: str,
    pool: list[tuple],
    bbox_count: dict[str, int],
    img_to_classes: dict[str, set[str]],
    image_presence: dict[str, int],
    rng: random.Random,
) -> list[tuple]:
    cfg = BALANCE_PRESETS.get(name, {})
    strategy = str(cfg.get("strategy", name))
    thresh = float(cfg.get("rfs_thresh", 0.001))
    power = float(cfg.get("rfs_power", 0.5))
    max_rep = int(cfg.get("max_repeat_per_image", 5))
    if strategy == "irfs":
        return _irfs_expand_pool(
            pool,
            bbox_count,
            rfs_thresh=thresh,
            rfs_power=power,
            max_repeat_per_image=max_rep,
            rng=rng,
        )
    if strategy in ("rfs", "hybrid", "hybrid-aug") or name.startswith("rfs") or name.startswith("hybrid"):
        return _rfs_expand_pool(
            pool,
            img_to_classes,
            image_presence,
            rfs_thresh=thresh,
            rfs_power=power,
            max_repeat_per_image=max_rep,
            rng=rng,
        )
    # weights-safe: no pool expansion in dry-run (report baseline)
    return list(pool)


def run_harness(
    *,
    workspace: str,
    dataset: str,
    presets: list[str] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    layout = WorkspaceLayout(workspace)
    pool, bbox_count, img_to_classes, image_presence = _load_train_pool(layout, dataset)
    presets = list(presets or DEFAULT_PRESETS)
    rng = random.Random(seed)
    before = _bbox_counts(pool)
    rows: list[dict[str, Any]] = []
    for name in presets:
        expanded = _expand_for_preset(name, pool, bbox_count, img_to_classes, image_presence, rng)
        after = _bbox_counts(expanded)
        rows.append(
            {
                "preset": name,
                "images_before": len(pool),
                "images_after": len(expanded),
                "bbox_before": before,
                "bbox_after": after,
                "head_tail_ratio_before": _head_tail_ratio(before),
                "head_tail_ratio_after": _head_tail_ratio(after),
            }
        )
    out_dir = os.path.join(layout.analytics, "balance-harness")
    os.makedirs(out_dir, exist_ok=True)
    payload = {"dataset": dataset, "seed": seed, "presets": rows}
    json_path = os.path.join(out_dir, f"{dataset}_harness.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    csv_path = os.path.join(out_dir, f"{dataset}_harness.csv")
    classes = sorted(before.keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["preset", "images_before", "images_after", "head_tail_before", "head_tail_after"]
        for c in classes:
            header.extend([f"{c}_before", f"{c}_after"])
        w.writerow(header)
        for row in rows:
            line = [
                row["preset"],
                row["images_before"],
                row["images_after"],
                row["head_tail_ratio_before"],
                row["head_tail_ratio_after"],
            ]
            bb, ba = row["bbox_before"], row["bbox_after"]
            for c in classes:
                line.extend([bb.get(c, 0), ba.get(c, 0)])
            w.writerow(line)
    payload["json_path"] = json_path
    payload["csv_path"] = csv_path
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Balance preset dry-run harness")
    p.add_argument("--workspace", type=str, default=None)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--presets", type=str, default=",".join(DEFAULT_PRESETS))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    workspace = resolve_workspace_root(args.workspace)
    presets = [x.strip() for x in str(args.presets).split(",") if x.strip()]
    result = run_harness(workspace=workspace, dataset=args.dataset, presets=presets, seed=int(args.seed))
    print(f"[OK] wrote {result['json_path']}")
    print(f"[OK] wrote {result['csv_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
