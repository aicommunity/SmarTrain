from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from smartrain.core.runtime.path_portable import store_path_under_workspace
from smartrain.services.inference_runtime_helpers import IMAGE_EXTS
from smartrain.services.visualization.annotation_loader import load_gt_labels
from smartrain.services.visualization.color_registry import LabelColorRegistry
from smartrain.services.visualization.contracts import FrameRecord, VisFrameStatus, VisRequest, VisSummary
from smartrain.services.visualization.infer_adapter import run_inference_for_split
from smartrain.services.visualization.output_writer import append_index_row, write_config, write_summary
from smartrain.services.visualization.rendering import render_combined_overlay, save_rendered_image


def _portable_target_paths(layout_root: str, target: dict[str, Any]) -> dict[str, str | None]:
    def _one(key: str) -> str | None:
        val = target.get(key)
        if not val:
            return None
        return store_path_under_workspace(layout_root, str(val))

    return {
        "dataset_root": _one("dataset_root"),
        "run_dir": _one("run_dir"),
        "model_path": _one("model_path"),
    }


def _iter_images(split_dir: Path) -> list[Path]:
    return sorted(p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _collect_frames(split_dirs: dict[str, Path], limit: int | None) -> list[tuple[str, Path, Path]]:
    frames: list[tuple[str, Path, Path]] = []
    for split_name, split_dir in split_dirs.items():
        for image_path in _iter_images(split_dir):
            frames.append((split_name, split_dir, image_path))
            if limit and len(frames) >= int(limit):
                return frames
    return frames


def _target_output_base(mode: str, target: dict[str, Any]) -> Path:
    if mode == "dataset":
        return Path(target["dataset_root"]) / "visualize"
    return Path(target["run_dir"]) / "visualize" / "gt_pred_overlays"


def _frame_output_path(mode: str, output_base: Path, split: str, split_dir: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(split_dir)
    if mode == "dataset":
        return output_base / split / rel
    return output_base / split / rel


def _build_frame_record(mode: str, output_base: Path, split: str, split_dir: Path, image_path: Path) -> FrameRecord:
    label_path, _labels = load_gt_labels(image_path)
    output = _frame_output_path(mode, output_base, split, split_dir, image_path)
    return FrameRecord(
        split=split,
        source_abs=image_path.resolve(),
        source_rel=str(image_path.relative_to(split_dir).as_posix()),
        label_abs=label_path,
        output_abs=output.resolve(),
    )


def _render_loop(
    *,
    req: VisRequest,
    target: dict[str, Any],
    mode: str,
    with_predictions: bool,
) -> int:
    split_dirs: dict[str, Path] = target["split_dirs"]
    class_names: dict[int, str] = target.get("class_names", {}) or {}
    output_base = _target_output_base(mode, target)
    output_base.mkdir(parents=True, exist_ok=True)
    index_path = output_base.parent / "index.jsonl" if mode != "dataset" else output_base / "index.jsonl"
    summary_path = output_base.parent / "summary.json" if mode != "dataset" else output_base / "summary.json"
    config_path = output_base.parent / "config.json" if mode != "dataset" else output_base / "config.json"
    if index_path.is_file():
        index_path.unlink()
    started = datetime.now(timezone.utc).isoformat()
    total = ok = skipped = errors = 0
    color_registry = LabelColorRegistry(Path(target["layout"].root))
    for nm in class_names.values():
        color_registry.ensure(str(nm))
    pred_by_split: dict[str, dict[str, list[dict[str, Any]]]] = {}
    frames = _collect_frames(split_dirs, req.limit)
    if with_predictions:
        split_items = list(split_dirs.items())
        for split_name, split_dir in tqdm(
            split_items,
            desc=f"vis:infer:{mode}",
            unit="split",
            file=sys.stdout,
            disable=len(split_items) <= 1,
        ):
            pred_by_split[split_name] = run_inference_for_split(
                layout=target["layout"],
                weights_path=Path(target["model_path"]),
                split_dir=split_dir,
                device=req.device,
                conf=req.conf,
                limit=req.limit,
            )
    for split_name, split_dir, image_path in tqdm(
        frames,
        desc=f"vis:{mode}",
        unit="img",
        file=sys.stdout,
        disable=len(frames) == 0,
    ):
        total += 1
        fr = _build_frame_record(mode, output_base, split_name, split_dir, image_path)
        if fr.output_abs.is_file() and not req.overwrite:
            skipped += 1
            append_index_row(
                index_path,
                VisFrameStatus(
                    source_rel=f"{split_name}/{fr.source_rel}",
                    split=split_name,
                    status="skipped",
                    reason="exists",
                    gt_count=0,
                    pred_count=0,
                    output_rel=str(fr.output_abs),
                ),
            )
            continue
        try:
            _label_path, gt_labels = load_gt_labels(fr.source_abs)
            pred_rows: list[dict[str, Any]] = []
            if with_predictions:
                pred_rows = pred_by_split.get(split_name, {}).get(str(fr.source_abs.resolve()), [])
            for lb in gt_labels:
                color_registry.ensure(class_names.get(int(lb.cls_id), f"class_{int(lb.cls_id)}"))
            for det in pred_rows:
                raw_name = det.get("class_name")
                if isinstance(raw_name, str) and raw_name.strip():
                    color_registry.ensure(raw_name.strip())
                    continue
                raw_cid = det.get("class_id", det.get("class_index", -1))
                try:
                    cid = int(raw_cid)
                except Exception:
                    try:
                        cid = int(float(raw_cid))
                    except Exception:
                        cid = -1
                color_registry.ensure(class_names.get(cid, f"class_{cid}"))
            rendered, original_format = render_combined_overlay(
                fr.source_abs,
                gt_labels,
                pred_rows,
                class_names,
                label_colors=color_registry.colors_rgb(),
                gt_faded=with_predictions,
            )
            save_rendered_image(rendered, fr.output_abs, original_format=original_format)
            ok += 1
            append_index_row(
                index_path,
                VisFrameStatus(
                    source_rel=f"{split_name}/{fr.source_rel}",
                    split=split_name,
                    status="ok",
                    reason=None,
                    gt_count=len(gt_labels),
                    pred_count=len(pred_rows),
                    output_rel=str(fr.output_abs),
                ),
            )
        except Exception as exc:
            errors += 1
            append_index_row(
                index_path,
                VisFrameStatus(
                    source_rel=f"{split_name}/{fr.source_rel}",
                    split=split_name,
                    status="error",
                    reason=str(exc),
                    gt_count=0,
                    pred_count=0,
                    output_rel=None,
                ),
            )
    finished = datetime.now(timezone.utc).isoformat()
    layout_root = str(target["layout"].root)
    write_summary(
        summary_path,
        VisSummary(
            mode=mode,
            target=_portable_target_paths(layout_root, target),
            total_frames=total,
            ok_frames=ok,
            skipped_frames=skipped,
            error_frames=errors,
            started_at=started,
            finished_at=finished,
            config={
                "splits": list(split_dirs.keys()),
                "overwrite": req.overwrite,
                "limit": req.limit,
                "device": req.device,
                "conf": req.conf,
                "with_predictions": with_predictions,
            },
        ),
    )
    write_config(
        config_path,
        {
            "mode": mode,
            "dataset": req.dataset,
            "run_ref": req.run_ref,
            "model_name": req.model_name,
            "weights": req.weights,
            "splits": list(split_dirs.keys()),
            "limit": req.limit,
            "conf": req.conf,
            "device": req.device,
            "overwrite": req.overwrite,
        },
    )
    color_registry.save()
    print(f"[OK] Visualization output: {output_base}")
    return 0 if errors == 0 else 2


def visualize_dataset(req: VisRequest, target: dict[str, Any]) -> int:
    return _render_loop(req=req, target=target, mode="dataset", with_predictions=False)


def visualize_model(req: VisRequest, target: dict[str, Any]) -> int:
    return _render_loop(req=req, target=target, mode="model", with_predictions=True)


def visualize_run(req: VisRequest, target: dict[str, Any]) -> int:
    return _render_loop(req=req, target=target, mode="run", with_predictions=True)

