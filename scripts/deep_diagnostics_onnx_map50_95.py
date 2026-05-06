from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics.utils.metrics import ap_per_class

from smartrain.workflows.testing.model_test_backends import _Gt, _Pred, _build_ultralytics_style_stats
from smartrain.workflows.testing.model_test_service import format_test_dir


def _load_names(data_yaml_path: str) -> list[str]:
    payload = yaml.safe_load(Path(data_yaml_path).read_text(encoding="utf-8")) or {}
    names = payload.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        # Ultralytics sometimes stores {id: name} as dict.
        try:
            return [str(v) for _k, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
        except Exception:
            return [str(v) for v in names.values()]
    return []


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _iter_jsonl(path: str):
    """Stream jsonl records to avoid holding all debug payload in memory."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _collect_preds_gts_from_records(records: list[dict[str, Any]]) -> tuple[list[_Pred], list[_Gt]]:
    preds: list[_Pred] = []
    gts: list[_Gt] = []
    for rec in records:
        image_path = str(rec.get("image_path", ""))
        for gt in rec.get("gts") or []:
            if not isinstance(gt, dict):
                continue
            try:
                gts.append(
                    _Gt(
                        image_path=image_path,
                        cls_id=int(gt.get("cls_id", 0)),
                        x1=float(gt.get("x1", 0.0)),
                        y1=float(gt.get("y1", 0.0)),
                        x2=float(gt.get("x2", 0.0)),
                        y2=float(gt.get("y2", 0.0)),
                    )
                )
            except Exception:
                continue
        for pred in rec.get("preds") or []:
            if not isinstance(pred, dict):
                continue
            try:
                preds.append(
                    _Pred(
                        image_path=image_path,
                        cls_id=int(pred.get("cls_id", 0)),
                        conf=float(pred.get("conf", 0.0)),
                        x1=float(pred.get("x1", 0.0)),
                        y1=float(pred.get("y1", 0.0)),
                        x2=float(pred.get("x2", 0.0)),
                        y2=float(pred.get("y2", 0.0)),
                    )
                )
            except Exception:
                continue
    return preds, gts


def _collect_preds_gts_from_jsonl_stream(path: str) -> tuple[dict[int, list[_Pred]], dict[int, list[_Gt]]]:
    preds_by_class: dict[int, list[_Pred]] = {}
    gts_by_class: dict[int, list[_Gt]] = {}
    for rec in _iter_jsonl(path):
        image_path = str(rec.get("image_path", ""))
        for gt in rec.get("gts") or []:
            if not isinstance(gt, dict):
                continue
            try:
                cid = int(gt.get("cls_id", 0))
                gts_by_class.setdefault(cid, []).append(
                    _Gt(
                        image_path=image_path,
                        cls_id=cid,
                        x1=float(gt.get("x1", 0.0)),
                        y1=float(gt.get("y1", 0.0)),
                        x2=float(gt.get("x2", 0.0)),
                        y2=float(gt.get("y2", 0.0)),
                    )
                )
            except Exception:
                continue
        for pred in rec.get("preds") or []:
            if not isinstance(pred, dict):
                continue
            try:
                cid = int(pred.get("cls_id", 0))
                preds_by_class.setdefault(cid, []).append(
                    _Pred(
                        image_path=image_path,
                        cls_id=cid,
                        conf=float(pred.get("conf", 0.0)),
                        x1=float(pred.get("x1", 0.0)),
                        y1=float(pred.get("y1", 0.0)),
                        x2=float(pred.get("x2", 0.0)),
                        y2=float(pred.get("y2", 0.0)),
                    )
                )
            except Exception:
                continue
    return preds_by_class, gts_by_class


def _read_deep_artifact_paths(root_dir: str, format_name: str, split: str) -> dict[str, str]:
    test_dir = format_test_dir(root_dir, format_name)
    deep_dir = os.path.join(test_dir, "deep_diagnostics")
    jsonl_path = os.path.join(deep_dir, f"debug_{split}.jsonl")
    summary_path = os.path.join(deep_dir, f"debug_{split}_summary.json")
    params_path = os.path.join(deep_dir, "debug_params.json")
    return {
        "deep_dir": deep_dir,
        "jsonl_path": jsonl_path,
        "summary_path": summary_path,
        "params_path": params_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep diagnostics analysis for ONNX mAP50-95 drop.")
    parser.add_argument("--root-dir", required=True, help="Run root directory (where test/, test_onnx/ live).")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Which split to analyze.")
    parser.add_argument("--pt-format", default="pt", help="Format name for PT baseline.")
    parser.add_argument("--onnx-format", default="onnx", help="Format name for ONNX comparison.")
    parser.add_argument("--out-dir", default=None, help="Output directory for graphs and markdown.")
    args = parser.parse_args()

    root_dir = str(Path(args.root_dir).resolve())
    split = str(args.split).strip().lower()
    pt_fmt = str(args.pt_format).strip().lower()
    onnx_fmt = str(args.onnx_format).strip().lower()

    out_dir = Path(args.out_dir) if args.out_dir else Path(root_dir) / "deep_diagnostics_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_paths = _read_deep_artifact_paths(root_dir, pt_fmt, split)
    onnx_paths = _read_deep_artifact_paths(root_dir, onnx_fmt, split)

    with open(pt_paths["summary_path"], "r", encoding="utf-8") as f:
        pt_summary = json.load(f)
    with open(onnx_paths["summary_path"], "r", encoding="utf-8") as f:
        onnx_summary = json.load(f)

    with open(pt_paths["params_path"], "r", encoding="utf-8") as f:
        pt_params = json.load(f)
    data_yaml_path = str(pt_params.get("data_yaml_path", "") or "")
    names = _load_names(data_yaml_path) if data_yaml_path else []

    # -------------------------------
    # 1) AP by IoU threshold (mean)
    # -------------------------------
    iou_thresholds = pt_summary.get("iou_thresholds") or np.linspace(0.5, 0.95, 10).astype(float).tolist()
    ap_pt = np.asarray(pt_summary.get("ap_mean_by_iou") or [0.0] * len(iou_thresholds), dtype=np.float32)
    ap_onnx = np.asarray(onnx_summary.get("ap_mean_by_iou") or [0.0] * len(iou_thresholds), dtype=np.float32)
    ap_delta = ap_onnx - ap_pt

    fig1_path = str(out_dir / "ap_by_iou_thresholds.png")
    plt.figure(figsize=(9, 5))
    x = np.arange(len(iou_thresholds))
    plt.plot(x, ap_pt, marker="o", linewidth=2, label=f"{pt_fmt} AP")
    plt.plot(x, ap_onnx, marker="o", linewidth=2, label=f"{onnx_fmt} AP")
    plt.xticks(x, [f"{t:.2f}" for t in iou_thresholds], rotation=0)
    plt.ylabel("Mean AP per IoU threshold")
    plt.xlabel("IoU threshold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.title(f"mAP50-95 components on {split.upper()}: {pt_fmt} vs {onnx_fmt}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=220)
    plt.close()

    fig2_path = str(out_dir / "ap_delta_by_iou_thresholds.png")
    plt.figure(figsize=(9, 5))
    plt.bar(x, ap_delta, color=["#b22222" if d < 0 else "#2e8b57" for d in ap_delta])
    plt.xticks(x, [f"{t:.2f}" for t in iou_thresholds])
    plt.ylabel(f"{onnx_fmt} - {pt_fmt} AP (mean)")
    plt.xlabel("IoU threshold")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.title("Where ONNX loses AP the most (by IoU threshold)")
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=220)
    plt.close()

    # -------------------------------
    # 2) TP/FP counts by IoU
    # -------------------------------
    tp_pt = np.asarray(pt_summary.get("tp_counts_by_iou") or [0] * len(iou_thresholds), dtype=np.int64)
    tp_onnx = np.asarray(onnx_summary.get("tp_counts_by_iou") or [0] * len(iou_thresholds), dtype=np.int64)
    fp_pt = np.asarray(pt_summary.get("fp_counts_by_iou") or [0] * len(iou_thresholds), dtype=np.int64)
    fp_onnx = np.asarray(onnx_summary.get("fp_counts_by_iou") or [0] * len(iou_thresholds), dtype=np.int64)

    fig3_path = str(out_dir / "tp_fp_counts_by_iou.png")
    plt.figure(figsize=(10, 5))
    plt.plot(x, tp_pt, marker="o", linewidth=2, label=f"{pt_fmt} TP")
    plt.plot(x, tp_onnx, marker="o", linewidth=2, label=f"{onnx_fmt} TP")
    plt.plot(x, fp_pt, marker="s", linewidth=1.5, linestyle="--", label=f"{pt_fmt} FP")
    plt.plot(x, fp_onnx, marker="s", linewidth=1.5, linestyle="--", label=f"{onnx_fmt} FP")
    plt.xticks(x, [f"{t:.2f}" for t in iou_thresholds])
    plt.ylabel("Counts")
    plt.xlabel("IoU threshold")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.title("Matching counts by IoU threshold (one-to-one within image)")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(fig3_path, dpi=220)
    plt.close()

    # -------------------------------
    # 3) Best-IoU hist for TP/FP at IoU=0.50
    # -------------------------------
    # best_iou bins are stored in debug jsonl payload, but the summary json doesn't include them.
    # We aggregate directly from jsonl.
    ti0 = 0  # corresponds to IoU threshold 0.50

    def _sum_best_iou_hist_stream(path: str, ti: int, which: str) -> tuple[list[float], np.ndarray]:
        # which in {"best_iou_tp_hist_by_iou", "best_iou_fp_hist_by_iou"}
        bins: list[float] | None = None
        total = None
        for rec in _iter_jsonl(path):
            matching = rec.get("matching") or {}
            if bins is None:
                bins = matching.get("best_iou_bins") or []
            hist_by_iou = matching.get(which) or []
            if isinstance(hist_by_iou, list) and ti < len(hist_by_iou):
                hist = np.asarray(hist_by_iou[ti], dtype=np.int64)
                total = hist if total is None else total + hist
        if bins is None or not bins:
            bins = list(np.linspace(0.0, 1.0, 11, dtype=np.float32))
        if total is None:
            total = np.zeros((len(bins) - 1,), dtype=np.int64)
        return bins, total

    bins_pt, tp_hist_pt = _sum_best_iou_hist_stream(pt_paths["jsonl_path"], ti0, "best_iou_tp_hist_by_iou")
    _, fp_hist_pt = _sum_best_iou_hist_stream(pt_paths["jsonl_path"], ti0, "best_iou_fp_hist_by_iou")
    bins_onnx, tp_hist_onnx = _sum_best_iou_hist_stream(onnx_paths["jsonl_path"], ti0, "best_iou_tp_hist_by_iou")
    _, fp_hist_onnx = _sum_best_iou_hist_stream(onnx_paths["jsonl_path"], ti0, "best_iou_fp_hist_by_iou")

    bins = np.asarray(bins_pt, dtype=np.float32)
    centers = (bins[:-1] + bins[1:]) / 2.0

    fig4_path = str(out_dir / "best_iou_hist_tp_fp_iou0_50.png")
    plt.figure(figsize=(10, 5))
    w = (bins[1] - bins[0]) * 0.4
    plt.bar(centers - w / 2, tp_hist_pt, width=w, label=f"{pt_fmt} TP", alpha=0.9)
    plt.bar(centers - w / 2, fp_hist_pt, width=w, label=f"{pt_fmt} FP", alpha=0.4, bottom=tp_hist_pt)
    plt.bar(centers + w / 2, tp_hist_onnx, width=w, label=f"{onnx_fmt} TP", alpha=0.9)
    plt.bar(centers + w / 2, fp_hist_onnx, width=w, label=f"{onnx_fmt} FP", alpha=0.4, bottom=tp_hist_onnx)
    plt.xlabel("Best IoU bin")
    plt.ylabel("Count of predictions (after NMS+matching within image)")
    plt.title(f"Localization diagnostics at IoU=0.50 ({split.upper()}): best-IoU histogram")
    plt.grid(True, axis="y", linestyle="--", alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(fig4_path, dpi=220)
    plt.close()

    # -------------------------------
    # 4) Per-class AP50-95 delta
    # -------------------------------
    iouv = np.linspace(0.5, 0.95, 10, dtype=np.float32)

    pt_preds_by_class, pt_gts_by_class = _collect_preds_gts_from_jsonl_stream(pt_paths["jsonl_path"])
    onnx_preds_by_class, onnx_gts_by_class = _collect_preds_gts_from_jsonl_stream(onnx_paths["jsonl_path"])

    class_ids = sorted(
        set(pt_preds_by_class.keys())
        | set(pt_gts_by_class.keys())
        | set(onnx_preds_by_class.keys())
        | set(onnx_gts_by_class.keys())
    )

    deltas: list[tuple[int, float, float, float]] = []
    for cid in class_ids:
        pt_preds_cls = pt_preds_by_class.get(cid, [])
        pt_gts_cls = pt_gts_by_class.get(cid, [])
        onnx_preds_cls = onnx_preds_by_class.get(cid, [])
        onnx_gts_cls = onnx_gts_by_class.get(cid, [])

        def _ap_for_one_class(preds_cls: list[_Pred], gts_cls: list[_Gt]) -> float:
            if not preds_cls or not gts_cls:
                return 0.0
            tp_cls, conf_cls, pred_cls_arr, target_cls_arr = _build_ultralytics_style_stats(preds_cls, gts_cls, iouv)
            names_map = {cid: (names[cid] if cid < len(names) else str(cid))} if names else {cid: str(cid)}
            _, _, _, _, _, ap, unique_classes, _, _, _, _, _ = ap_per_class(
                tp=tp_cls,
                conf=conf_cls,
                pred_cls=pred_cls_arr,
                target_cls=target_cls_arr,
                plot=False,
                names=names_map,
            )
            if not isinstance(unique_classes, np.ndarray) or not isinstance(ap, np.ndarray):
                return 0.0
            if ap.ndim != 2 or ap.shape[0] == 0:
                return 0.0
            return float(ap[0].mean())  # mean across IoUs -> AP50-95

        ap_pt = _ap_for_one_class(pt_preds_cls, pt_gts_cls)
        ap_onnx = _ap_for_one_class(onnx_preds_cls, onnx_gts_cls)
        deltas.append((cid, ap_pt, ap_onnx, ap_onnx - ap_pt))

    deltas.sort(key=lambda x: x[3])  # most negative first

    topk = min(15, len(deltas))
    top = deltas[:topk]
    top_labels = [names[cid] if cid < len(names) else str(cid) for cid, _, _, _ in top]
    top_delta = [d for _cid, _pt, _onnx, d in top]
    top_ap_pt = [_pt for _cid, _pt, _onnx, _d in top]
    top_ap_onnx = [_onnx for _cid, _pt, _onnx, _d in top]

    fig5_path = str(out_dir / "per_class_ap50_95_delta_top.png")
    plt.figure(figsize=(12, 6))
    x2 = np.arange(len(top_labels))
    plt.bar(x2 - 0.2, top_ap_pt, width=0.4, label=f"{pt_fmt}")
    plt.bar(x2 + 0.2, top_ap_onnx, width=0.4, label=f"{onnx_fmt}")
    plt.xticks(x2, top_labels, rotation=45, ha="right")
    plt.ylabel("AP50-95 (mean AP across IoUs)")
    plt.xlabel("Class")
    plt.title(f"Top classes by ONNX AP50-95 drop ({split.upper()})")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(fig5_path, dpi=220)
    plt.close()

    # -------------------------------
    # Markdown report
    # -------------------------------
    worst = sorted(deltas, key=lambda x: x[3])[:5]
    worst_lines = []
    for cid, appt, aponnx, d in worst:
        label = names[cid] if cid < len(names) else str(cid)
        worst_lines.append(f"- {label}: ΔAP50-95 = {d:+.4f} (pt={appt:.4f}, onnx={aponnx:.4f})")

    report_path = str(out_dir / "deep_diagnostics_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Deep diagnostics: ONNX mAP50-95 drop ({split.upper()})\n\n")
        f.write("## Goal\n")
        f.write(
            "Доказательно локализовать вклад IoU-threshold’ов и (по возможности) классов в падение mAP50-95 для ONNX по сравнению с PT.\n\n"
        )

        f.write("## How AP50-95 is computed\n\n")
        f.write(
            "1) Для каждого IoU threshold $t \\in \\{0.50, 0.55, \\dots, 0.95\\}$ строится AP(t).\n"
            "2) AP50-95 считается как среднее по $t$:\n\n"
            "$$\\text{mAP50-95} = \\frac{1}{10} \\sum_{t \\in \\{0.50..0.95\\}} AP(t).$$\n\n"
        )
        f.write(
            "3) Внутри каждого изображения применяется one-to-one сопоставление GT↔pred и затем TP/FP накапливаются по confidence-сортировке.\n\n"
        )

        f.write("## Key findings (from deep artifacts)\n\n")
        f.write(f"- Baseline: `{pt_fmt}`, compared: `{onnx_fmt}`\n")
        f.write(f"- PT mAP50-95: {pt_summary.get('map5095', 0.0):.6f}\n")
        f.write(f"- ONNX mAP50-95: {onnx_summary.get('map5095', 0.0):.6f}\n\n")

        f.write("### IoU thresholds that break first\n\n")
        f.write("| IoU | ΔAP (onnx - pt) |\n|---:|---:|\n")
        for ti, t in enumerate(iou_thresholds):
            f.write(f"| {float(t):.2f} | {float(ap_delta[ti]):+.4f} |\n")
        f.write("\n")

        f.write("### Top classes with largest AP50-95 drops\n\n")
        f.write("\n".join(worst_lines))
        f.write("\n\n")

        f.write("## Graphs\n\n")
        f.write(f"1. ![ap_by_iou_thresholds]({Path(fig1_path).name})\n")
        f.write(f"2. ![ap_delta_by_iou_thresholds]({Path(fig2_path).name})\n")
        f.write(f"3. ![tp_fp_counts_by_iou]({Path(fig3_path).name})\n")
        f.write(f"4. ![best_iou_hist_tp_fp_iou0_50]({Path(fig4_path).name})\n")
        f.write(f"5. ![per_class_ap50_95_delta_top]({Path(fig5_path).name})\n")

    print(f"[OK] Report generated: {report_path}")
    print(f"[OK] Graphs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

