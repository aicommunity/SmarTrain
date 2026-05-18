"""Canonical filenames for PT Ultralytics test artifacts (analyze + completeness)."""

from __future__ import annotations

# Files copied by analyze_ultralytics_test_service.collect_ultralytics_test_artifacts (image side).
ULTRALYTICS_TEST_COLLECT_IMAGE_NAMES: tuple[str, ...] = (
    "PR_curve.png",
    "BoxPR_curve.png",
    "F1_curve.png",
    "BoxF1_curve.png",
    "P_curve.png",
    "BoxP_curve.png",
    "R_curve.png",
    "BoxR_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "val_batch0_pred.jpg",
    "val_batch0_labels.jpg",
)


def ultralytics_pt_rich_files_required(task_type: str | None) -> tuple[str, ...]:
    """Filenames that must exist under tests/test-ultralytics for has_complete_test_artifacts(pt)."""
    t = (task_type or "detect").strip().lower()
    base = (
        "args.yaml",
        "pr.csv",
        "pr_per_class.csv",
    )
    # Ultralytics DetMetrics / SegmentMetrics both run box branch with prefix "Box" first; see
    # ultralytics.utils.metrics.SegmentMetrics.process (box curves + mask curves). Completeness for
    # SmarTrain analyze aligns with the same Box* + confusion plots as detection.
    detect_style = (
        "BoxF1_curve.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
    )
    if t in {"classification", "classify", "cls"}:
        return base
    if t in {"segmentation", "segment", "seg"}:
        return base + detect_style
    return base + detect_style


def native_format_rich_files_required() -> tuple[str, ...]:
    """Rich artifact names for ONNX/engine/TRT native eval (unchanged contract)."""
    return (
        "args.yaml",
        "pr.csv",
        "pr_per_class.csv",
        "BoxF1_curve.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
    )


def _read_training_task_type(root_dir: str) -> str | None:
    import json
    from pathlib import Path

    p = Path(root_dir) / "training_metadata.json"
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    ti = payload.get("training_info")
    if isinstance(ti, dict):
        raw = ti.get("task_type") or ti.get("task")
        if raw is not None and str(raw).strip():
            return str(raw).strip().lower()
    return None


def rich_files_required_for_format(root_dir: str, format_name: str) -> tuple[str, ...]:
    fmt = str(format_name or "pt").strip().lower()
    if fmt == "pt":
        return ultralytics_pt_rich_files_required(_read_training_task_type(root_dir))
    return native_format_rich_files_required()
