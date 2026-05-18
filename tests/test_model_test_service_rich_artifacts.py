from __future__ import annotations

import json
from pathlib import Path

import yaml

from smartrain.workflows.testing.model_test_service import get_test_artifacts_status
from smartrain.workflows.testing.ultralytics_test_contract import ultralytics_pt_rich_files_required


def _complete_rec() -> dict:
    return {
        "objectives": {
            "A": {"global": {"threshold": 0.1}},
            "B": {"global": {"threshold": 0.1}},
            "C": {"global": {"threshold": 0.1}},
        }
    }


def test_pt_rich_artifacts_require_all_detection_files(tmp_path: Path) -> None:
    run = tmp_path / "run1"
    tests = run / "tests"
    tu = tests / "test-ultralytics"
    tu.mkdir(parents=True)
    (run / "training_metadata.json").write_text(
        json.dumps({"training_info": {"task_type": "detection"}}), encoding="utf-8"
    )
    data_yaml = tmp_path / "ds" / "data.yaml"
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text(
        yaml.dump({"path": str(tmp_path / "ds"), "train": "images/train", "test": "images/test", "names": ["a"]}),
        encoding="utf-8",
    )
    (tu / "args.yaml").write_text(yaml.dump({"data": str(data_yaml)}), encoding="utf-8")
    (tests / "test_metrics.csv").write_text("mAP50-95,mAP50,Box-F1,Box-P,Box-R\n0,0,0,0,0\n", encoding="utf-8")
    for split in ("test", "val"):
        (tests / f"confidence_recommendations_{split}.json").write_text(
            json.dumps(_complete_rec()), encoding="utf-8"
        )
    for n in ("args.yaml", "pr.csv", "pr_per_class.csv"):
        (tu / n).write_text("x", encoding="utf-8")

    st = get_test_artifacts_status(str(run), "pt")
    assert not st.rich_artifacts_complete
    assert any(m.startswith("rich_artifact:") for m in st.missing)

    for name in ultralytics_pt_rich_files_required("detection"):
        p = tu / name
        if not p.is_file():
            if name.endswith(".png"):
                p.write_bytes(b"\x89PNG\r\n\x1a\n")
            else:
                p.write_text("x", encoding="utf-8")

    st2 = get_test_artifacts_status(str(run), "pt")
    assert st2.rich_artifacts_complete
    assert not any(m.startswith("rich_artifact:") for m in st2.missing)


def test_pt_rich_artifacts_segmentation_matches_detection_plot_set(tmp_path: Path) -> None:
    """Segment val still emits Box* curves from DetMetrics before mask branch (Ultralytics)."""
    run = tmp_path / "run_seg"
    tests = run / "tests"
    tu = tests / "test-ultralytics"
    tu.mkdir(parents=True)
    (run / "training_metadata.json").write_text(
        json.dumps({"training_info": {"task_type": "segmentation"}}), encoding="utf-8"
    )
    data_yaml = tmp_path / "ds2" / "data.yaml"
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text(
        yaml.dump({"path": str(tmp_path / "ds2"), "train": "images/train", "test": "images/test", "names": ["a"]}),
        encoding="utf-8",
    )
    (tu / "args.yaml").write_text(yaml.dump({"data": str(data_yaml)}), encoding="utf-8")
    (tests / "test_metrics.csv").write_text("mAP50-95,mAP50,Box-F1,Box-P,Box-R\n0,0,0,0,0\n", encoding="utf-8")
    for split in ("test", "val"):
        (tests / f"confidence_recommendations_{split}.json").write_text(
            json.dumps(_complete_rec()), encoding="utf-8"
        )
    for name in ultralytics_pt_rich_files_required("segmentation"):
        p = tu / name
        if name.endswith(".png"):
            p.write_bytes(b"\x89PNG\r\n\x1a\n")
        else:
            p.write_text("x", encoding="utf-8")

    st = get_test_artifacts_status(str(run), "pt")
    assert st.rich_artifacts_complete
    assert ultralytics_pt_rich_files_required("segmentation") == ultralytics_pt_rich_files_required("detection")
