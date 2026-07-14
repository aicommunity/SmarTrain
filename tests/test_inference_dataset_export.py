from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from PIL import Image

from smartrain.services.datasets.yolo_labels import (
    YoloSegment,
    read_yolo_labels,
    task_output_dict_to_yolo_label,
)
from smartrain.services.inference_dataset_export import (
    ExportOptions,
    export_yolo_dataset,
    filter_task_outputs,
    resolve_autolabel_dataset_dir,
    resolve_export_options,
    validate_export_options,
)


def test_resolve_autolabel_dataset_dir() -> None:
    out = resolve_autolabel_dataset_dir("/tmp/run", "raw_images")
    assert out.name == "raw_images_autolabeled"


def test_task_output_dict_to_yolo_label_bbox() -> None:
    det = {
        "bbox_original_xyxy": [10.0, 20.0, 30.0, 40.0],
        "class_index": 1,
        "confidence": 0.9,
    }
    lb = task_output_dict_to_yolo_label(det, 100, 100)
    assert lb is not None
    assert lb.cls_id == 1
    assert lb.cx == pytest.approx(0.2)
    assert lb.cy == pytest.approx(0.3)


def test_task_output_dict_to_yolo_label_segmentation_prefers_polygon() -> None:
    seg = {
        "polygon_original_xy": [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]],
        "bbox_original_xyxy": [10.0, 10.0, 30.0, 30.0],
        "class_index": 2,
        "confidence": 0.88,
    }
    lb = task_output_dict_to_yolo_label(seg, 100, 100, task_type="segmentation")
    assert isinstance(lb, YoloSegment)
    assert lb.cls_id == 2
    assert len(lb.points) == 4


def test_export_yolo_dataset_segmentation_writes_polygon_labels(tmp_path: Path) -> None:
    src = tmp_path / "seg_images"
    src.mkdir()
    img = src / "a.jpg"
    Image.new("RGB", (100, 80), color=(1, 2, 3)).save(img)
    report = {
        "task_type": "segmentation",
        "source": {"mode": "folder", "path_absolute": str(src)},
        "model": {
            "source": "models",
            "name": "seg_demo",
            "weights_absolute": "/w/seg.pt",
            "provider": {"type": "builtin", "id": "ultralytics"},
        },
        "parameters": {"conf": 0.25, "img_size": 640, "device": "cpu", "half": False, "data_mode": "folder", "limit": 0},
        "images": [
            {
                "image_path_absolute": str(img),
                "image_size": {"width": 100, "height": 80},
                "task_outputs": {
                    "segments": [
                        {
                            "polygon_original_xy": [[10.0, 12.0], [40.0, 12.0], [40.0, 48.0], [10.0, 48.0]],
                            "class_index": 0,
                            "class_name": "obj",
                            "confidence": 0.91,
                        }
                    ]
                },
            }
        ],
    }
    out_root = tmp_path / "inference_run"
    out_root.mkdir()
    report_path = out_root / "inference_results.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    class _Layout:
        root = str(tmp_path)

    summary, _ = export_yolo_dataset(
        report,
        out_root=out_root,
        source_short="seg_images",
        report_path=report_path,
        options=ExportOptions(True, False, 0.25, 1.0, export_split_dirs=False),
        layout=_Layout(),  # type: ignore[arg-type]
    )
    label_path = out_root / "seg_images_autolabeled" / "labels" / "a.txt"
    assert summary.images_exported == 1
    assert label_path.is_file()
    raw = label_path.read_text(encoding="utf-8").strip().split()
    assert len(raw) > 5
    labels = read_yolo_labels(str(label_path))
    assert len(labels) == 1
    assert isinstance(labels[0], YoloSegment)


def test_export_yolo_dataset_does_not_create_dir_when_all_empty(tmp_path: Path) -> None:
    src = tmp_path / "raw_images"
    src.mkdir()
    img = src / "a.jpg"
    Image.new("RGB", (50, 50)).save(img)
    report = {
        "task_type": "detection",
        "source": {"mode": "folder", "path_absolute": str(src)},
        "model": {
            "source": "models",
            "name": "demo",
            "weights_absolute": "/w/demo.pt",
            "provider": {"type": "builtin", "id": "ultralytics"},
        },
        "parameters": {"conf": 0.25, "img_size": 640, "device": "cpu", "half": False, "data_mode": "folder", "limit": 0},
        "images": [
            {
                "image_path_absolute": str(img),
                "image_size": {"width": 50, "height": 50},
                "task_outputs": {"detections": []},
            }
        ],
    }
    out_root = tmp_path / "inference_run"
    out_root.mkdir()
    report_path = out_root / "inference_results.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    class _Layout:
        root = str(tmp_path)

    summary, exported = export_yolo_dataset(
        report,
        out_root=out_root,
        source_short="raw_images",
        report_path=report_path,
        options=ExportOptions(True, True, 0.25, 1.0, export_split_dirs=False),
        layout=_Layout(),  # type: ignore[arg-type]
    )
    assert summary.dataset_dir is None
    assert summary.images_exported == 0
    assert not exported
    assert not (out_root / "raw_images_autolabeled").exists()


def test_wrap_inference_report_v2_includes_export_artifacts() -> None:
    from smartrain.run_model_contract.schema import wrap_inference_report_v2

    payload = {
        "task_type": "detection",
        "output": "/tmp/inference_results.json",
        "artifacts": {
            "environment_profile": {"path_absolute": "/tmp/environment_profile.json"},
            "autolabel_dataset": {"path_absolute": "/tmp/raw_autolabeled", "images_exported": 2},
            "pred_overlays": {"path_absolute": "/tmp/pred_overlays", "images_rendered": 2},
        },
    }
    out = wrap_inference_report_v2(payload)
    assert out["v2"]["artifacts"]["autolabel_dataset"]["images_exported"] == 2
    assert out["v2"]["artifacts"]["pred_overlays"]["images_rendered"] == 2


def test_filter_task_outputs_conf_range() -> None:
    row = {
        "task_outputs": {
            "detections": [
                {"confidence": 0.2, "class_index": 0},
                {"confidence": 0.5, "class_index": 1},
                {"confidence": 0.9, "class_index": 2},
            ]
        }
    }
    filtered = filter_task_outputs(row, "detection", 0.4, 0.8)
    assert len(filtered) == 1
    assert filtered[0]["class_index"] == 1


def test_resolve_export_options_visualize_default() -> None:
    args = argparse.Namespace(export_dataset=True, export_visualize=None)
    opts = resolve_export_options(args)
    assert opts.export_visualize is True
    args2 = argparse.Namespace(export_dataset=False, export_visualize=None)
    assert resolve_export_options(args2).export_visualize is False


def test_validate_export_options_rejects_invalid_range() -> None:
    with pytest.raises(ValueError):
        validate_export_options(ExportOptions(True, True, 0.8, 0.2))


def test_export_yolo_dataset_skips_empty_and_writes_manifest(tmp_path: Path) -> None:
    src = tmp_path / "raw_images"
    src.mkdir()
    img_a = src / "a.jpg"
    img_b = src / "b.jpg"
    Image.new("RGB", (100, 80), color=(1, 2, 3)).save(img_a)
    Image.new("RGB", (100, 80), color=(4, 5, 6)).save(img_b)

    report = {
        "task_type": "detection",
        "source": {"mode": "folder", "path_absolute": str(src)},
        "model": {
            "source": "models",
            "name": "demo",
            "weights_absolute": "/w/demo.pt",
            "provider": {"type": "builtin", "id": "ultralytics"},
        },
        "parameters": {"conf": 0.25, "img_size": 640, "device": "cpu", "half": False, "data_mode": "folder", "limit": 0},
        "images": [
            {
                "image_path_absolute": str(img_a),
                "image_size": {"width": 100, "height": 80},
                "task_outputs": {
                    "detections": [
                        {
                            "bbox_original_xyxy": [10.0, 12.0, 40.0, 48.0],
                            "class_index": 0,
                            "class_name": "obj",
                            "confidence": 0.91,
                        }
                    ]
                },
            },
            {
                "image_path_absolute": str(img_b),
                "image_size": {"width": 100, "height": 80},
                "task_outputs": {"detections": []},
            },
        ],
    }
    out_root = tmp_path / "inference_run"
    out_root.mkdir()
    report_path = out_root / "inference_results.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    class _Layout:
        root = str(tmp_path)

    summary, exported = export_yolo_dataset(
        report,
        out_root=out_root,
        source_short="raw_images",
        report_path=report_path,
        options=ExportOptions(True, True, 0.25, 1.0, export_split_dirs=False),
        layout=_Layout(),  # type: ignore[arg-type]
    )
    dataset_dir = out_root / "raw_images_autolabeled"
    assert summary.images_exported == 1
    assert summary.images_skipped_empty == 1
    assert (dataset_dir / "images" / "a.jpg").is_file()
    assert (dataset_dir / "labels" / "a.txt").is_file()
    assert not (dataset_dir / "images" / "b.jpg").exists()
    labels = read_yolo_labels(str(dataset_dir / "labels" / "a.txt"))
    assert len(labels) == 1
    manifest = json.loads((dataset_dir / "autolabel_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["name"] == "demo"
    assert manifest["summary"]["images_exported"] == 1
    assert len(exported) == 1


def test_export_yolo_dataset_unique_stems_for_collision(tmp_path: Path) -> None:
    src = tmp_path / "nested"
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir(parents=True)
    img1 = src / "a" / "same.jpg"
    img2 = src / "b" / "same.jpg"
    Image.new("RGB", (50, 50)).save(img1)
    Image.new("RGB", (50, 50)).save(img2)
    det = {
        "bbox_original_xyxy": [5.0, 5.0, 20.0, 20.0],
        "class_index": 0,
        "class_name": "obj",
        "confidence": 0.8,
    }
    report = {
        "task_type": "detection",
        "source": {"mode": "folder", "path_absolute": str(src)},
        "model": {"source": "weights", "name": "demo", "weights_absolute": "/w.pt", "provider": {"type": "builtin", "id": "ultralytics"}},
        "parameters": {"conf": 0.25, "img_size": 640, "device": "cpu", "half": False, "data_mode": "folder", "limit": 0},
        "images": [
            {"image_path_absolute": str(img1), "image_size": {"width": 50, "height": 50}, "task_outputs": {"detections": [det]}},
            {"image_path_absolute": str(img2), "image_size": {"width": 50, "height": 50}, "task_outputs": {"detections": [dict(det)]}},
        ],
    }
    out_root = tmp_path / "run"
    out_root.mkdir()
    report_path = out_root / "inference_results.json"
    report_path.write_text("{}", encoding="utf-8")

    class _Layout:
        root = str(tmp_path)

    summary, _ = export_yolo_dataset(
        report,
        out_root=out_root,
        source_short="nested",
        report_path=report_path,
        options=ExportOptions(True, False, 0.25, 1.0, export_split_dirs=False),
        layout=_Layout(),  # type: ignore[arg-type]
    )
    dataset_dir = out_root / "nested_autolabeled"
    assert summary.images_exported == 2
    assert (dataset_dir / "labels" / "same.txt").is_file()
    assert (dataset_dir / "labels" / "same__2.txt").is_file()


def _det_row(img: Path, conf: float = 0.9) -> dict:
    return {
        "image_path_absolute": str(img),
        "image_size": {"width": 50, "height": 50},
        "task_outputs": {
            "detections": [
                {
                    "bbox_original_xyxy": [5.0, 5.0, 20.0, 20.0],
                    "class_index": 0,
                    "class_name": "obj",
                    "confidence": conf,
                }
            ]
        },
    }


def test_export_yolo_independent_parts_and_skip_empty_slot(tmp_path: Path) -> None:
    src = tmp_path / "raw"
    src.mkdir()
    imgs = []
    for name in ("a.jpg", "b.jpg", "c.jpg", "empty.jpg"):
        p = src / name
        Image.new("RGB", (50, 50)).save(p)
        imgs.append(p)

    report = {
        "task_type": "detection",
        "source": {"mode": "folder", "path_absolute": str(src)},
        "model": {
            "source": "models",
            "name": "demo",
            "weights_absolute": "/w.pt",
            "provider": {"type": "builtin", "id": "ultralytics"},
        },
        "parameters": {"conf": 0.25, "img_size": 640, "device": "cpu", "half": False},
        "images": [
            _det_row(imgs[0]),
            {"image_path_absolute": str(imgs[3]), "image_size": {"width": 50, "height": 50}, "task_outputs": {"detections": []}},
            _det_row(imgs[1]),
            _det_row(imgs[2]),
        ],
    }
    out_root = tmp_path / "run"
    out_root.mkdir()
    report_path = out_root / "inference_results.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    class _Layout:
        root = str(tmp_path)

    summary, exported = export_yolo_dataset(
        report,
        out_root=out_root,
        source_short="raw",
        report_path=report_path,
        options=ExportOptions(True, False, 0.25, 1.0, export_split_dirs=True, export_files_per_dir=2),
        layout=_Layout(),  # type: ignore[arg-type]
    )
    root = out_root / "raw_autolabeled"
    assert summary.layout == "independent_parts"
    assert summary.parts_count == 2
    assert summary.images_exported == 3
    assert summary.images_skipped_empty == 1
    assert len(exported) == 3
    assert (root / "part_000" / "data.yaml").is_file()
    assert (root / "part_000" / "autolabel_manifest.json").is_file()
    assert (root / "part_001" / "data.yaml").is_file()
    assert len(list((root / "part_000" / "images").glob("*"))) == 2
    assert len(list((root / "part_001" / "images").glob("*"))) == 1
    root_manifest = json.loads((root / "autolabel_manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["export_parameters"]["layout"] == "independent_parts"
    assert root_manifest["export_parameters"]["parts"] == 2
    assert len(root_manifest["parts"]) == 2


def test_export_prediction_overlays_mirror_parts(tmp_path: Path) -> None:
    from smartrain.services.inference_dataset_export import export_prediction_overlays

    src = tmp_path / "raw"
    src.mkdir()
    imgs = []
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        p = src / name
        Image.new("RGB", (50, 50)).save(p)
        imgs.append(p)
    report = {
        "task_type": "detection",
        "source": {"mode": "folder"},
        "images": [_det_row(p) for p in imgs],
    }

    class _Layout:
        root = str(tmp_path)

    options = ExportOptions(True, True, 0.25, 1.0, export_split_dirs=True, export_files_per_dir=2)
    exported = {str(p.resolve()) for p in imgs}
    saved, overlay_dir = export_prediction_overlays(
        report,
        out_root=tmp_path / "run",
        layout=_Layout(),  # type: ignore[arg-type]
        exported_only=exported,
        use_export_filter=True,
        options=options,
    )
    assert overlay_dir is not None
    assert len(saved) == 3
    root = Path(overlay_dir)
    assert (root / "part_000" / "a.jpg").is_file()
    assert (root / "part_000" / "b.jpg").is_file()
    assert (root / "part_001" / "c.jpg").is_file()


def test_validate_export_options_rejects_bad_files_per_dir() -> None:
    with pytest.raises(ValueError):
        validate_export_options(ExportOptions(True, True, 0.25, 1.0, export_split_dirs=True, export_files_per_dir=0))
