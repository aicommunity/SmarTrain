from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from smartrain.workflows.datasets.dataset_orient import main as orient_main
from smartrain.workflows.datasets.datasets_json_former import main as scan_main
from smartrain.core.runtime.workspace_paths import deploy_workspace


def _make_ref_image(path: Path) -> None:
    """
    Create an asymmetric synthetic image with strong gradients:
    - thick vertical bar on the left
    - small square on the bottom-right
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 48), color=(20, 20, 20))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 10, 47], fill=(240, 240, 240))
    d.rectangle([50, 34, 62, 46], fill=(120, 200, 120))
    img.save(path, format="JPEG", quality=90)


def _rotate_90cw(in_path: Path, out_path: Path) -> None:
    img = Image.open(in_path).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.transpose(Image.Transpose.ROTATE_270).save(out_path, format="JPEG", quality=90)


def _prepare_workspace(tmp_path: Path) -> tuple[Path, Path]:
    deploy_workspace(str(tmp_path))
    raw = tmp_path / "raw_data" / "ds_o"
    (raw / "train" / "images").mkdir(parents=True, exist_ok=True)
    (raw / "train" / "labels").mkdir(parents=True, exist_ok=True)
    ref = raw / "train" / "images" / "ref.jpg"
    bad = raw / "train" / "images" / "bad.jpg"
    _make_ref_image(ref)
    _rotate_90cw(ref, bad)
    # bbox around the left bar in the *bad* image:
    # After 90cw rotation, the left vertical bar becomes a top horizontal bar.
    # Put a bbox around that top bar (roughly height 10px in a 48x64 image).
    (raw / "train" / "labels" / "bad.txt").write_text("0 0.50 0.10 1.00 0.20\n", encoding="utf-8")
    (raw / "data.yaml").write_text("nc: 1\nnames: ['obj']\n", encoding="utf-8")
    scan_main(["--workspace", str(tmp_path)])
    return ref, bad


def test_orient_creates_new_dataset_and_rotates_bbox(tmp_path: Path) -> None:
    ref, _bad = _prepare_workspace(tmp_path)
    orient_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_o",
            "--reference",
            str(ref),
            "--min-score",
            "1",
            "--on-uncertain",
            "fail",
            "--no-legend",
        ]
    )

    out = tmp_path / "datasets" / "ds_o_oriented"
    assert out.is_dir()
    assert (out / "dataset_passport.json").is_file()
    stats_csv = out / "orient_stats.csv"
    assert stats_csv.is_file()
    assert (
        "image_path,label_path,method,raw_k,offset_k,final_k,rotated,score_best,scores_json,uncertain,on_uncertain,action,output_image_path,output_label_path"
        in stats_csv.read_text(encoding="utf-8").splitlines()[0]
    )
    p = json.loads((out / "dataset_passport.json").read_text(encoding="utf-8"))
    assert p["command"] == "orient"

    # The output should contain bad.jpg (corrected orientation) and label.
    out_lbl = out / "train" / "labels" / "bad.txt"
    assert out_lbl.is_file()
    parts = out_lbl.read_text(encoding="utf-8").strip().split()
    assert len(parts) == 5
    # After correction, bbox should be tall and near the left side (x small, h > w).
    cx = float(parts[1])
    w = float(parts[3])
    h = float(parts[4])
    assert cx < 0.35
    assert h > w

    info = json.loads((tmp_path / "datasets" / "datasets_info.json").read_text(encoding="utf-8"))
    assert "ds_o_oriented" in info
    assert info["ds_o_oriented"]["data_path"] == "datasets/ds_o_oriented"


def test_orient_rotnet_saves_model_in_dataset_and_can_reuse(tmp_path: Path) -> None:
    ref, _bad = _prepare_workspace(tmp_path)
    # First run: train RotNet (fast: 1 epoch) and orient.
    orient_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_o",
            "--method",
            "rotnet",
            "--rotnet-epochs",
            "1",
            "--rotnet-image-size",
            "64",
            "--rotnet-device",
            "cpu",
            "--min-score",
            "0",
            "--on-uncertain",
            "keep",
            "--no-legend",
        ]
    )
    # Model should be saved inside source dataset root (datasets copy of ds_o).
    ds_root = tmp_path / "datasets" / "ds_o"
    model_path = ds_root / ".orient_rotnet" / "model.pt"
    assert model_path.is_file()

    # Second run: reuse saved model without training.
    orient_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_o",
            "--method",
            "rotnet",
            "--rotnet-epochs",
            "0",
            "--rotnet-device",
            "cpu",
            "--min-score",
            "0",
            "--on-uncertain",
            "keep",
            "--no-legend",
        ]
    )
    out2 = tmp_path / "datasets" / "ds_o_oriented_2"
    assert (out2 / "orient_stats.csv").is_file()


def test_orient_polygon_label_preserved(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    raw = tmp_path / "raw_data" / "ds_poly"
    (raw / "train" / "images").mkdir(parents=True, exist_ok=True)
    (raw / "train" / "labels").mkdir(parents=True, exist_ok=True)
    ref = raw / "train" / "images" / "ref.jpg"
    bad = raw / "train" / "images" / "bad.jpg"
    _make_ref_image(ref)
    _rotate_90cw(ref, bad)
    (raw / "train" / "labels" / "bad.txt").write_text(
        "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
        encoding="utf-8",
    )
    (raw / "data.yaml").write_text("nc: 1\nnames: ['obj']\n", encoding="utf-8")
    scan_main(["--workspace", str(tmp_path)])
    orient_main(
        [
            "--workspace",
            str(tmp_path),
            "--dataset",
            "ds_poly",
            "--reference",
            str(ref),
            "--min-score",
            "1",
            "--on-uncertain",
            "fail",
            "--no-legend",
        ]
    )
    out_lbl = tmp_path / "datasets" / "ds_poly_oriented" / "train" / "labels" / "bad.txt"
    assert out_lbl.is_file()
    from smartrain.services.datasets.yolo_labels import YoloSegment, read_yolo_labels

    labels = read_yolo_labels(str(out_lbl))
    assert len(labels) == 1
    assert isinstance(labels[0], YoloSegment)

