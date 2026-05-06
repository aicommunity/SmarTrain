from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from PIL import Image

from smartrain.workflows.datasets.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.workflows.datasets.dataset_hash import main as hash_main
from smartrain.workflows.datasets.dataset_roi_yolo import _ensure_data_yaml_after_roi, main as roi_main
from smartrain.datasets_json_former import main as datasets_json_main
from smartrain.workspace_paths import DATASETS_INFO_FILE, deploy_workspace, resolve_path_under_workspace


def _write_minimal_data_yaml(images_dir: Path) -> None:
    root = images_dir.parent
    (root / "data.yaml").write_text("nc: 1\nnames: ['cls0']\n", encoding="utf-8")


def _write_jpg(path: Path, *, size: tuple[int, int] = (64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color=(1, 2, 3))
    im.save(path, format="JPEG", quality=85)


def _make_cvat11_tree(nested: Path) -> None:
    images_dir = nested / "images"
    _write_jpg(images_dir / "img001.jpg", size=(100, 80))
    (nested / "annotations.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <name>taskX</name>
      <labels>
        <label><name>cat</name><type>bbox</type><attributes></attributes></label>
      </labels>
    </task>
  </meta>
  <image id="0" name="img001.jpg" width="100" height="80">
    <box label="cat" xtl="10" ytl="10" xbr="60" ybr="50" occluded="0" z_order="0"></box>
  </image>
</annotations>
""",
        encoding="utf-8",
    )


def test_iter_image_label_buckets_cvat11(tmp_path: Path) -> None:
    ds_root = tmp_path / "ds" / "payload"
    _make_cvat11_tree(ds_root)
    tmp_labels = tmp_path / "tmp"
    tmp_labels.mkdir()
    info = {
        "classes": {"cat": 0},
        "structure": "cvat11",
    }
    buckets = iter_image_label_buckets(
        str(ds_root),
        "cvat11",
        info,
        dataset_name="ds1",
        temp_root=str(tmp_labels),
        exclude_test=False,
    )
    assert len(buckets) == 1
    img_d, lbl_d = buckets[0]
    assert Path(lbl_d).name == "ds1"
    assert any(Path(lbl_d).rglob("*.txt"))


def test_resolve_dataset_root_zip_workspace(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    flat = tmp_path / "flat_ds"
    (flat / "images").mkdir(parents=True)
    (flat / "labels").mkdir(parents=True)
    _write_jpg(flat / "images" / "a.jpg", size=(32, 32))
    (flat / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _write_minimal_data_yaml(flat / "images")

    zip_path = tmp_path / "datasets" / "packed.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in flat.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=str(f.relative_to(flat)))

    rel = "datasets/packed.zip"
    info_path = tmp_path / "datasets" / DATASETS_INFO_FILE
    info_path.write_text(
        json.dumps(
            {
                "packed": {
                    "classes": {"cls0": 0},
                    "structure": "flat",
                    "data_path": rel,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    entry = json.loads(info_path.read_text(encoding="utf-8"))["packed"]
    root = resolve_dataset_root_for_entry(
        "packed",
        entry,
        workspace_root=str(tmp_path),
        source_catalog_dir=str(tmp_path / "datasets"),
        legacy_source_parent=str(tmp_path / "datasets"),
    )
    assert Path(root).is_dir()
    assert (Path(root) / "images" / "a.jpg").is_file()


def test_hash_source_dataset_after_zip_resolve(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    flat = tmp_path / "flat_ds"
    (flat / "images").mkdir(parents=True)
    (flat / "labels").mkdir(parents=True)
    _write_jpg(flat / "images" / "a.jpg", size=(32, 32))
    (flat / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _write_minimal_data_yaml(flat / "images")

    zip_path = tmp_path / "raw_data" / "hzip.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in flat.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=str(f.relative_to(flat)))

    datasets_json_main(["--workspace", str(tmp_path)])

    import io
    import sys

    buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = buf
        sys.stderr = io.StringIO()
        hash_main(
            [
                "--workspace",
                str(tmp_path),
                "--raw-dataset",
                "hzip",
            ]
        )
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    finally:
        sys.stdout, sys.stderr = old_out, old_err

    h = buf.getvalue().strip()
    assert len(h) == 8
    assert all(c in "0123456789abcdef" for c in h)


def _zip_flat(flat: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in flat.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=str(f.relative_to(flat)))


def test_hash_zip_metadata_flag(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    flat = tmp_path / "meta_flat"
    (flat / "images").mkdir(parents=True)
    (flat / "labels").mkdir(parents=True)
    _write_jpg(flat / "images" / "a.jpg", size=(16, 16))
    (flat / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _write_minimal_data_yaml(flat / "images")
    zip_path = tmp_path / "raw_data" / "meta.zip"
    _zip_flat(flat, zip_path)
    datasets_json_main(["--workspace", str(tmp_path)])

    import io
    import sys

    buf = io.StringIO()
    old_out = sys.stdout
    try:
        sys.stdout = buf
        hash_main(
            [
                "--workspace",
                str(tmp_path),
                "--raw-dataset",
                "meta",
                "--hash-zip-metadata",
            ]
        )
    except SystemExit as e:
        assert e.code in (0, None)
    finally:
        sys.stdout = old_out

    h = buf.getvalue().strip()
    assert len(h) == 8


def test_ensure_data_yaml_after_roi_when_no_source_yaml(tmp_path: Path) -> None:
    out = tmp_path / "roi_out"
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    _ensure_data_yaml_after_roi(
        str(out),
        {"classes": {"zebra": 1, "apple": 0}},
    )
    y = (out / "data.yaml").read_text(encoding="utf-8")
    assert "nc: 2" in y or "nc: 2\n" in y
    assert "apple" in y and "zebra" in y
    cfg = yaml.safe_load(y)
    assert "path" not in cfg
    assert cfg.get("train") == "images" and cfg.get("val") == "images"


def test_roi_dataset_name_with_zip_suffix_resolves_key(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    sd = tmp_path / "raw_data" / "dszip"
    (sd / "images").mkdir(parents=True)
    (sd / "labels").mkdir(parents=True)
    _write_jpg(sd / "images" / "a.jpg", size=(32, 32))
    (sd / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _write_minimal_data_yaml(sd / "images")
    datasets_json_main(["--workspace", str(tmp_path)])

    fake_result = MagicMock()
    fake_result.boxes = None
    fake_model = MagicMock()
    fake_model.task = "detect"
    fake_model.predict = MagicMock(return_value=[fake_result])
    out_dir = tmp_path / "datasets" / "from_zip_suffix"

    with patch("smartrain.workflows.datasets.dataset_roi_yolo.YOLO", return_value=fake_model):
        roi_main(
            [
                "--workspace",
                str(tmp_path),
                "--dataset-name",
                "dszip.zip",
                "--output-path",
                str(out_dir),
                "--weights",
                str(tmp_path / "w.pt"),
            ]
        )

    assert (out_dir / "images" / "a.jpg").is_file()


def test_roi_workspace_flat_mock_yolo(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    sd = tmp_path / "raw_data" / "roi_ds"
    (sd / "images").mkdir(parents=True)
    (sd / "labels").mkdir(parents=True)
    _write_jpg(sd / "images" / "one.jpg", size=(64, 64))
    (sd / "labels" / "one.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _write_minimal_data_yaml(sd / "images")

    datasets_json_main(["--workspace", str(tmp_path)])
    out_dir = tmp_path / "datasets" / "roi_ds_roi_test"

    fake_result = MagicMock()
    fake_result.boxes = None

    fake_model = MagicMock()
    fake_model.task = "detect"
    fake_model.predict = MagicMock(return_value=[fake_result])

    with patch("smartrain.workflows.datasets.dataset_roi_yolo.YOLO", return_value=fake_model):
        roi_main(
            [
                "--workspace",
                str(tmp_path),
                "--dataset-name",
                "roi_ds",
                "--output-path",
                str(out_dir),
                "--weights",
                str(tmp_path / "dummy.pt"),
            ]
        )

    assert (out_dir / "images" / "one.jpg").is_file()
    assert (out_dir / "labels" / "one.txt").is_file()
    assert (out_dir / "dataset_passport.json").is_file()


def test_roi_workspace_external_path_via_datasets_list(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    ext = tmp_path / "outside" / "ext_ds"
    (ext / "images").mkdir(parents=True)
    (ext / "labels").mkdir(parents=True)
    _write_jpg(ext / "images" / "e.jpg", size=(48, 48))
    (ext / "labels" / "e.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _write_minimal_data_yaml(ext / "images")

    list_file = tmp_path / "raw_data" / "datasets_list.txt"
    list_file.write_text(f"{ext}\n", encoding="utf-8")
    datasets_json_main(["--workspace", str(tmp_path)])

    info = json.loads((tmp_path / "datasets" / DATASETS_INFO_FILE).read_text(encoding="utf-8"))
    key = "ext_ds"
    assert key in info
    # scan copies external datasets to datasets and indexes the copy (raw_data - source only)
    assert resolve_path_under_workspace(str(tmp_path), info[key]["data_path"]) == str(
        (tmp_path / "datasets" / key).resolve()
    )

    out_dir = tmp_path / "datasets" / "ext_roi"
    fake_result = MagicMock()
    fake_result.boxes = None
    fake_model = MagicMock()
    fake_model.task = "detect"
    fake_model.predict = MagicMock(return_value=[fake_result])

    with patch("smartrain.workflows.datasets.dataset_roi_yolo.YOLO", return_value=fake_model):
        roi_main(
            [
                "--workspace",
                str(tmp_path),
                "--dataset-name",
                key,
                "--output-path",
                str(out_dir),
                "--weights",
                str(tmp_path / "dummy.pt"),
            ]
        )

    assert (out_dir / "images" / "e.jpg").is_file()


def test_roi_partial_args_do_not_trigger_interactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from smartrain.workspace_paths import DATASETS_INFO_FILE

    root = str(tmp_path)
    ds_dir = tmp_path / "datasets" / "roi_ds"
    (ds_dir / "images").mkdir(parents=True, exist_ok=True)
    (ds_dir / "labels").mkdir(parents=True, exist_ok=True)
    _write_jpg(ds_dir / "images" / "one.jpg")
    (ds_dir / "labels" / "one.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    _write_minimal_data_yaml(ds_dir / "images")
    (tmp_path / "datasets" / DATASETS_INFO_FILE).write_text(
        json.dumps({"roi_ds": {"classes": {"cls0": 0}, "structure": "flat"}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    with pytest.raises(SystemExit) as e:
        roi_main(["--workspace", root])
    out = (capsys.readouterr().out + capsys.readouterr().err + str(e.value)).lower()
    assert "incomplete arguments" in out.lower()


def test_roi_multiple_datasets_batch_mode(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    rd = tmp_path / "raw_data"
    for name in ("ds1", "ds2"):
        ds = rd / name
        (ds / "images").mkdir(parents=True, exist_ok=True)
        (ds / "labels").mkdir(parents=True, exist_ok=True)
        _write_jpg(ds / "images" / "a.jpg", size=(64, 64))
        (ds / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        _write_minimal_data_yaml(ds / "images")
    # make ds2 content different so scan keeps both
    _write_jpg((rd / "ds2" / "images" / "a.jpg"), size=(80, 60))
    datasets_json_main(["--workspace", str(tmp_path)])
    out_base = tmp_path / "datasets" / "roi_batch_out"
    fake_result = MagicMock()
    fake_result.boxes = None
    fake_model = MagicMock()
    fake_model.task = "detect"
    fake_model.predict = MagicMock(return_value=[fake_result])
    with patch("smartrain.workflows.datasets.dataset_roi_yolo.YOLO", return_value=fake_model):
        roi_main(
            [
                "--workspace",
                str(tmp_path),
                "--datasets",
                "ds1,ds2",
                "--output-path",
                str(out_base),
                "--weights",
                str(tmp_path / "dummy.pt"),
            ]
        )
    assert (out_base / "ds1_roi" / "images" / "a.jpg").is_file()
    assert (out_base / "ds2_roi" / "images" / "a.jpg").is_file()


def test_roi_legacy_direct_source_without_datasets_info(tmp_path: Path) -> None:
    ds_root = tmp_path / "some_dataset_root"
    (ds_root / "images").mkdir(parents=True, exist_ok=True)
    (ds_root / "labels").mkdir(parents=True, exist_ok=True)
    _write_jpg(ds_root / "images" / "one.jpg", size=(64, 64))
    (ds_root / "labels" / "one.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    _write_minimal_data_yaml(ds_root / "images")

    out_dir = tmp_path / "out_roi"
    fake_result = MagicMock()
    fake_result.boxes = None
    fake_model = MagicMock()
    fake_model.task = "detect"
    fake_model.predict = MagicMock(return_value=[fake_result])
    with patch("smartrain.workflows.datasets.dataset_roi_yolo.YOLO", return_value=fake_model):
        roi_main(
            [
                "--source-path",
                str(ds_root),
                "--output-path",
                str(out_dir),
                "--weights",
                str(tmp_path / "dummy.pt"),
            ]
        )

    assert (out_dir / "images" / "one.jpg").is_file()
    assert (out_dir / "labels" / "one.txt").is_file()


def test_roi_legacy_direct_images_only_without_labels(tmp_path: Path) -> None:
    ds_root = tmp_path / "imgs_only"
    ds_root.mkdir(parents=True, exist_ok=True)
    _write_jpg(ds_root / "a.jpg", size=(64, 64))

    out_dir = tmp_path / "out_roi_imgs_only"
    fake_result = MagicMock()
    fake_result.boxes = None
    fake_model = MagicMock()
    fake_model.task = "detect"
    fake_model.predict = MagicMock(return_value=[fake_result])
    with patch("smartrain.workflows.datasets.dataset_roi_yolo.YOLO", return_value=fake_model):
        roi_main(
            [
                "--source-path",
                str(ds_root),
                "--output-path",
                str(out_dir),
                "--weights",
                str(tmp_path / "dummy.pt"),
            ]
        )

    assert (out_dir / "a.jpg").is_file()
    assert not (out_dir / "labels").exists()
    assert not (out_dir / "images").exists()
    assert not (out_dir / "data.yaml").exists()


def test_roi_images_only_flag_ignores_existing_labels(tmp_path: Path) -> None:
    ds_root = tmp_path / "ds_with_labels"
    (ds_root / "images").mkdir(parents=True, exist_ok=True)
    (ds_root / "labels").mkdir(parents=True, exist_ok=True)
    _write_jpg(ds_root / "images" / "a.jpg", size=(64, 64))
    (ds_root / "labels" / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    _write_minimal_data_yaml(ds_root / "images")

    out_dir = tmp_path / "out_roi_images_only_flag"
    fake_result = MagicMock()
    fake_result.boxes = None
    fake_model = MagicMock()
    fake_model.task = "detect"
    fake_model.predict = MagicMock(return_value=[fake_result])
    with patch("smartrain.workflows.datasets.dataset_roi_yolo.YOLO", return_value=fake_model):
        roi_main(
            [
                "--source-path",
                str(ds_root),
                "--output-path",
                str(out_dir),
                "--weights",
                str(tmp_path / "dummy.pt"),
                "--images-only",
            ]
        )

    assert (out_dir / "images__a.jpg").is_file()
    assert not (out_dir / "labels").exists()
    assert not (out_dir / "images").exists()
    assert not (out_dir / "data.yaml").exists()
