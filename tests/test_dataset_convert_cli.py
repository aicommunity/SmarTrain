from __future__ import annotations

import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from smartrain.services.datasets.dataset_convert_cli import main as dataset_convert_main
from smartrain.services.datasets.dataset_convert_service import (
    TARGET_CVAT11,
    TARGET_YOLO,
    run_conversion,
    DatasetSource,
    ConvertOptions,
)


def _write_jpg(path: Path, *, size: tuple[int, int] = (200, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(30, 40, 50)).save(path, format="JPEG", quality=90)


def _write_cvsdcldet_sample(source: Path) -> None:
    _write_jpg(source / "img_a.jpg", size=(200, 100))
    (source / "img_a.json").write_text(
        json.dumps(
            {
                "detections": [
                    {"classId": 0, "class_name": "cat", "x": 10, "y": 20, "width": 50, "height": 30, "score": 0.9},
                ]
            }
        ),
        encoding="utf-8",
    )


def _make_cvat11_zip(tmp_path: Path) -> Path:
    task = tmp_path / "task1"
    images = task / "images"
    images.mkdir(parents=True, exist_ok=True)
    _write_jpg(images / "img001.jpg", size=(100, 80))
    annotations = task / "annotations.xml"
    annotations.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <name>task1</name>
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
    zip_path = tmp_path / "cvat11.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(annotations, arcname="task1/annotations.xml")
        zf.write(images / "img001.jpg", arcname="task1/images/img001.jpg")
    return zip_path


def _make_yolo_flat(tmp_path: Path) -> Path:
    root = tmp_path / "yolo_flat"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    _write_jpg(root / "images" / "img001.jpg", size=(100, 80))
    (root / "labels" / "img001.txt").write_text("0 0.35 0.375 0.5 0.5\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        "train: images\nval: images\ntest: images\nnc: 1\nnames: ['cat']\n",
        encoding="utf-8",
    )
    return root


def test_cvat_zip_to_yolo_service(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    cvat_zip = _make_cvat11_zip(tmp_path)
    out = tmp_path / "yolo_out"
    from smartrain.services.datasets.dataset_source_resolver import (
        resolved_to_dataset_source,
        resolve_dataset_source,
    )

    resolved = resolve_dataset_source(str(ws), cvat_zip)
    source = resolved_to_dataset_source(resolved)
    result = run_conversion(source, TARGET_YOLO, out, opts=ConvertOptions())
    assert result.output_dir is not None
    assert (result.output_dir / "data.yaml").is_file()
    assert (result.output_dir / "labels" / "img001.txt").is_file()


def test_yolo_to_cvat11_with_zip_service(tmp_path: Path) -> None:
    yolo = _make_yolo_flat(tmp_path)
    out = tmp_path / "cvat_out"
    source = DatasetSource(path=yolo, structure="flat", name=yolo.name)
    result = run_conversion(
        source,
        TARGET_CVAT11,
        out,
        opts=ConvertOptions(create_zip=True, delete_after_zip=True),
    )
    assert result.zip_path is not None
    assert result.zip_path.is_file()
    assert not out.is_dir()


def test_cvsdcldet_to_cvat11_cli(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "src"
    source.mkdir()
    _write_cvsdcldet_sample(source)
    out = tmp_path / "cvat_out"
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")
    dataset_convert_main(
        [
            "--source-dir",
            str(source),
            "--to",
            TARGET_CVAT11,
            "--output-dir",
            str(out),
            "--rename-classes",
            "cat",
            "feline",
        ]
    )
    tree = ET.parse(out / "annotations.xml")
    labels = {b.get("label") for b in tree.findall(".//box")}
    assert labels == {"feline"}


def test_cvsdcldet_to_cvat11_with_zip_delete_folder(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    _write_cvsdcldet_sample(source)
    out = tmp_path / "cvat_out"
    source_ds = DatasetSource(path=source, structure="cvsdcldet", name=source.name)
    result = run_conversion(
        source_ds,
        TARGET_CVAT11,
        out,
        opts=ConvertOptions(create_zip=True, delete_after_zip=True),
    )
    assert result.zip_path is not None
    assert result.zip_path.is_file()
    assert not out.is_dir()


def test_cvat_zip_import_cli(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    cvat_zip = _make_cvat11_zip(tmp_path)
    out = tmp_path / "yolo_cli"
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")
    monkeypatch.chdir(ws)
    dataset_convert_main(
        [
            "--source",
            str(cvat_zip),
            "--to",
            TARGET_YOLO,
            "--output-dir",
            str(out),
        ]
    )
    assert (out / "data.yaml").is_file()


def test_yolo_export_cli(tmp_path: Path, monkeypatch) -> None:
    yolo = _make_yolo_flat(tmp_path)
    out = tmp_path / "cvat_out"
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")
    dataset_convert_main(
        [
            "--source-dir",
            str(yolo),
            "--to",
            TARGET_CVAT11,
            "--output-dir",
            str(out),
            "--zip",
            "--delete-after-zip",
        ]
    )
    assert (out.parent / f"{out.name}.cvat11.zip").is_file()


def _make_cvsdcldet_zip(tmp_path: Path) -> Path:
    source = tmp_path / "src_build"
    source.mkdir()
    _write_cvsdcldet_sample(source)
    zip_path = tmp_path / "src.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=f"src/{path.name}")
    return zip_path


def test_cvsdcldet_zip_to_cvat11_cli(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "ws"
    raw = ws / "raw_data"
    raw.mkdir(parents=True)
    zip_path = _make_cvsdcldet_zip(tmp_path)
    target_zip = raw / "det.zip"
    target_zip.write_bytes(zip_path.read_bytes())
    out = ws / "converted_raw_data" / "out"
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")
    monkeypatch.setenv("SMART_TRAIN_WORKSPACE", str(ws))
    dataset_convert_main(
        [
            "--source",
            str(target_zip),
            "--to",
            TARGET_CVAT11,
            "--output-dir",
            str(out),
        ]
    )
    assert (out / "annotations.xml").is_file()


def test_cvsdcldet_zip_source_dir_flag(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    zip_path = _make_cvsdcldet_zip(tmp_path)
    out = tmp_path / "cvat_out"
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")
    monkeypatch.setenv("SMART_TRAIN_WORKSPACE", str(ws))
    dataset_convert_main(
        [
            "--source-dir",
            str(zip_path),
            "--to",
            TARGET_CVAT11,
            "--output-dir",
            str(out),
        ]
    )
    assert (out / "annotations.xml").is_file()


def test_interactive_raw_data_zip_selection(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "ws"
    raw = ws / "raw_data"
    raw.mkdir(parents=True)
    zip_path = _make_cvsdcldet_zip(tmp_path)
    target_zip = raw / "det_set.zip"
    target_zip.write_bytes(zip_path.read_bytes())
    out = ws / "converted_raw_data" / "out"
    label = None
    for candidate in __import__(
        "smartrain.services.datasets.dataset_source_resolver",
        fromlist=["list_raw_data_candidates"],
    ).list_raw_data_candidates(str(ws)):
        if candidate.path == target_zip.resolve():
            label = candidate.label
            break
    assert label is not None

    prompts = iter(
        [
            label,
            "CVAT for images 1.1 (folder: annotations.xml + images/)",
            str(out.relative_to(ws)),
            "n",
            "n",
        ]
    )
    monkeypatch.setenv("SMART_TRAIN_WORKSPACE", str(ws))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "smartrain.services.datasets.dataset_convert_cli.prompt_choice",
        lambda *_a, **_k: next(prompts),
    )
    monkeypatch.setattr(
        "smartrain.services.datasets.dataset_convert_cli.prompt_text",
        lambda *_a, **_k: next(prompts),
    )
    monkeypatch.setattr(
        "smartrain.services.datasets.dataset_convert_cli.prompt_yes_no",
        lambda *_a, **_k: next(prompts) == "y",
    )
    dataset_convert_main([])
    assert (out / "annotations.xml").is_file()
