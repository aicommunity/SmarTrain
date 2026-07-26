from __future__ import annotations

import json
import zipfile
from pathlib import Path

from smartrain.workflows.datasets.datasets_json_former import main as datasets_json_main
from smartrain.core.runtime.workspace_paths import CLASS_NAMES_FILE, DATASETS_INFO_FILE, deploy_workspace


def _make_flat_dataset(root: Path, name: str) -> Path:
    ds = root / name
    (ds / "images").mkdir(parents=True, exist_ok=True)
    (ds / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "images" / "img_001.jpg").write_bytes(b"\x00")
    (ds / "labels" / "img_001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (ds / "data.yaml").write_text("nc: 1\nnames: ['defect']\n", encoding="utf-8")
    return ds


def _zip_dataset(dataset_dir: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=str(file_path.relative_to(dataset_dir)))
    return zip_path


def test_datasets_list_mixed_directory_and_zip(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets_root"
    output_root = tmp_path / "output"
    datasets_root.mkdir()
    output_root.mkdir()

    ds_dir = _make_flat_dataset(tmp_path, "dataset_from_dir")
    ds_zip_src = _make_flat_dataset(tmp_path, "dataset_from_zip_src")
    ds_zip = _zip_dataset(ds_zip_src, tmp_path / "dataset_from_zip.zip")

    list_file = tmp_path / "datasets_list.txt"
    list_file.write_text(
        f"{ds_dir}\n# comment line\n{ds_zip}\n",
        encoding="utf-8",
    )

    datasets_json_main(
        [
            "--datasets-path",
            str(datasets_root),
            "--output-path",
            str(output_root),
            "--datasets-list",
            str(list_file),
        ]
    )

    info = json.loads((output_root / DATASETS_INFO_FILE).read_text(encoding="utf-8"))
    classes = json.loads((output_root / CLASS_NAMES_FILE).read_text(encoding="utf-8"))

    assert "dataset_from_dir" in info
    assert "dataset_from_zip" in info
    assert info["dataset_from_dir"]["structure"] == "flat"
    assert info["dataset_from_zip"]["structure"] == "flat"
    assert info["dataset_from_dir"]["data_path"] == str(ds_dir.resolve())
    assert info["dataset_from_zip"]["data_path"] == str(ds_zip.resolve())
    assert classes["defect"] == "defect"


def test_datasets_list_relative_path_resolution(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets_root"
    output_root = tmp_path / "output"
    lists_root = tmp_path / "lists"
    datasets_root.mkdir()
    output_root.mkdir()
    lists_root.mkdir()

    ds_dir = _make_flat_dataset(tmp_path, "relative_dataset")
    list_file = lists_root / "datasets_list.txt"
    list_file.write_text("../relative_dataset\n", encoding="utf-8")

    datasets_json_main(
        [
            "--datasets-path",
            str(datasets_root),
            "--output-path",
            str(output_root),
            "--datasets-list",
            str(list_file),
        ]
    )

    info = json.loads((output_root / DATASETS_INFO_FILE).read_text(encoding="utf-8"))
    assert "relative_dataset" in info
    assert info["relative_dataset"]["data_path"] == str(ds_dir.resolve())


def test_workspace_auto_reads_default_datasets_list(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    deploy_workspace(str(workspace))

    ds_dir = _make_flat_dataset(tmp_path, "auto_list_dataset")
    list_file = workspace / "raw_data" / "datasets_list.txt"
    list_file.write_text(f"{ds_dir}\n", encoding="utf-8")

    datasets_json_main(["--workspace", str(workspace)])

    info = json.loads((workspace / "datasets" / DATASETS_INFO_FILE).read_text(encoding="utf-8"))
    assert "auto_list_dataset" in info
    assert info["auto_list_dataset"]["structure"] == "flat"
    assert "dataset_hash" in info["auto_list_dataset"]
    assert "source_hash" in info["auto_list_dataset"]
    assert "source_ref" in info["auto_list_dataset"]


def test_workspace_dataset_data_path_is_posix_relative(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    deploy_workspace(str(workspace))
    ds_dir = _make_flat_dataset(workspace / "raw_data", "under_ws_ds")
    list_file = workspace / "raw_data" / "datasets_list.txt"
    list_file.write_text(f"{ds_dir}\n", encoding="utf-8")

    datasets_json_main(["--workspace", str(workspace)])

    info = json.loads((workspace / "datasets" / DATASETS_INFO_FILE).read_text(encoding="utf-8"))
    assert info["under_ws_ds"]["data_path"] == "datasets/under_ws_ds"
    assert "\\" not in info["under_ws_ds"]["data_path"]
    src_ref = str(info["under_ws_ds"].get("source_ref") or "")
    assert src_ref == "raw_data/under_ws_ds"
    assert "\\" not in src_ref
