from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from smartrain.workflows.datasets.datasets_json_former import main as datasets_json_main
from smartrain.providers.core.global_index import list_provider_records, upsert_provider_record
from smartrain.core.runtime.workspace_paths import (
    CLASS_NAMES_FILE,
    DATASETS_INFO_FILE,
    DATASETS_SCAN_SUMMARY_FILE,
    deploy_workspace,
)


def _flat_dataset(root: Path, name: str, *, extra_classes: bool = False) -> None:
    ds = root / name
    (ds / "images").mkdir(parents=True, exist_ok=True)
    (ds / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "images" / "a.jpg").write_bytes(b"\x00")
    if extra_classes:
        (ds / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n", encoding="utf-8")
        (ds / "data.yaml").write_text(
            "train: images\nval: images\ntest: images\n\nnc: 3\nnames: ['bee', 'wasp', 'unused']\n",
            encoding="utf-8",
        )
    else:
        (ds / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (ds / "data.yaml").write_text("nc: 1\nnames: ['bee']\n", encoding="utf-8")


def _cvat11_dataset(root: Path, name: str, *, nested_image: bool = False, extra_label: bool = False) -> None:
    ds = root / name
    (ds / "images").mkdir(parents=True, exist_ok=True)
    if nested_image:
        sub = ds / "images" / "subfolder"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "img001.jpg").write_bytes(b"\x00")
        img_name = "subfolder/img001.jpg"
    else:
        (ds / "images" / "img001.jpg").write_bytes(b"\x00")
        img_name = "img001.jpg"
    labels_meta = "<label><name>bee</name></label>"
    boxes = '<box label="bee" xtl="10" ytl="10" xbr="40" ybr="30"/>'
    if extra_label:
        labels_meta += "<label><name>wasp</name></label><label><name>unused</name></label>"
        boxes += '<box label="wasp" xtl="50" ytl="10" xbr="70" ybr="30"/>'
    (ds / "annotations.xml").write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><labels>{labels_meta}</labels></task></meta>
  <image id="0" name="{img_name}" width="100" height="80">
    {boxes}
  </image>
</annotations>
""",
        encoding="utf-8",
    )


def test_datasets_json_writes_scan_summary_and_diff(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    sd = tmp_path / "raw_data"
    _flat_dataset(sd, "ds_a")

    datasets_json_main(["--workspace", str(tmp_path)])
    sum1 = tmp_path / "datasets" / DATASETS_SCAN_SUMMARY_FILE
    assert sum1.is_file()
    j1 = json.loads(sum1.read_text(encoding="utf-8"))
    assert j1["datasets"]["final"] == ["ds_a"]
    assert j1["datasets"]["added"] == ["ds_a"]
    assert j1["datasets"]["removed"] == []
    assert "bee" in j1["class_names"]["final"]
    passport_1 = tmp_path / "datasets" / "ds_a" / "dataset_passport.json"
    assert passport_1.is_file()
    p1 = json.loads(passport_1.read_text(encoding="utf-8"))
    assert p1["command"] == "scan"
    assert p1["parameters"]["kind"] == "initial"

    _flat_dataset(sd, "ds_b")
    (sd / "ds_b" / "data.yaml").write_text("nc: 1\nnames: ['antelope']\n", encoding="utf-8")

    datasets_json_main(["--workspace", str(tmp_path)])
    j2 = json.loads(sum1.read_text(encoding="utf-8"))
    assert set(j2["datasets"]["final"]) == {"ds_a", "ds_b"}
    assert j2["datasets"]["added"] == ["ds_b"]
    assert j2["datasets"]["removed"] == []
    assert "antelope" in j2["class_names"]["added"]
    assert (tmp_path / "datasets" / "ds_b" / "dataset_passport.json").is_file()

    import shutil

    shutil.rmtree(sd / "ds_b")
    datasets_json_main(["--workspace", str(tmp_path)])
    j3 = json.loads(sum1.read_text(encoding="utf-8"))
    # raw_data - update source only; deleting in raw_data does NOT remove datasets from datasets.
    assert set(j3["datasets"]["final"]) == {"ds_a", "ds_b"}
    assert j3["datasets"]["removed"] == []


def test_scan_skips_duplicate_content_with_other_name(tmp_path: Path, capsys) -> None:
    deploy_workspace(tmp_path)
    rd = tmp_path / "raw_data"
    _flat_dataset(rd, "ds_a")
    _flat_dataset(rd, "ds_b")
    # Make ds_b fully identical to ds_a content.
    (rd / "ds_b" / "data.yaml").write_text("nc: 1\nnames: ['bee']\n", encoding="utf-8")

    datasets_json_main(["--workspace", str(tmp_path)])
    out = capsys.readouterr().out
    info = json.loads((tmp_path / "datasets" / DATASETS_INFO_FILE).read_text(encoding="utf-8"))
    assert len(info) == 1
    assert set(info.keys()) in ({"ds_a"}, {"ds_b"})
    assert "matches datasets" in out


def test_scan_marks_modified_and_stops_sync_for_dataset(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    rd = tmp_path / "raw_data"
    _flat_dataset(rd, "ds_a")

    datasets_json_main(["--workspace", str(tmp_path)])
    info_path = tmp_path / "datasets" / DATASETS_INFO_FILE
    info1 = json.loads(info_path.read_text(encoding="utf-8"))
    initial_hash = info1["ds_a"]["dataset_hash"]
    assert info1["ds_a"]["modified"] is False

    # Manual change in datasets should mark modified=true.
    ds_label = tmp_path / "datasets" / "ds_a" / "labels" / "a.txt"
    ds_label.write_text("0 0.4 0.4 0.25 0.25\n", encoding="utf-8")
    datasets_json_main(["--workspace", str(tmp_path)])
    info2 = json.loads(info_path.read_text(encoding="utf-8"))
    assert info2["ds_a"]["modified"] is True
    assert info2["ds_a"]["dataset_hash"] != initial_hash

    # Change source in raw_data, but sync must be blocked by modified=true.
    raw_label = rd / "ds_a" / "labels" / "a.txt"
    raw_label.write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    datasets_json_main(["--workspace", str(tmp_path)])
    info3 = json.loads(info_path.read_text(encoding="utf-8"))
    assert info3["ds_a"]["modified"] is True
    assert info3["ds_a"]["dataset_hash"] == info2["ds_a"]["dataset_hash"]


def test_scan_converts_cvat11_to_training_ready_layout(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    rd = tmp_path / "raw_data"
    _cvat11_dataset(rd, "cvat_src")

    datasets_json_main(["--workspace", str(tmp_path)])
    out_root = tmp_path / "datasets" / "cvat_src"
    assert (out_root / "data.yaml").is_file()
    assert (out_root / "labels" / "img001.txt").is_file()
    yaml_text = (out_root / "data.yaml").read_text(encoding="utf-8")
    assert "train: images" in yaml_text
    cfg = yaml.safe_load(yaml_text)
    assert "path" not in cfg
    assert "names:" in yaml_text


def test_scan_recreates_deleted_dataset_even_if_duplicate_hash_elsewhere(
    tmp_path: Path,
) -> None:
    """Deleting datasets/<name> must not be blocked by dedupe vs another folder with same content."""
    deploy_workspace(tmp_path)
    rd = tmp_path / "raw_data"
    _cvat11_dataset(rd, "cvat_src")

    datasets_json_main(["--workspace", str(tmp_path)])
    src_ds = tmp_path / "datasets" / "cvat_src"
    assert src_ds.is_dir()

    shutil.copytree(src_ds, tmp_path / "datasets" / "cvat_shadow")
    shutil.rmtree(src_ds)

    datasets_json_main(["--workspace", str(tmp_path)])
    assert (tmp_path / "datasets" / "cvat_src").is_dir()
    assert (tmp_path / "datasets" / "cvat_src" / "labels" / "img001.txt").is_file()


def test_scan_cvat11_nested_image_mirrors_labels_under_subfolder(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    rd = tmp_path / "raw_data"
    _cvat11_dataset(rd, "cvat_nested", nested_image=True)

    datasets_json_main(["--workspace", str(tmp_path)])
    out_root = tmp_path / "datasets" / "cvat_nested"
    assert (out_root / "images" / "subfolder" / "img001.jpg").is_file()
    assert (out_root / "labels" / "subfolder" / "img001.txt").is_file()
    yaml_text = (out_root / "data.yaml").read_text(encoding="utf-8")
    assert "train: images" in yaml_text


def test_scan_purge_processed_raw_yes_deletes_processed_source(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    deploy_workspace(tmp_path)
    rd = tmp_path / "raw_data"
    _flat_dataset(rd, "ds_a")

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    datasets_json_main(["--workspace", str(tmp_path), "--purge-processed-raw"])
    out = capsys.readouterr().out
    assert "Requested to remove processed sources" in out
    assert not (rd / "ds_a").exists()
    assert (tmp_path / "datasets" / "ds_a").is_dir()


def test_scan_purge_processed_raw_no_keeps_sources(tmp_path: Path, monkeypatch, capsys) -> None:
    deploy_workspace(tmp_path)
    rd = tmp_path / "raw_data"
    _flat_dataset(rd, "ds_a")

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    datasets_json_main(["--workspace", str(tmp_path), "--purge-processed-raw"])
    out = capsys.readouterr().out
    assert "removal of processed sources from raw_data cancelled".lower() in out.lower()
    assert (rd / "ds_a").is_dir()


def test_scan_marks_provider_record_stale_when_paths_missing(tmp_path: Path, monkeypatch) -> None:
    deploy_workspace(tmp_path)
    _flat_dataset(tmp_path / "raw_data", "ds_a")
    cfg_root = tmp_path / "cfg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_root))
    upsert_provider_record(
        {
            "provider_id": "dr-yolo",
            "display_name": "DR-YOLO",
            "repo_path": str(tmp_path / "missing_repo"),
            "venv_path": str(tmp_path / "missing_venv"),
            "install_root": str(tmp_path),
            "install_state": "installed",
            "detected_capabilities": {"train": True, "infer": True},
            "repo_ref": {"remote_url": "https://example.invalid", "branch": "main", "commit": "abc"},
            "installed_at": "2026-01-01T00:00:00+00:00",
            "last_validated_at": "2026-01-01T00:00:00+00:00",
            "last_error": None,
        }
    )
    datasets_json_main(["--workspace", str(tmp_path)])
    recs = list_provider_records()
    rec = next(r for r in recs if r.get("provider_id") == "dr-yolo")
    assert rec["install_state"] == "stale"
    assert "missing repo_path" in str(rec.get("last_error", ""))


def test_scan_strip_unused_classes_yolo_flat(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    rd = tmp_path / "raw_data"
    _flat_dataset(rd, "ds_strip", extra_classes=True)

    datasets_json_main(["--workspace", str(tmp_path), "--strip-unused-classes"])
    cfg = yaml.safe_load((tmp_path / "datasets" / "ds_strip" / "data.yaml").read_text(encoding="utf-8"))
    assert cfg["names"] == ["bee", "wasp"]
    info = json.loads((tmp_path / "datasets" / DATASETS_INFO_FILE).read_text(encoding="utf-8"))
    assert info["ds_strip"]["classes"] == {"bee": 0, "wasp": 1}
    assert info["ds_strip"]["modified"] is True


def test_scan_strip_unused_classes_cvat11_after_conversion(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    rd = tmp_path / "raw_data"
    _cvat11_dataset(rd, "cvat_strip", extra_label=True)

    datasets_json_main(["--workspace", str(tmp_path), "--strip-unused-classes"])
    cfg = yaml.safe_load((tmp_path / "datasets" / "cvat_strip" / "data.yaml").read_text(encoding="utf-8"))
    assert cfg["names"] == ["bee", "wasp"]
    info = json.loads((tmp_path / "datasets" / DATASETS_INFO_FILE).read_text(encoding="utf-8"))
    assert set(info["cvat_strip"]["classes"].keys()) == {"bee", "wasp"}


def test_scan_strip_unused_classes_skips_existing(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    rd = tmp_path / "raw_data"
    _flat_dataset(rd, "ds_a", extra_classes=True)

    datasets_json_main(["--workspace", str(tmp_path)])
    before = (tmp_path / "datasets" / "ds_a" / "data.yaml").read_text(encoding="utf-8")

    datasets_json_main(["--workspace", str(tmp_path), "--strip-unused-classes"])
    after = (tmp_path / "datasets" / "ds_a" / "data.yaml").read_text(encoding="utf-8")
    assert before == after


def test_scan_strip_unused_classes_default_off(tmp_path: Path) -> None:
    deploy_workspace(tmp_path)
    rd = tmp_path / "raw_data"
    _flat_dataset(rd, "ds_off", extra_classes=True)

    datasets_json_main(["--workspace", str(tmp_path)])
    cfg = yaml.safe_load((tmp_path / "datasets" / "ds_off" / "data.yaml").read_text(encoding="utf-8"))
    assert cfg["names"] == ["bee", "wasp", "unused"]
