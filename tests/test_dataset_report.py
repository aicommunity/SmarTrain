from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def tiny_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ds = ws / "datasets" / "tiny"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    for i in range(6):
        img = Image.new("RGB", (128, 128), color=(20 + i * 30, 40, 60 + i * 10))
        img.save(ds / "images" / f"im{i}.jpg")
    for i in range(6):
        cls = i % 2
        cx, cy = 0.45 + (i % 3) * 0.05, 0.5
        (ds / "labels" / f"im{i}.txt").write_text(f"{cls} {cx} {cy} 0.2 0.2\n", encoding="utf-8")
    (ds / "data.yaml").write_text(
        "train: images\nval: images\ntest: images\nnc: 2\nnames: [cat, dog]\n",
        encoding="utf-8",
    )
    catalog = {
        "tiny": {
            "classes": {"cat": 0, "dog": 1},
            "structure": "flat",
            "elements_count": 6,
        }
    }
    (ws / "datasets").mkdir(parents=True, exist_ok=True)
    with open(ws / "datasets" / "datasets_info.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    (ws / "analytics").mkdir(parents=True, exist_ok=True)
    (ws / "tmp").mkdir(parents=True, exist_ok=True)
    return ws


def test_dataset_report_markdown_and_assets(tiny_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMART_TRAIN_WORKSPACE", str(tiny_workspace))
    out = tiny_workspace / "report_out"
    from smartrain.dataset_report import main

    main(
        [
            "--workspace",
            str(tiny_workspace),
            "--dataset",
            "tiny",
            "--output-dir",
            str(out),
            "--examples-per-class",
            "3",
            "--languages",
            "en,ru",
            "--seed",
            "0",
            "--no-pdf",
            "--no-odt",
        ]
    )
    assert (out / "en" / "index.md").is_file()
    assert (out / "ru" / "index.md").is_file()
    assert (out / "assets").is_dir()
    cat_dir = out / "assets" / "cat"
    dog_dir = out / "assets" / "dog"
    assert cat_dir.is_dir() and dog_dir.is_dir()
    assert len(list(cat_dir.glob("*.png"))) <= 3
    assert len(list(dog_dir.glob("*.png"))) <= 3
    en = (out / "en" / "index.md").read_text(encoding="utf-8")
    assert "cat" in en and "dog" in en


def test_dataset_report_default_output_under_analytics(tiny_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMART_TRAIN_WORKSPACE", str(tiny_workspace))
    from smartrain.dataset_report import main

    main(
        [
            "--workspace",
            str(tiny_workspace),
            "--dataset",
            "tiny",
            "--examples-per-class",
            "2",
            "--languages",
            "en",
            "--no-pdf",
            "--no-odt",
            "--seed",
            "1",
        ]
    )
    dr = tiny_workspace / "analytics" / "datasets-reports"
    assert dr.is_dir()
    subdirs = [p for p in dr.iterdir() if p.is_dir()]
    assert len(subdirs) >= 1
    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
    assert (latest / "en" / "index.md").is_file()


def test_report_odt_builtin_and_pdf_fpdf2_without_pandoc(
    tiny_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fpdf")
    monkeypatch.setenv("SMART_TRAIN_WORKSPACE", str(tiny_workspace))
    out = tiny_workspace / "report_odt_pdf"
    monkeypatch.setattr("smartrain.dataset_report._try_pandoc_odt", lambda *a, **k: False)
    monkeypatch.setattr("smartrain.dataset_report._try_pandoc_pdf", lambda *a, **k: False)
    from smartrain.dataset_report import main

    main(
        [
            "--workspace",
            str(tiny_workspace),
            "--dataset",
            "tiny",
            "--output-dir",
            str(out),
            "--examples-per-class",
            "2",
            "--languages",
            "en",
            "--seed",
            "0",
        ]
    )
    assert (out / "report-en.odt").is_file()
    assert (out / "report-en.pdf").is_file()


def test_dataset_report_requires_dataset_non_interactive(
    tiny_workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SMART_TRAIN_WORKSPACE", str(tiny_workspace))
    monkeypatch.setenv("SMART_TRAIN_INTERACTIVE_ALLOWED", "0")
    from smartrain.dataset_report import main

    main(["--workspace", str(tiny_workspace)])
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "Incomplete arguments" in out or "[ERROR]" in out
