from __future__ import annotations

from pathlib import Path

import pytest

from smartrain.external_providers import runner


def test_run_external_train_invokes_python_script(monkeypatch, tmp_path: Path) -> None:
    train_py = tmp_path / "train.py"
    train_py.write_text("print('ok')\n", encoding="utf-8")
    calls: list[tuple[list[str], str]] = []

    def _fake_run(cmd, cwd=None, env=None):
        calls.append((list(cmd), str(cwd)))
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)
    monkeypatch.setattr(runner, "external_python_in_env", lambda _v: "/tmp/venv/bin/python")
    rc = runner.run_external_train(
        "dr-yolo",
        str(tmp_path),
        "/tmp/venv",
        dataset_path=str(tmp_path),
        model="yolov8n.pt",
        epochs=1,
        batch=1,
        imgsz=640,
        device="cpu",
        target_dir=str(tmp_path / "runs"),
    )
    assert rc == 0
    assert calls
    assert calls[0][0][0] == "/tmp/venv/bin/python"
    assert calls[0][0][1].endswith("train.py")


@pytest.mark.parametrize(
    "provider_id,expected_script",
    [
        ("dr-yolo", "train.py"),
        ("leaf-yolo", "train.py"),
        ("mfel-yolo", "mfel_train_launcher.py"),
        ("mp-yolo", "mp_train_launcher.py"),
        ("ssdm-yolo", "train.py"),
        ("enhanced-yolov8", "train.py"),
    ],
)
def test_run_external_train_supported_providers(monkeypatch, tmp_path: Path, provider_id: str, expected_script: str) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_run(cmd, cwd=None, env=None):
        calls.append((list(cmd), str(cwd)))

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)
    monkeypatch.setattr(runner, "external_python_in_env", lambda _v: "/tmp/venv/bin/python")
    rc = runner.run_external_train(
        provider_id,
        str(tmp_path),
        "/tmp/venv",
        dataset_path=str(tmp_path),
        model="yolov8n.pt",
        epochs=1,
        batch=1,
        imgsz=640,
        device="cpu",
        target_dir=str(tmp_path / "runs"),
    )
    assert rc == 0
    assert calls
    assert calls[0][0][1].endswith(expected_script)


@pytest.mark.parametrize(
    "provider_id,expected_script",
    [
        ("dr-yolo", "detect.py"),
        ("leaf-yolo", "detect.py"),
        ("mfel-yolo", "mfel_infer_launcher.py"),
        ("mp-yolo", "mp_infer_launcher.py"),
        ("ssdm-yolo", "detect.py"),
        ("enhanced-yolov8", "detect.py"),
    ],
)
def test_run_external_infer_supported_providers(monkeypatch, tmp_path: Path, provider_id: str, expected_script: str) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_run(cmd, cwd=None, env=None):
        calls.append((list(cmd), str(cwd)))

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)
    monkeypatch.setattr(runner, "external_python_in_env", lambda _v: "/tmp/venv/bin/python")
    rc = runner.run_external_infer(
        provider_id,
        str(tmp_path),
        "/tmp/venv",
        model_path="yolov8n.pt",
        source_path=str(tmp_path / "images"),
        conf=0.25,
        imgsz=640,
        device="cpu",
    )
    assert rc == 0
    assert calls
    assert calls[0][0][1].endswith(expected_script)

