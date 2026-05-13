from __future__ import annotations

import builtins
import sys
from types import ModuleType
from pathlib import Path

from smartrain.external_providers.launchers import mfel_infer_launcher, mfel_train_launcher, mp_infer_launcher


def test_mfel_train_launcher_reports_missing_dcnv4(monkeypatch, tmp_path: Path, capsys) -> None:
    repo = tmp_path / "mfel"
    repo.mkdir(parents=True, exist_ok=True)

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ultralytics":
            raise ModuleNotFoundError("No module named 'DCNv4'", name="DCNv4")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    rc = mfel_train_launcher.main(
        ["--repo", str(repo), "--data", str(repo / "data.yaml"), "--model", "yolov8n.pt"]
    )
    assert rc == 2
    assert "DCNv4" in capsys.readouterr().err


def test_mfel_infer_launcher_reports_missing_dcnv4(monkeypatch, tmp_path: Path, capsys) -> None:
    repo = tmp_path / "mfel"
    repo.mkdir(parents=True, exist_ok=True)

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ultralytics":
            raise ModuleNotFoundError("No module named 'DCNv4'", name="DCNv4")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    rc = mfel_infer_launcher.main(
        ["--repo", str(repo), "--model", "yolov8n.pt", "--source", str(repo / "images")]
    )
    assert rc == 2
    assert "DCNv4" in capsys.readouterr().err


def test_mfel_train_launcher_resolves_custom_model_aliases(tmp_path: Path) -> None:
    repo = tmp_path / "mfel"
    cfg_dir = repo / "ultralytics" / "cfg" / "MFEL-YOLO"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "MFEL-YOLO.yaml").write_text("nc: 1\n", encoding="utf-8")
    (cfg_dir / "E_PAN+.yaml").write_text("nc: 1\n", encoding="utf-8")

    resolved_main = mfel_train_launcher._resolve_mfel_model_spec(repo, "mfel-yolo")
    resolved_epan = mfel_train_launcher._resolve_mfel_model_spec(repo, "e_pan+")
    assert resolved_main.endswith("MFEL-YOLO.yaml")
    assert resolved_epan.endswith("E_PAN+.yaml")


def test_mfel_infer_launcher_device_cpu_upgrades_to_zero_when_cuda_available(monkeypatch) -> None:
    fake_torch = ModuleType("torch")

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    fake_torch.cuda = _FakeCuda()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    resolved = mfel_infer_launcher._resolve_mfel_predict_device("cpu")
    assert resolved == "0"


def test_mfel_infer_launcher_device_cpu_kept_without_cuda(monkeypatch) -> None:
    fake_torch = ModuleType("torch")

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch.cuda = _FakeCuda()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    resolved = mfel_infer_launcher._resolve_mfel_predict_device("cpu")
    assert resolved == "cpu"


def test_mfel_infer_structured_outputs_classification_top1() -> None:
    probs = type(
        "P",
        (),
        {"top1": 3, "top1conf": 0.88, "top5": [3, 1], "top5conf": [0.88, 0.1]},
    )()
    pred = type("R", (), {"probs": probs})()
    out = mfel_infer_launcher._extract_task_outputs([pred], "classification")
    cls = out.get("classification")
    assert isinstance(cls, dict)
    assert cls["top1"]["class_index"] == 3


def test_mp_infer_structured_outputs_segmentation_polygon() -> None:
    tensor_box = type("T", (), {"cpu": lambda self: self, "numpy": lambda self: [[1.0, 2.0, 5.0, 6.0]]})()
    tensor_cls = type("T", (), {"cpu": lambda self: self, "numpy": lambda self: [0.0]})()
    tensor_conf = type("T", (), {"cpu": lambda self: self, "numpy": lambda self: [0.9]})()
    boxes = type("B", (), {"xyxy": tensor_box, "cls": tensor_cls, "conf": tensor_conf, "__len__": lambda self: 1})()
    masks = type("M", (), {"xy": [[[1.0, 2.0], [2.0, 3.0], [5.0, 6.0]]]})()
    pred = type("R", (), {"boxes": boxes, "masks": masks})()
    out = mp_infer_launcher._extract_task_outputs([pred], "segmentation")
    segs = out.get("segments")
    assert isinstance(segs, list) and len(segs) == 1
    assert segs[0]["polygon_roi_xy"][0] == [1.0, 2.0]
