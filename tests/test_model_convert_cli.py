from __future__ import annotations

from pathlib import Path

import smartrain.model_convert_cli as mcc


def test_discover_models_lists_only_pt(tmp_path: Path):
    models_dir = tmp_path / "models"
    runs_dir = tmp_path / "runs"
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "a.pt").write_text("pt", encoding="utf-8")
    (models_dir / "a.onnx").write_text("onnx", encoding="utf-8")
    (runs_dir / "b.pt").write_text("pt", encoding="utf-8")
    (runs_dir / "b.onnx").write_text("onnx", encoding="utf-8")

    discovered = mcc._discover_models(tmp_path)
    paths = [p for _src, p in discovered]
    assert models_dir / "a.pt" in paths
    assert runs_dir / "b.pt" in paths
    assert models_dir / "a.onnx" not in paths
    assert runs_dir / "b.onnx" not in paths


def _base_args(**overrides):
    import argparse

    ns = argparse.Namespace(
        output_dir=None,
        format="tensorrt-trt",
        batch=1,
        dynamic=False,
        precision="fp32",
        imgsz=None,
        opset=17,
        simplify=True,
        half=False,
        nms=False,
        workspace_gib=None,
        device=None,
        data=None,
        fraction=1.0,
        force=False,
        continue_on_error=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_convert_pt_to_trt_reuses_existing_onnx(monkeypatch, tmp_path: Path):
    pt = tmp_path / "m.pt"
    onnx = tmp_path / "m.onnx"
    pt.write_text("pt", encoding="utf-8")
    onnx.write_text("onnx", encoding="utf-8")

    used_onnx: list[Path] = []
    monkeypatch.setattr(mcc, "_resolve_imgsz_from_args_and_model", lambda _a, _p: (640, "cli"))
    monkeypatch.setattr(mcc, "_check_tensorrt_ready", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_check_trtexec_gpu_ready", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_check_trtexec_runtime_ready", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_extract_onnx_signature", lambda _p: {"opset": 17, "batch": 1, "h": 640, "w": 640, "dynamic": False, "half": False, "simplify": True, "nms": False})
    monkeypatch.setattr(mcc, "_export_named_onnx_from_pt", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not build dedicated onnx")))

    def _fake_trtexec(onnx_path, engine_target, _args, _imgsz):
        used_onnx.append(Path(onnx_path))
        engine_target.write_text("engine", encoding="utf-8")
        return True, "ok"

    monkeypatch.setattr(mcc, "_trtexec_export_from_onnx", _fake_trtexec)

    ok_any, failed_any, _skipped_any, artifacts_ok, artifacts_failed, _artifacts_skipped = mcc._convert_one(pt, _base_args())
    assert ok_any is True
    assert failed_any is False
    assert artifacts_failed == 0
    assert artifacts_ok >= 1
    assert used_onnx == [onnx]
    assert (tmp_path / "m.trt").exists()


def test_convert_pt_to_trt_builds_dedicated_onnx_on_mismatch(monkeypatch, tmp_path: Path):
    pt = tmp_path / "m.pt"
    onnx = tmp_path / "m.onnx"
    pt.write_text("pt", encoding="utf-8")
    onnx.write_text("onnx", encoding="utf-8")

    monkeypatch.setattr(mcc, "_resolve_imgsz_from_args_and_model", lambda _a, _p: (1280, "cli"))
    monkeypatch.setattr(mcc, "_check_tensorrt_ready", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_check_trtexec_gpu_ready", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_check_trtexec_runtime_ready", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_extract_onnx_signature", lambda _p: {"opset": 17, "batch": 1, "h": 640, "w": 640, "dynamic": False, "half": False, "simplify": True, "nms": False})

    class _FakeYOLO:
        def __init__(self, _path: str):
            self.path = _path

        def export(self, **_kwargs):
            exported = tmp_path / "tmp_export.onnx"
            exported.write_text("new_onnx", encoding="utf-8")
            return str(exported)

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    built_onnx: list[Path] = []

    def _fake_trtexec(onnx_path, engine_target, _args, _imgsz):
        built_onnx.append(Path(onnx_path))
        engine_target.write_text("engine", encoding="utf-8")
        return True, "ok"

    monkeypatch.setattr(mcc, "_trtexec_export_from_onnx", _fake_trtexec)
    monkeypatch.setattr(mcc, "_validate_onnx_export", lambda _p: (True, "ok"))

    ok_any, failed_any, _skipped_any, artifacts_ok, artifacts_failed, _artifacts_skipped = mcc._convert_one(pt, _base_args())
    assert ok_any is True
    assert failed_any is False
    assert artifacts_failed == 0
    assert artifacts_ok >= 2  # dedicated ONNX + TRT
    assert built_onnx, "trtexec should receive dedicated onnx path"
    assert built_onnx[0].name.startswith("m_imgsz1280x1280_b1_static_op17_fp32_simplify1_nms0_trtprep")
    assert onnx.read_text(encoding="utf-8") == "onnx"
    assert (tmp_path / "m.trt").exists()


def test_parser_rejects_legacy_tensorrt_value():
    parser = mcc.build_model_convert_arg_parser()
    try:
        parser.parse_args(["--input", "x.pt", "--format", "tensorrt"])
        assert False, "legacy format must be rejected"
    except SystemExit as exc:
        assert int(exc.code) != 0


def test_get_export_format_availability_marks_trt_unavailable(monkeypatch):
    monkeypatch.setattr(mcc, "_check_tensorrt_ready", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_check_trtexec_dependencies", lambda: (False, "missing deps"))
    monkeypatch.setattr(mcc, "_check_trtexec_gpu_ready", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_check_trtexec_runtime_ready", lambda: (True, ""))
    availability = mcc._get_export_format_availability()
    assert availability["onnx"][0] is True
    assert availability["tensorrt-engine"][0] is True
    assert availability["tensorrt-trt"] == (False, "missing deps")


def test_get_export_format_availability_marks_trt_unavailable_on_gpu_probe(monkeypatch):
    monkeypatch.setattr(mcc, "_check_tensorrt_ready", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_check_trtexec_dependencies", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_check_trtexec_gpu_ready", lambda: (False, "gpu missing"))
    monkeypatch.setattr(mcc, "_check_trtexec_runtime_ready", lambda: (True, ""))
    availability = mcc._get_export_format_availability()
    assert availability["onnx"][0] is True
    assert availability["tensorrt-engine"][0] is True
    assert availability["tensorrt-trt"] == (False, "gpu missing")


def test_get_export_format_availability_marks_trt_unavailable_on_runtime_probe(monkeypatch):
    monkeypatch.setattr(mcc, "_check_tensorrt_ready", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_check_trtexec_dependencies", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_check_trtexec_gpu_ready", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_check_trtexec_runtime_ready", lambda: (False, "runtime fail"))
    availability = mcc._get_export_format_availability()
    assert availability["onnx"][0] is True
    assert availability["tensorrt-engine"][0] is True
    assert availability["tensorrt-trt"] == (False, "runtime fail")


def test_validate_args_rejects_unavailable_format(monkeypatch):
    parser = mcc.build_model_convert_arg_parser()
    args = _base_args(format="tensorrt-trt", input="x.pt")
    monkeypatch.setattr(
        mcc,
        "_get_export_format_availability",
        lambda: {"onnx": (True, ""), "tensorrt-engine": (True, ""), "tensorrt-trt": (False, "missing deps")},
    )
    try:
        mcc._validate_args(args, interactive_allowed=False, parser=parser, argv=["--input", "x.pt", "--format", "tensorrt-trt"])
        assert False, "unavailable format must be rejected"
    except SystemExit as exc:
        assert int(exc.code) != 0


def test_interactive_shows_unavailable_and_reprompts_format(monkeypatch, tmp_path: Path):
    import argparse

    args = argparse.Namespace(
        input=None,
        format=None,
        batch=1,
        dynamic=False,
        precision="fp32",
        imgsz=None,
        opset=17,
        simplify=True,
        half=False,
        nms=False,
        output_dir=None,
    )
    monkeypatch.setattr(
        mcc,
        "_discover_models",
        lambda _root: [("models", tmp_path / "m.pt")],
    )
    monkeypatch.setattr(mcc, "_collect_input_models", lambda _p: [tmp_path / "m.pt"])
    monkeypatch.setattr(mcc, "_resolve_imgsz_from_args_and_model", lambda _a, _p: (640, "cli"))
    monkeypatch.setattr(
        mcc,
        "_get_export_format_availability",
        lambda: {
            "onnx": (True, ""),
            "tensorrt-engine": (True, ""),
            "tensorrt-trt": (False, "runtime fail"),
        },
    )
    choices = iter([
        "models: m.pt",
        "tensorrt-trt (unavailable: runtime fail)",
        "onnx",
        "static",
        "auto",
    ])
    monkeypatch.setattr(mcc, "prompt_choice", lambda *a, **k: next(choices))
    monkeypatch.setattr(mcc, "prompt_int", lambda *a, **k: 1)
    monkeypatch.setattr(mcc, "prompt_text", lambda *a, **k: "")
    monkeypatch.setattr(mcc, "prompt_yes_no", lambda *a, **k: False)

    mcc._interactive_fill(args, tmp_path)
    assert args.format == "onnx"
