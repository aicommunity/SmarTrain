from __future__ import annotations

import sys
import types
from pathlib import Path

import smartrain.workflows.models.model_convert_cli as mcc


def test_discover_models_lists_only_pt(tmp_path: Path):
    models_dir = tmp_path / "models"
    runs_dir = tmp_path / "runs"
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "a.pt").write_text("pt", encoding="utf-8")
    (models_dir / "a.onnx").write_text("onnx", encoding="utf-8")
    run_dir = runs_dir / "ds1" / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")
    (run_dir / "run-1.pt").write_text("pt", encoding="utf-8")
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_text("legacy-pt", encoding="utf-8")

    discovered = mcc._discover_models(tmp_path, allowed_suffixes=(".pt",))
    paths = [p for _src, p in discovered]
    assert models_dir / "a.pt" in paths
    assert run_dir / "models" / "run-1.pt" in paths
    assert models_dir / "a.onnx" not in paths
    assert run_dir / "train" / "weights" / "best.pt" not in paths


def test_discover_models_lists_only_onnx(tmp_path: Path):
    models_dir = tmp_path / "models"
    runs_dir = tmp_path / "runs"
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "a.pt").write_text("pt", encoding="utf-8")
    (models_dir / "a.onnx").write_text("onnx", encoding="utf-8")
    run_dir = runs_dir / "ds1" / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")
    (run_dir / "run-1.onnx").write_text("onnx", encoding="utf-8")
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.onnx").write_text("legacy-onnx", encoding="utf-8")

    discovered = mcc._discover_models(tmp_path, allowed_suffixes=(".onnx",))
    paths = [p for _src, p in discovered]
    assert models_dir / "a.onnx" in paths
    assert run_dir / "models" / "run-1.onnx" in paths
    assert models_dir / "a.pt" not in paths
    assert run_dir / "train" / "weights" / "best.onnx" not in paths


def test_discover_models_materializes_canonical_from_legacy_run(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "ds1" / "run-1"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text(
        '{"paths":{"best_model":"train/weights/best.pt"}}',
        encoding="utf-8",
    )
    legacy_best = run_dir / "train" / "weights" / "best.pt"
    legacy_best.write_text("legacy-pt", encoding="utf-8")

    discovered = mcc._discover_models(tmp_path, allowed_suffixes=(".pt",))
    paths = [p for _src, p in discovered]
    canonical = run_dir / "models" / "run-1.pt"
    assert canonical in paths
    assert canonical.is_file()
    assert not legacy_best.exists()


def test_resolve_imgsz_from_run_models_uses_training_metadata(tmp_path: Path):
    run_dir = tmp_path / "runs" / "ds1" / "run-1"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text(
        '{"training_info":{"hyperparameters":{"image_size":1280}}}',
        encoding="utf-8",
    )
    pt = models_dir / "run-1.pt"
    pt.write_text("pt", encoding="utf-8")
    value, source = mcc._resolve_imgsz_from_args_and_model(_base_args(imgsz=None), pt)
    assert value == 1280
    assert source == "training_metadata"


def test_resolve_public_onnx_target_uses_variant_name_without_train_dir(tmp_path: Path):
    run_dir = tmp_path / "runs" / "ds1" / "run-no-train"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")
    pt = models_dir / "run-no-train.pt"
    pt.write_text("pt", encoding="utf-8")

    expected = {
        "h": 640,
        "w": 640,
        "batch": 1,
        "dynamic": False,
        "opset": 17,
        "half": False,
        "simplify": True,
        "nms": False,
    }
    target = mcc._resolve_public_onnx_target(pt, models_dir, expected)
    assert target.name == "run-no-train_imgsz640x640_b1_static_op17_fp32_simplify1_nms0.onnx"


def test_resolve_public_onnx_target_uses_variant_name_for_runs_models_without_metadata(tmp_path: Path):
    run_dir = tmp_path / "runs" / "ds1" / "run-no-meta"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    pt = models_dir / "run-no-meta.pt"
    pt.write_text("pt", encoding="utf-8")
    expected = {
        "h": 640,
        "w": 640,
        "batch": 1,
        "dynamic": False,
        "opset": 17,
        "half": False,
        "simplify": True,
        "nms": False,
    }
    target = mcc._resolve_public_onnx_target(pt, models_dir, expected)
    assert target.name == "run-no-meta_imgsz640x640_b1_static_op17_fp32_simplify1_nms0.onnx"


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
    assert len(used_onnx) == 1
    assert used_onnx[0].name.endswith("_trtprep.onnx")
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
    assert onnx.read_text(encoding="utf-8") == "new_onnx"
    assert (tmp_path / "m.trt").exists()


def test_cleanup_trtprep_artifacts_removes_stale_variants(tmp_path: Path):
    out_dir = tmp_path
    keep_unrelated = out_dir / "m_imgsz640x640_b1_static_op17_fp32_simplify1_nms0.onnx"
    stale_a = out_dir / "m_imgsz640x640_b1_static_op17_fp32_simplify1_nms0_trtprep.onnx"
    stale_b = out_dir / "m_imgsz640x640_b1_static_op17_fp32_simplify1_nms0_trtprep_v2.onnx"
    other_model = out_dir / "n_imgsz640x640_b1_static_op17_fp32_simplify1_nms0_trtprep.onnx"
    keep_unrelated.write_text("x", encoding="utf-8")
    stale_a.write_text("x", encoding="utf-8")
    stale_b.write_text("x", encoding="utf-8")
    other_model.write_text("x", encoding="utf-8")

    mcc._cleanup_trtprep_artifacts(out_dir, "m")

    assert not stale_a.exists()
    assert not stale_b.exists()
    assert keep_unrelated.exists()
    assert other_model.exists()


def test_convert_trt_only_materializes_public_and_cache_onnx_for_run(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "runs" / "ds1" / "run-1"
    models_dir = run_dir / "models"
    train_dir = run_dir / "train"
    models_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")
    pt = models_dir / "run-1.pt"
    pt.write_text("pt", encoding="utf-8")

    monkeypatch.setattr(mcc, "_resolve_imgsz_from_args_and_model", lambda _a, _p: (1280, "cli"))
    monkeypatch.setattr(mcc, "_check_tensorrt_ready", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_check_trtexec_gpu_ready", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_check_trtexec_runtime_ready", lambda: (True, ""))
    monkeypatch.setattr(mcc, "_extract_onnx_signature", lambda _p: None)
    monkeypatch.setattr(mcc, "_validate_onnx_export", lambda _p: (True, "ok"))

    class _FakeYOLO:
        def __init__(self, _path: str):
            self.path = _path

        def export(self, **_kwargs):
            exported = tmp_path / "tmp_export_for_trt.onnx"
            exported.write_text("onnx", encoding="utf-8")
            return str(exported)

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    monkeypatch.setattr(
        mcc,
        "_trtexec_export_from_onnx",
        lambda _onnx, engine_target, _args, _imgsz: (engine_target.write_text("trt", encoding="utf-8") or True, "ok"),
    )

    ok_any, failed_any, _skipped_any, artifacts_ok, artifacts_failed, _artifacts_skipped = mcc._convert_one(
        pt, _base_args(format="tensorrt-trt")
    )
    assert ok_any is True
    assert failed_any is False
    assert artifacts_failed == 0
    assert artifacts_ok >= 2
    assert not (models_dir / "run-1.onnx").exists()
    assert (models_dir / "run-1_imgsz1280x1280_b1_static_op17_fp32_simplify1_nms0.onnx").exists()
    dedicated = sorted(models_dir.glob("*_trtprep.onnx"))
    assert dedicated, "dedicated trtprep ONNX must exist"
    assert not (models_dir / "run-1.trt").exists()
    assert (models_dir / "run-1_imgsz1280x1280_b1_static_fp32.trt").exists()


def test_convert_onnx_only_syncs_public_and_trtprep_for_run(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "runs" / "ds1" / "run-1"
    models_dir = run_dir / "models"
    train_dir = run_dir / "train"
    models_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")
    pt = models_dir / "run-1.pt"
    pt.write_text("pt", encoding="utf-8")

    monkeypatch.setattr(mcc, "_resolve_imgsz_from_args_and_model", lambda _a, _p: (1280, "cli"))
    monkeypatch.setattr(mcc, "_validate_onnx_export", lambda _p: (True, "ok"))

    class _FakeYOLO:
        def __init__(self, _path: str):
            self.path = _path

        def export(self, **_kwargs):
            exported = tmp_path / "tmp_export_for_onnx.onnx"
            exported.write_text("onnx", encoding="utf-8")
            return str(exported)

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)

    ok_any, failed_any, _skipped_any, artifacts_ok, artifacts_failed, _artifacts_skipped = mcc._convert_one(
        pt, _base_args(format="onnx")
    )
    assert ok_any is True
    assert failed_any is False
    assert artifacts_failed == 0
    assert artifacts_ok >= 1
    assert not (models_dir / "run-1.onnx").exists()
    assert (models_dir / "run-1_imgsz1280x1280_b1_static_op17_fp32_simplify1_nms0.onnx").exists()
    dedicated = sorted(models_dir.glob("*_trtprep.onnx"))
    assert dedicated, "dedicated trtprep ONNX must be synchronized in onnx-only mode"


def test_sync_onnx_artifact_removes_legacy_plain_onnx_for_run(tmp_path: Path):
    run_dir = tmp_path / "runs" / "ds1" / "run-cleanup"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    source_pt = models_dir / "run-cleanup.pt"
    source_pt.write_text("pt", encoding="utf-8")
    src_onnx = models_dir / "tmp_src.onnx"
    src_onnx.write_text("onnx", encoding="utf-8")
    plain = models_dir / "run-cleanup.onnx"
    plain.write_text("legacy", encoding="utf-8")
    (models_dir / "run-cleanup.onnx.meta.json").write_text("{}", encoding="utf-8")
    target = models_dir / "run-cleanup_imgsz1280x1280_b1_static_op17_fp32_simplify1_nms0.onnx"
    ok, reason = mcc._sync_onnx_artifact(
        src_onnx,
        target,
        force=True,
        source_path_for_meta=source_pt,
        run_root=run_dir,
        params={"imgsz": "1280", "batch": 1, "dynamic": False, "opset": 17, "simplify": True, "half": False, "nms": False},
    )
    assert ok is True, reason
    assert target.exists()
    assert not plain.exists()
    assert not (models_dir / "run-cleanup.onnx.meta.json").exists()


def test_interactive_onnx_plus_trt_exports_onnx_once(monkeypatch, tmp_path: Path):
    source_pt = tmp_path / "m.pt"
    source_pt.write_text("pt", encoding="utf-8")
    export_calls = {"onnx": 0}

    class _FakeYOLO:
        def __init__(self, _path: str):
            self.path = _path

        def export(self, **kwargs):
            if kwargs.get("format") == "onnx":
                export_calls["onnx"] += 1
                exported = tmp_path / "single_export.onnx"
                exported.write_text("onnx", encoding="utf-8")
                return str(exported)
            exported = tmp_path / "engine.engine"
            exported.write_text("engine", encoding="utf-8")
            return str(exported)

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    monkeypatch.setattr(mcc, "_validate_onnx_export", lambda _p: (True, "ok"))
    monkeypatch.setattr(
        mcc,
        "_trtexec_export_from_onnx",
        lambda _onnx, trt_target, _args, _imgsz: (trt_target.write_text("trt", encoding="utf-8") or True, "ok"),
    )

    ctx = mcc.InteractiveContext(
        source_kind="pt",
        source_path=source_pt,
        target_onnx=True,
        target_engine=False,
        target_trt=True,
        output_dir=tmp_path,
        force=True,
        force_onnx=True,
        force_engine=True,
        force_trt=True,
        onnx_imgsz=640,
        onnx_imgsz_source="cli",
        onnx_batch=1,
        onnx_dynamic=False,
        device=None,
        engine_precision="fp32",
        engine_workspace_gib=None,
        trt_precision="fp32",
        trt_workspace_gib=None,
        data=None,
        fraction=1.0,
        opset=17,
        simplify=True,
        half=False,
        nms=False,
    )
    result = mcc._run_interactive_pipeline(ctx)
    assert result.stats.failed == 0
    assert export_calls["onnx"] == 1
    assert (tmp_path / "m.onnx").exists()
    assert sorted(tmp_path.glob("*_trtprep.onnx"))


def test_interactive_onnx_engine_trt_exports_onnx_once(monkeypatch, tmp_path: Path):
    source_pt = tmp_path / "m.pt"
    source_pt.write_text("pt", encoding="utf-8")
    export_calls = {"onnx": 0, "engine": 0}

    class _FakeYOLO:
        def __init__(self, _path: str):
            self.path = _path

        def export(self, **kwargs):
            fmt = kwargs.get("format")
            if fmt == "onnx":
                export_calls["onnx"] += 1
                exported = tmp_path / "single_export.onnx"
                exported.write_text("onnx", encoding="utf-8")
                return str(exported)
            if fmt == "engine":
                export_calls["engine"] += 1
                exported = tmp_path / "single_export.engine"
                exported.write_text("engine", encoding="utf-8")
                return str(exported)
            raise AssertionError(f"unexpected format: {fmt}")

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    monkeypatch.setattr(mcc, "_validate_onnx_export", lambda _p: (True, "ok"))
    monkeypatch.setattr(
        mcc,
        "_trtexec_export_from_onnx",
        lambda _onnx, trt_target, _args, _imgsz: (trt_target.write_text("trt", encoding="utf-8") or True, "ok"),
    )

    ctx = mcc.InteractiveContext(
        source_kind="pt",
        source_path=source_pt,
        target_onnx=True,
        target_engine=True,
        target_trt=True,
        output_dir=tmp_path,
        force=True,
        force_onnx=True,
        force_engine=True,
        force_trt=True,
        onnx_imgsz=640,
        onnx_imgsz_source="cli",
        onnx_batch=1,
        onnx_dynamic=False,
        device=None,
        engine_precision="fp32",
        engine_workspace_gib=None,
        trt_precision="fp32",
        trt_workspace_gib=None,
        data=None,
        fraction=1.0,
        opset=17,
        simplify=True,
        half=False,
        nms=False,
    )
    result = mcc._run_interactive_pipeline(ctx)
    assert result.stats.failed == 0
    assert export_calls["onnx"] == 1
    assert export_calls["engine"] == 1
    assert (tmp_path / "m.onnx").exists()
    assert sorted(tmp_path.glob("*_trtprep.onnx"))
    assert (tmp_path / "m.engine").exists()
    assert (tmp_path / "m.trt").exists()


def test_interactive_onnx_skip_syncs_trtprep_for_trt(monkeypatch, tmp_path: Path):
    source_pt = tmp_path / "m.pt"
    source_pt.write_text("pt", encoding="utf-8")
    public = tmp_path / "m.onnx"
    public.write_text("public-onnx", encoding="utf-8")

    used_onnx: list[Path] = []
    monkeypatch.setattr(
        mcc,
        "_extract_onnx_signature",
        lambda _p: {"opset": 17, "batch": 1, "h": 640, "w": 640, "dynamic": False, "half": False, "simplify": True, "nms": False},
    )

    class _FakeYOLO:
        def __init__(self, _path: str):
            self.path = _path

        def export(self, **_kwargs):
            raise AssertionError("ONNX export must not run when public ONNX already exists")

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)

    def _fake_trtexec(onnx_path, engine_target, _args, _imgsz):
        used_onnx.append(Path(onnx_path))
        engine_target.write_text("trt", encoding="utf-8")
        return True, "ok"

    monkeypatch.setattr(mcc, "_trtexec_export_from_onnx", _fake_trtexec)

    ctx = mcc.InteractiveContext(
        source_kind="pt",
        source_path=source_pt,
        target_onnx=True,
        target_engine=False,
        target_trt=True,
        output_dir=tmp_path,
        force=False,
        force_onnx=False,
        force_engine=False,
        force_trt=True,
        onnx_imgsz=640,
        onnx_imgsz_source="cli",
        onnx_batch=1,
        onnx_dynamic=False,
        device=None,
        engine_precision="fp32",
        engine_workspace_gib=None,
        trt_precision="fp32",
        trt_workspace_gib=None,
        data=None,
        fraction=1.0,
        opset=17,
        simplify=True,
        half=False,
        nms=False,
    )
    result = mcc._run_interactive_pipeline(ctx)
    assert result.stats.failed == 0
    dedicated = sorted(tmp_path.glob("*_trtprep.onnx"))
    assert dedicated, "dedicated trtprep ONNX must be created from existing public ONNX"
    assert dedicated[0].read_text(encoding="utf-8") == "public-onnx"
    assert len(used_onnx) == 1
    assert used_onnx[0].name.endswith("_trtprep.onnx")
    assert (tmp_path / "m.trt").exists()


def test_interactive_onnx_decline_overwrite_skips_export(monkeypatch, tmp_path: Path):
    source_pt = tmp_path / "m.pt"
    source_pt.write_text("pt", encoding="utf-8")
    public = tmp_path / "m.onnx"
    dedicated = tmp_path / "m_imgsz640x640_b1_static_op17_fp32_simplify1_nms0_trtprep.onnx"
    public.write_text("public", encoding="utf-8")
    dedicated.write_text("cache", encoding="utf-8")

    class _FakeYOLO:
        def __init__(self, _path: str):
            self.path = _path

        def export(self, **_kwargs):
            raise AssertionError("ONNX export must not run when overwrite is declined")

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)

    ctx = mcc.InteractiveContext(
        source_kind="pt",
        source_path=source_pt,
        target_onnx=True,
        target_engine=False,
        target_trt=False,
        output_dir=tmp_path,
        force=False,
        force_onnx=False,
        force_engine=False,
        force_trt=False,
        onnx_imgsz=640,
        onnx_imgsz_source="cli",
        onnx_batch=1,
        onnx_dynamic=False,
        device=None,
        engine_precision="fp32",
        engine_workspace_gib=None,
        trt_precision="fp32",
        trt_workspace_gib=None,
        data=None,
        fraction=1.0,
        opset=17,
        simplify=True,
        half=False,
        nms=False,
    )
    result = mcc._run_interactive_pipeline(ctx)
    assert result.stats.failed == 0
    assert result.stats.artifacts_skipped == 1
    assert dedicated.read_text(encoding="utf-8") == "cache"


def test_interactive_trt_targets_existing_skip_without_onnx_export(monkeypatch, tmp_path: Path):
    source_pt = tmp_path / "m.pt"
    source_pt.write_text("pt", encoding="utf-8")
    (tmp_path / "m.engine").write_text("engine", encoding="utf-8")
    (tmp_path / "m.trt").write_text("trt", encoding="utf-8")

    class _FakeYOLO:
        def __init__(self, _path: str):
            self.path = _path

        def export(self, **_kwargs):
            raise AssertionError("ONNX cache export must not run when all TRT targets are skipped")

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)

    ctx = mcc.InteractiveContext(
        source_kind="pt",
        source_path=source_pt,
        target_onnx=False,
        target_engine=True,
        target_trt=True,
        output_dir=tmp_path,
        force=False,
        force_onnx=False,
        force_engine=False,
        force_trt=False,
        onnx_imgsz=640,
        onnx_imgsz_source="cli",
        onnx_batch=1,
        onnx_dynamic=False,
        device=None,
        engine_precision="fp32",
        engine_workspace_gib=None,
        trt_precision="fp32",
        trt_workspace_gib=None,
        data=None,
        fraction=1.0,
        opset=17,
        simplify=True,
        half=False,
        nms=False,
    )
    result = mcc._run_interactive_pipeline(ctx)
    assert result.stats.failed == 0
    assert result.stats.skipped == 2


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
    assert availability["tensorrt-trt"][0] is False
    assert "runtime fail" in availability["tensorrt-trt"][1]
    assert "format=tensorrt-engine" in availability["tensorrt-trt"][1]


def test_detect_trtexec_capabilities_prefers_mempoolsize(monkeypatch):
    def _fake_run(cmd, **_kwargs):
        assert cmd[-1] == "--help"
        return types.SimpleNamespace(
            returncode=0,
            stdout="--memPoolSize --buildOnly",
            stderr="",
        )

    monkeypatch.setattr(mcc.subprocess, "run", _fake_run)
    caps = mcc._detect_trtexec_capabilities("/usr/bin/trtexec")
    assert caps.supports_build_only is True
    assert caps.workspace_mode == "memPoolSize"


def test_runtime_probe_fallback_without_buildonly(monkeypatch, tmp_path: Path):
    class _FakeTensorProto:
        FLOAT = 1

    class _FakeHelper:
        @staticmethod
        def make_tensor_value_info(*_args, **_kwargs):
            return object()

        @staticmethod
        def make_node(*_args, **_kwargs):
            return object()

        @staticmethod
        def make_graph(*_args, **_kwargs):
            return object()

        @staticmethod
        def make_model(*_args, **_kwargs):
            return object()

        @staticmethod
        def make_operatorsetid(*_args, **_kwargs):
            return object()

    fake_onnx = types.SimpleNamespace(
        TensorProto=_FakeTensorProto,
        helper=_FakeHelper,
        save=lambda _model, path: Path(path).write_text("onnx", encoding="utf-8"),
    )
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    monkeypatch.setattr(mcc, "_resolve_trtexec_bin", lambda: "/usr/bin/trtexec")
    monkeypatch.setattr(
        mcc,
        "_get_trtexec_capabilities",
        lambda _bin: mcc.TrtexecCapabilities(
            supports_build_only=True,
            supports_explicit_batch=True,
            workspace_mode="workspace",
        ),
    )
    monkeypatch.setattr(mcc.trt_checks.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "probe_dir"))
    (tmp_path / "probe_dir").mkdir(parents=True, exist_ok=True)

    calls: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        save_engine_arg = [arg for arg in cmd if arg.startswith("--saveEngine=")][0]
        engine_path = Path(save_engine_arg.split("=", 1)[1])
        if "--buildOnly" in cmd:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="Unknown option: --buildOnly")
        engine_path.write_text("ok", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mcc.subprocess, "run", _fake_run)
    monkeypatch.setattr(mcc.shutil, "rmtree", lambda *_a, **_k: None)
    monkeypatch.setattr(mcc, "_TRTEXEC_RUNTIME_CACHE", None)

    ok, reason = mcc._check_trtexec_runtime_ready()
    assert ok is True
    assert reason == ""
    assert any("--buildOnly" in cmd for cmd in calls)
    assert any("--buildOnly" not in cmd for cmd in calls)


def test_trtexec_export_skips_explicit_batch_when_unsupported(monkeypatch, tmp_path: Path):
    onnx_path = tmp_path / "m.onnx"
    onnx_path.write_text("onnx", encoding="utf-8")
    engine_target = tmp_path / "m.trt"
    args = _base_args(workspace_gib=1.0, batch=1, dynamic=False, precision="fp32")

    monkeypatch.setattr(mcc, "_resolve_trtexec_bin", lambda: "/usr/bin/trtexec")
    monkeypatch.setattr(mcc, "_guess_onnx_input_name", lambda _p: "images")
    monkeypatch.setattr(
        mcc,
        "_get_trtexec_capabilities",
        lambda _bin: mcc.TrtexecCapabilities(
            supports_build_only=True,
            supports_explicit_batch=False,
            workspace_mode="memPoolSize",
        ),
    )
    seen_cmds: list[list[str]] = []

    class _FakePopen:
        def __init__(self, cmd, **_kwargs):
            seen_cmds.append(list(cmd))
            self.returncode = None
            self._done = False

        def poll(self):
            if not self._done:
                self._done = True
                return None
            self.returncode = 0
            engine_target.write_text("engine", encoding="utf-8")
            return 0

    def _fake_sleep(_secs):
        return None

    def _fake_mkstemp(**_kwargs):
        p = tmp_path / "trtexec.log"
        p.write_text("", encoding="utf-8")
        return (123, str(p))

    monkeypatch.setattr(mcc.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(mcc.time, "sleep", _fake_sleep)
    monkeypatch.setattr(mcc.tempfile, "mkstemp", _fake_mkstemp)

    ok, reason = mcc._trtexec_export_from_onnx(onnx_path, engine_target, args, 640)
    assert ok is True
    assert reason == "ok"
    assert seen_cmds
    cmd = seen_cmds[0]
    assert "--explicitBatch" not in cmd
    assert any(arg.startswith("--memPoolSize=workspace:") for arg in cmd)


def test_trtexec_export_failure_reports_log_tail(monkeypatch, tmp_path: Path):
    onnx_path = tmp_path / "m.onnx"
    onnx_path.write_text("onnx", encoding="utf-8")
    engine_target = tmp_path / "m.trt"
    args = _base_args(workspace_gib=None, batch=1, dynamic=False, precision="fp32")

    monkeypatch.setattr(mcc, "_resolve_trtexec_bin", lambda: "/usr/bin/trtexec")
    monkeypatch.setattr(mcc, "_guess_onnx_input_name", lambda _p: "images")
    monkeypatch.setattr(
        mcc,
        "_get_trtexec_capabilities",
        lambda _bin: mcc.TrtexecCapabilities(
            supports_build_only=True,
            supports_explicit_batch=False,
            workspace_mode="workspace",
        ),
    )

    class _FailPopen:
        def __init__(self, _cmd, **_kwargs):
            out = _kwargs.get("stdout")
            if out is not None:
                out.write("line1\nline2\nfatal error\n")
                out.flush()
            self.returncode = 1

        def poll(self):
            return 1

    def _fake_mkstemp(**_kwargs):
        p = tmp_path / "trtexec_error.log"
        p.write_text("", encoding="utf-8")
        return (123, str(p))

    monkeypatch.setattr(mcc.subprocess, "Popen", _FailPopen)
    monkeypatch.setattr(mcc.tempfile, "mkstemp", _fake_mkstemp)

    ok, reason = mcc._trtexec_export_from_onnx(onnx_path, engine_target, args, 640)
    assert ok is False
    assert "fatal error" in reason
    assert "full log:" in reason


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


def test_validate_args_rejects_onnx_to_engine(monkeypatch):
    parser = mcc.build_model_convert_arg_parser()
    args = _base_args(format="tensorrt-engine", input="x.onnx")
    monkeypatch.setattr(
        mcc,
        "_get_export_format_availability",
        lambda: {"onnx": (True, ""), "tensorrt-engine": (True, ""), "tensorrt-trt": (True, "")},
    )
    try:
        mcc._validate_args(args, interactive_allowed=False, parser=parser, argv=["--input", "x.onnx", "--format", "tensorrt-engine"])
        assert False, "onnx to engine must be rejected"
    except SystemExit as exc:
        assert int(exc.code) != 0


def test_interactive_pt_wizard_reprompts_unavailable_target_model(monkeypatch, tmp_path: Path):
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
        workspace=None,
        device=None,
        data=None,
        fraction=1.0,
        workspace_gib=None,
        force=False,
        continue_on_error=False,
    )
    monkeypatch.setattr(
        mcc,
        "_discover_models",
        lambda _root, *, allowed_suffixes: [("models", tmp_path / ("m.pt" if allowed_suffixes == (".pt",) else "m.onnx"))],
    )
    monkeypatch.setattr(mcc, "_collect_input_models", lambda _p: [tmp_path / "m.pt"])
    monkeypatch.setattr(mcc, "_check_tensorrt_ready", lambda: (True, ""))
    monkeypatch.setattr(
        mcc,
        "_get_export_format_availability",
        lambda: {"onnx": (True, ""), "tensorrt-engine": (True, ""), "tensorrt-trt": (False, "runtime fail")},
    )
    choices = iter(["pt", "models: m.pt", "auto", "static"])
    target_lists = iter([["onnx", "trt (unavailable: runtime fail)"], ["onnx"]])
    seen_prompts: list[str] = []

    def _fake_prompt_choice(prompt, *a, **k):
        seen_prompts.append(prompt)
        return next(choices)

    monkeypatch.setattr(mcc, "prompt_choice", _fake_prompt_choice)
    monkeypatch.setattr(
        mcc,
        "prompt_multi_choice_csv",
        lambda prompt, *_a, **_k: (seen_prompts.append(prompt) or next(target_lists)),
    )
    ints = iter([1, 17])
    monkeypatch.setattr(mcc, "prompt_int", lambda *a, **k: next(ints))
    monkeypatch.setattr(mcc, "prompt_text", lambda *a, **k: "")
    monkeypatch.setattr(mcc, "prompt_yes_no", lambda *a, **k: False)

    mcc._interactive_fill(args, tmp_path)
    assert getattr(args, "_source_kind") == "pt"
    assert getattr(args, "_target_onnx") is True
    assert getattr(args, "_target_engine") is False
    assert getattr(args, "_target_trt") is False
    assert seen_prompts[:5] == [
        "Source model type",
        "Select input model",
        "Targets",
        "Targets",
        "ONNX image size mode",
    ]


def test_interactive_onnx_wizard_exits_when_no_targets_available(monkeypatch, tmp_path: Path):
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
        workspace=None,
        device=None,
        data=None,
        fraction=1.0,
        workspace_gib=None,
        force=False,
        continue_on_error=False,
    )
    monkeypatch.setattr(
        mcc,
        "_discover_models",
        lambda _root, *, allowed_suffixes: [("models", tmp_path / ("m.onnx" if allowed_suffixes == (".onnx",) else "m.pt"))],
    )
    monkeypatch.setattr(mcc, "_check_tensorrt_ready", lambda: (True, ""))
    monkeypatch.setattr(
        mcc,
        "_get_export_format_availability",
        lambda: {"onnx": (True, ""), "tensorrt-engine": (True, ""), "tensorrt-trt": (False, "runtime fail")},
    )
    choices = iter(["onnx", "models: m.onnx"])
    monkeypatch.setattr(mcc, "prompt_choice", lambda *a, **k: next(choices))
    monkeypatch.setattr(mcc, "prompt_int", lambda *a, **k: 1)
    monkeypatch.setattr(mcc, "prompt_text", lambda *a, **k: "")
    monkeypatch.setattr(mcc, "prompt_yes_no", lambda *a, **k: False)

    try:
        mcc._interactive_fill(args, tmp_path)
        assert False, "interactive flow must stop when no targets are available"
    except SystemExit as exc:
        assert "No target models are available for onnx source" in str(exc)


def test_interactive_onnx_wizard_offers_only_trt(monkeypatch, tmp_path: Path):
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
        workspace=None,
        device=None,
        data=None,
        fraction=1.0,
        workspace_gib=None,
        force=False,
        continue_on_error=False,
    )
    monkeypatch.setattr(
        mcc,
        "_discover_models",
        lambda _root, *, allowed_suffixes: [("models", tmp_path / "m.onnx")],
    )
    monkeypatch.setattr(mcc, "_check_tensorrt_ready", lambda: (True, ""))
    monkeypatch.setattr(
        mcc,
        "_get_export_format_availability",
        lambda: {"onnx": (True, ""), "tensorrt-engine": (True, ""), "tensorrt-trt": (True, "")},
    )

    calls: list[tuple[str, list[str]]] = []
    choices = iter(["onnx", "models: m.onnx", "auto", "static", "fp32"])
    multi_choices = iter([["trt"]])

    def _fake_prompt_choice(prompt, options, *a, **k):
        calls.append((prompt, list(options)))
        return next(choices)

    monkeypatch.setattr(mcc, "prompt_choice", _fake_prompt_choice)
    monkeypatch.setattr(
        mcc,
        "prompt_multi_choice_csv",
        lambda prompt, options, *a, **k: (calls.append((prompt, list(options))) or next(multi_choices)),
    )
    monkeypatch.setattr(mcc, "prompt_int", lambda *a, **k: 1)
    monkeypatch.setattr(mcc, "prompt_text", lambda *a, **k: "")
    monkeypatch.setattr(mcc, "prompt_yes_no", lambda *a, **k: False)

    mcc._interactive_fill(args, tmp_path)
    target_prompt = [opts for prompt, opts in calls if prompt == "Targets"][0]
    assert target_prompt == ["trt"]


def test_interactive_pipeline_engine_uses_pt_source_not_session_onnx(monkeypatch, tmp_path: Path):
    source_pt = tmp_path / "m.pt"
    source_pt.write_text("pt", encoding="utf-8")

    yolo_inits: list[Path] = []

    class _FakeYOLO:
        def __init__(self, path: str):
            yolo_inits.append(Path(path))

        def export(self, **_kwargs):
            exported = tmp_path / "ultralytics.engine"
            exported.write_text("engine", encoding="utf-8")
            return str(exported)

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)

    def _fake_export_named_onnx_from_pt(*_a, **_k):
        target_path = Path(_k["target_path"])
        target_path.write_text("onnx", encoding="utf-8")
        return True, "ok"

    monkeypatch.setattr(mcc, "_export_named_onnx_from_pt", _fake_export_named_onnx_from_pt)
    monkeypatch.setattr(mcc, "_maybe_move_output", lambda *_a, **_k: (True, "ok"))

    ctx = mcc.InteractiveContext(
        source_kind="pt",
        source_path=source_pt,
        target_onnx=True,
        target_engine=True,
        target_trt=False,
        output_dir=tmp_path,
        force=True,
        force_onnx=True,
        force_engine=True,
        force_trt=True,
        onnx_imgsz=640,
        onnx_imgsz_source="cli",
        onnx_batch=1,
        onnx_dynamic=False,
        device=None,
        engine_precision="fp32",
        engine_workspace_gib=None,
        trt_precision="fp32",
        trt_workspace_gib=None,
        data=None,
        fraction=1.0,
        opset=17,
        simplify=True,
        half=False,
        nms=False,
    )
    result = mcc._run_interactive_pipeline(ctx)
    assert result.stats.failed == 0
    assert yolo_inits
    assert all(p.suffix == ".pt" for p in yolo_inits)
