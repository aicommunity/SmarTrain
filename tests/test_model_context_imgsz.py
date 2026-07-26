from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from smartrain.services.inference_runtime_helpers import apply_inference_imgsz_from_model
from smartrain.workflows.models.model_artifact_imgsz import (
    DEFAULT_INFERENCE_IMGSZ,
    extract_imgsz_from_sidecar_payload,
    parse_imgsz_from_artifact_filename,
)
from smartrain.workflows.models.model_context import (
    FALLBACK_IMGSZ_SOURCE,
    infer_img_size_with_source,
    resolve_inference_imgsz,
)


def test_parse_imgsz_from_artifact_filename() -> None:
    assert parse_imgsz_from_artifact_filename("run_imgsz1280x1280_b1_static.onnx") == 1280
    assert parse_imgsz_from_artifact_filename("model.onnx") is None


def test_extract_imgsz_from_sidecar_payload() -> None:
    assert extract_imgsz_from_sidecar_payload({"params": {"imgsz": 960}}) == 960
    assert extract_imgsz_from_sidecar_payload({"image_size": 800}) == 800
    assert extract_imgsz_from_sidecar_payload({}) is None


def test_infer_from_training_metadata(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "bundle"
    model_dir.mkdir(parents=True)
    weights = model_dir / "best.pt"
    weights.write_text("pt", encoding="utf-8")
    (model_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"hyperparameters": {"image_size": 1280}}}),
        encoding="utf-8",
    )

    value, source = infer_img_size_with_source(weights)
    assert value == 1280
    assert source == "training_metadata"


def test_infer_from_ancestor_training_metadata(tmp_path: Path) -> None:
    bundle = tmp_path / "models" / "promoted_bundle"
    nested = bundle / "variants"
    nested.mkdir(parents=True)
    weights = nested / "detect_imgsz640x640_b1.onnx"
    weights.write_text("onnx", encoding="utf-8")
    (bundle / "training_metadata.json").write_text(
        json.dumps({"hyperparameters": {"image_size": 1536}}),
        encoding="utf-8",
    )

    value, source = infer_img_size_with_source(weights)
    assert value == 1536
    assert source == "training_metadata"


def test_infer_from_artifact_filename_without_metadata(tmp_path: Path) -> None:
    weights = tmp_path / "run_imgsz1024x1024_b1_static.onnx"
    weights.write_text("onnx", encoding="utf-8")

    value, source = infer_img_size_with_source(weights)
    assert value == 1024
    assert source == "artifact_filename"


def test_infer_from_sidecar_metadata(tmp_path: Path) -> None:
    weights = tmp_path / "model.onnx"
    weights.write_text("onnx", encoding="utf-8")
    sidecar = tmp_path / "model.onnx.meta.json"
    sidecar.write_text(json.dumps({"params": {"imgsz": 736}}), encoding="utf-8")

    value, source = infer_img_size_with_source(weights)
    assert value == 736
    assert source == "sidecar_metadata"


def test_infer_from_onnx_input_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "plain.onnx"
    weights.write_text("onnx", encoding="utf-8")

    monkeypatch.setattr(
        "smartrain.core.models.model_context.extract_onnx_input_imgsz",
        lambda _path: 800,
    )

    value, source = infer_img_size_with_source(weights)
    assert value == 800
    assert source == "onnx_input_shape"


def test_infer_fallback_when_no_sources(tmp_path: Path) -> None:
    weights = tmp_path / "model.pt"
    weights.write_text("pt", encoding="utf-8")

    value, source = infer_img_size_with_source(weights)
    assert value is None
    assert source == FALLBACK_IMGSZ_SOURCE


def test_resolve_inference_imgsz_explicit_cli(tmp_path: Path) -> None:
    weights = tmp_path / "model.pt"
    weights.write_text("pt", encoding="utf-8")

    imgsz, source = resolve_inference_imgsz(weights, explicit=512)
    assert imgsz == 512
    assert source == "cli"


def test_resolve_inference_imgsz_fallback(tmp_path: Path) -> None:
    weights = tmp_path / "model.pt"
    weights.write_text("pt", encoding="utf-8")

    imgsz, source = resolve_inference_imgsz(weights)
    assert imgsz == DEFAULT_INFERENCE_IMGSZ
    assert source == FALLBACK_IMGSZ_SOURCE


def test_apply_inference_imgsz_from_model_info(capsys, tmp_path: Path) -> None:
    weights = tmp_path / "run_imgsz1280x1280_b1_static.onnx"
    weights.write_text("onnx", encoding="utf-8")
    args = argparse.Namespace(img_size=None)

    imgsz, source = apply_inference_imgsz_from_model(weights, args)

    assert imgsz == 1280
    assert source == "artifact_filename"
    assert args.img_size == 1280
    assert args.img_size_source == "artifact_filename"
    assert "[INFO] Resolved input size: 1280 (source: artifact_filename)" in capsys.readouterr().out


def test_apply_inference_imgsz_from_model_warns_on_fallback(capsys, tmp_path: Path) -> None:
    weights = tmp_path / "model.pt"
    weights.write_text("pt", encoding="utf-8")
    args = argparse.Namespace(img_size=None)

    imgsz, source = apply_inference_imgsz_from_model(weights, args)

    assert imgsz == DEFAULT_INFERENCE_IMGSZ
    assert source == FALLBACK_IMGSZ_SOURCE
    assert args.img_size == DEFAULT_INFERENCE_IMGSZ
    assert args.img_size_source == FALLBACK_IMGSZ_SOURCE
    out = capsys.readouterr().out
    assert "[WARN]" in out
    assert str(DEFAULT_INFERENCE_IMGSZ) in out
