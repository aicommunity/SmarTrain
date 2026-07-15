from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from smartrain.services.inference_runtime_helpers import (
    apply_inference_batch_from_model,
    default_inference_batch_for_model,
    resolve_onnx_batch_constraint,
)
from smartrain.workflows.models.model_artifact_imgsz import (
    extract_batch_from_sidecar_payload,
    extract_onnx_input_batch,
)


def test_extract_batch_from_sidecar_payload() -> None:
    assert extract_batch_from_sidecar_payload({"params": {"batch": 1, "dynamic": False}}) == (1, False)
    assert extract_batch_from_sidecar_payload({"params": {"batch": 4, "dynamic": True}}) == (4, True)
    assert extract_batch_from_sidecar_payload({"params": {"dynamic": "static"}}) == (None, False)
    assert extract_batch_from_sidecar_payload({}) == (None, None)


def test_extract_onnx_input_batch_static(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    weights = tmp_path / "model.onnx"
    weights.write_text("onnx", encoding="utf-8")

    d0 = SimpleNamespace(dim_value=1, dim_param="")
    d1 = SimpleNamespace(dim_value=3, dim_param="")
    d2 = SimpleNamespace(dim_value=640, dim_param="")
    d3 = SimpleNamespace(dim_value=640, dim_param="")
    fake_model = SimpleNamespace(graph=SimpleNamespace(input=[SimpleNamespace(type=SimpleNamespace(tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=[d0, d1, d2, d3]))))]))

    monkeypatch.setitem(__import__("sys").modules, "onnx", SimpleNamespace(load=lambda _p: fake_model))
    assert extract_onnx_input_batch(weights) == (1, False)


def test_extract_onnx_input_batch_dynamic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    weights = tmp_path / "model.onnx"
    weights.write_text("onnx", encoding="utf-8")

    d0 = SimpleNamespace(dim_value=0, dim_param="batch")
    d1 = SimpleNamespace(dim_value=3, dim_param="")
    d2 = SimpleNamespace(dim_value=640, dim_param="")
    d3 = SimpleNamespace(dim_value=640, dim_param="")
    fake_model = SimpleNamespace(graph=SimpleNamespace(input=[SimpleNamespace(type=SimpleNamespace(tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=[d0, d1, d2, d3]))))]))

    monkeypatch.setitem(__import__("sys").modules, "onnx", SimpleNamespace(load=lambda _p: fake_model))
    assert extract_onnx_input_batch(weights) == (None, True)


def test_resolve_constraint_from_sidecar_when_onnx_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "model.onnx"
    weights.write_text("onnx", encoding="utf-8")
    (tmp_path / "model.onnx.meta.json").write_text(
        json.dumps({"params": {"batch": 1, "dynamic": False}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "smartrain.workflows.models.model_artifact_imgsz.extract_onnx_input_batch",
        lambda _p: (None, None),
    )

    batch, dynamic, source = resolve_onnx_batch_constraint(weights)
    assert batch == 1
    assert dynamic is False
    assert source == "sidecar_metadata"


def test_resolve_constraint_skips_pt(tmp_path: Path) -> None:
    weights = tmp_path / "model.pt"
    weights.write_text("pt", encoding="utf-8")
    assert resolve_onnx_batch_constraint(weights) == (None, None, "n/a")


def test_apply_clamps_static_onnx(capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "model.onnx"
    weights.write_text("onnx", encoding="utf-8")
    monkeypatch.setattr(
        "smartrain.services.inference_runtime_helpers.resolve_onnx_batch_constraint",
        lambda _p: (1, False, "onnx_input_shape"),
    )
    args = argparse.Namespace(batch_size=8)

    assert apply_inference_batch_from_model(weights, args) == 1
    assert args.batch_size == 1
    out = capsys.readouterr().out
    assert "[WARN]" in out
    assert "clamping --batch-size from 8 to 1" in out


def test_apply_keeps_dynamic_onnx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "model.onnx"
    weights.write_text("onnx", encoding="utf-8")
    monkeypatch.setattr(
        "smartrain.services.inference_runtime_helpers.resolve_onnx_batch_constraint",
        lambda _p: (None, True, "onnx_input_shape"),
    )
    args = argparse.Namespace(batch_size=8)

    assert apply_inference_batch_from_model(weights, args) == 8
    assert args.batch_size == 8


def test_apply_keeps_requested_when_within_fixed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "model.onnx"
    weights.write_text("onnx", encoding="utf-8")
    monkeypatch.setattr(
        "smartrain.services.inference_runtime_helpers.resolve_onnx_batch_constraint",
        lambda _p: (4, False, "onnx_input_shape"),
    )
    args = argparse.Namespace(batch_size=2)

    assert apply_inference_batch_from_model(weights, args) == 2
    assert args.batch_size == 2


def test_default_inference_batch_for_static_onnx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "model.onnx"
    weights.write_text("onnx", encoding="utf-8")
    monkeypatch.setattr(
        "smartrain.services.inference_runtime_helpers.resolve_onnx_batch_constraint",
        lambda _p: (1, False, "onnx_input_shape"),
    )
    assert default_inference_batch_for_model(weights, fallback=8) == 1


def test_default_inference_batch_for_pt(tmp_path: Path) -> None:
    weights = tmp_path / "model.pt"
    weights.write_text("pt", encoding="utf-8")
    assert default_inference_batch_for_model(weights, fallback=8) == 8
