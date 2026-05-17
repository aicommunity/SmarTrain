from __future__ import annotations

from pathlib import Path

from smartrain.run_model_contract.io.read.factory import ReadAdapterFactory
from smartrain.run_model_contract.io.read.model_adapter import ModelAdapter
from smartrain.run_model_contract.io.read.run_adapter import RunAdapter


def test_factory_resolves_explicit_kinds(tmp_path: Path) -> None:
    factory = ReadAdapterFactory()
    assert isinstance(factory.resolve("run", str(tmp_path)), RunAdapter)
    assert isinstance(factory.resolve("model", str(tmp_path)), ModelAdapter)


def test_factory_inferrs_kind_from_ref(tmp_path: Path) -> None:
    run_ref = tmp_path / "runs" / "x"
    model_ref = tmp_path / "models" / "y"
    assert isinstance(ReadAdapterFactory().resolve(None, str(run_ref)), RunAdapter)
    assert isinstance(ReadAdapterFactory().resolve(None, str(model_ref)), ModelAdapter)

