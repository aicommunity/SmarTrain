from __future__ import annotations

from pathlib import Path

from smartrain.adapters.canonical.read.model_adapter import ModelAdapter


def test_model_adapter_reads_canonical_payload(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "demo_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "demo_model.pt").write_bytes(b"fake")

    payload = ModelAdapter().read(str(model_dir))
    assert payload.models
    assert payload.models[0].model_id == "demo_model"
    assert payload.models[0].format == "pt"

