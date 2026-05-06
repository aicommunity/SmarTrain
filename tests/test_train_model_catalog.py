from __future__ import annotations

import pytest

from smartrain.core.training.train_backend_registry import (
    _aliases_from_yaml_names,
    default_train_provider,
    get_train_backend_spec,
    list_train_providers,
)
from smartrain.core.training.train_model_catalog import TrainModelCatalog
from smartrain.core.training.train_model_resolver import TrainModelResolver


def test_train_model_catalog_contains_expected_aliases() -> None:
    aliases = TrainModelCatalog().supported_aliases()
    assert "yolov8n" in aliases
    assert "yolo11x" in aliases


def test_train_model_resolver_adds_pt_for_yolo_alias() -> None:
    resolved = TrainModelResolver().resolve("yolo11s", default_model="yolov8n", add_pt_when_missing=True)
    assert resolved.normalized == "yolo11s.pt"
    assert resolved.is_supported_alias is True


def test_train_model_resolver_keeps_custom_model_supported_flag_false() -> None:
    resolved = TrainModelResolver().resolve(
        "custom-fork-model.pt",
        default_model="yolov8n",
        add_pt_when_missing=True,
    )
    assert resolved.normalized == "custom-fork-model.pt"
    assert resolved.is_supported_alias is False


def test_train_backend_registry_default_provider() -> None:
    assert default_train_provider() == "ultralytics"
    assert "ultralytics" in list_train_providers()


def test_train_backend_registry_unknown_provider_raises() -> None:
    with pytest.raises(ValueError):
        get_train_backend_spec("missing-provider")


def test_aliases_from_yaml_names_includes_rtdetr() -> None:
    aliases = _aliases_from_yaml_names(["yolo11.yaml", "rtdetr-l.yaml", "rtdetr-x.yaml"])
    assert "rtdetr-l" in aliases
    assert "rtdetr-x" in aliases
    assert "yolo11" in aliases


def test_train_model_catalog_is_populated_from_backend_spec() -> None:
    aliases = TrainModelCatalog().supported_aliases()
    assert len(aliases) > 0
    # RT-DETR aliases are expected in modern ultralytics builds.
    # In fallback mode (when dynamic discovery is unavailable), only YOLO aliases may be present.
    assert any(a.startswith("rtdetr") for a in aliases) or "yolov8n" in aliases


def test_external_provider_catalog_uses_provider_specific_aliases() -> None:
    aliases = TrainModelCatalog(provider="dr-yolo").supported_aliases()
    assert "yolov8n" in aliases


def test_mfel_catalog_reads_custom_yaml_aliases(tmp_path) -> None:
    cfg_dir = tmp_path / "ultralytics" / "cfg" / "MFEL-YOLO"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "MFEL-YOLO.yaml").write_text("nc: 1\n", encoding="utf-8")
    (cfg_dir / "E_PAN+.yaml").write_text("nc: 1\n", encoding="utf-8")
    aliases = TrainModelCatalog(provider="mfel-yolo", provider_repo_path=str(tmp_path)).supported_aliases()
    assert "mfel-yolo" in aliases
    assert "e_pan+" in aliases

