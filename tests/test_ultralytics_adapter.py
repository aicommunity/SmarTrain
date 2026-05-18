from __future__ import annotations

from types import SimpleNamespace

from smartrain.backends.ultralytics_adapter import UltralyticsAdapter


def test_ultralytics_adapter_exposes_capability_contract() -> None:
    adapter = UltralyticsAdapter()
    assert adapter.backend_id == "ultralytics"
    assert adapter.capabilities.can_train is True
    assert adapter.capabilities.can_test is True
    assert adapter.capabilities.can_infer is True
    assert adapter.capabilities.supports(task_type="detection", model_format="pt") is True


def test_ultralytics_adapter_infer_validates_request() -> None:
    adapter = UltralyticsAdapter()
    result = adapter.infer(request=SimpleNamespace(model_format="", model_path=""))
    assert result.success is False
    assert result.error


def test_ultralytics_adapter_infer_uses_task_hint(monkeypatch) -> None:
    adapter = UltralyticsAdapter()
    captured: dict[str, str] = {}

    def _fake_create(self, **kwargs):
        captured["task_type"] = str(kwargs.get("task_type"))
        return object()

    monkeypatch.setattr(UltralyticsAdapter, "create_inference_backend", _fake_create)
    result = adapter.infer(
        request=SimpleNamespace(model_format="pt", model_path="/tmp/model.pt", task_type="classify")
    )
    assert result.success is True
    assert result.task_type == "classification"
    assert captured["task_type"] == "classification"
