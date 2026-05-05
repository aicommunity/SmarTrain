from __future__ import annotations

from types import SimpleNamespace

from smartrain.backends.external_provider_adapter import ExternalProviderAdapter


def test_external_provider_adapter_validates_request() -> None:
    adapter = ExternalProviderAdapter(provider_id="dr-yolo", repo_path="/tmp/repo", venv_path="/tmp/venv")
    result = adapter.infer(request=SimpleNamespace(model_path="", source_path=""))
    assert result.success is False
    assert result.error


def test_external_provider_adapter_returns_success_from_runtime(monkeypatch) -> None:
    adapter = ExternalProviderAdapter(provider_id="dr-yolo", repo_path="/tmp/repo", venv_path="/tmp/venv")

    class _Runtime:
        def run_batch(self, **_kwargs):
            return 0

    monkeypatch.setattr(ExternalProviderAdapter, "create_runtime_backend", lambda self: _Runtime())
    result = adapter.infer(
        request=SimpleNamespace(
            model_path="/tmp/model.pt",
            source_path="/tmp/images",
            model_format="external",
            conf=0.2,
            imgsz=640,
            device="cpu",
        )
    )
    assert result.success is True
    assert result.backend == "external:dr-yolo"
