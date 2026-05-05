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


def test_external_provider_adapter_run_train_delegates_to_runner(monkeypatch) -> None:
    adapter = ExternalProviderAdapter(provider_id="dr-yolo", repo_path="/tmp/repo", venv_path="/tmp/venv")
    captured: dict[str, object] = {}

    def _fake_train(provider_id, repo_path, venv_path, **kwargs):
        captured["provider_id"] = provider_id
        captured["repo_path"] = repo_path
        captured["venv_path"] = venv_path
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr("smartrain.backends.external_provider_adapter.run_external_train", _fake_train)
    rc = adapter.run_train(
        dataset_path="/tmp/ds",
        model="yolo11n",
        epochs=1,
        batch=2,
        imgsz=640,
    )
    assert rc == 0
    assert captured["provider_id"] == "dr-yolo"
