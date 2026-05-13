from __future__ import annotations

from smartrain.providers import cli as providers_cli


def test_providers_install_non_interactive(monkeypatch) -> None:
    seen: list[str] = []

    class _Res:
        def __init__(self, provider_id: str):
            self.provider_id = provider_id
            self.action = "installed"
            self.ok = True
            self.message = "ok"

    monkeypatch.setattr(
        providers_cli,
        "install_provider",
        lambda provider_id, target_dir=None: (seen.append(provider_id), _Res(provider_id))[1],
    )
    rc = providers_cli.main(["install", "--provider", "dr-yolo", "-y"])
    assert rc == 0
    assert seen == ["dr-yolo"]

