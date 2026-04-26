from __future__ import annotations

from smartrain import cli


def test_deps_sync_torch_prints_skipped_message(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "smartrain.external_providers.installer.sync_torch_cuda_policy_current_env",
        lambda **_k: ("skipped", "existing torch CUDA runtime is 13.0; keeping installed version"),
    )

    cli.cmd_deps_sync_torch()
    out = capsys.readouterr().out
    assert "[SKIPPED]" in out
    assert "13.0" in out

