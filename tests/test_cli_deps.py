from __future__ import annotations

import click
import pytest

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


def test_deps_doctor_returns_nonzero_when_pandoc_missing(monkeypatch, capsys) -> None:
    from smartrain.services.deps.optional_extras import DepCheckRow, ExportDepsReport

    monkeypatch.setattr(
        "smartrain.services.deps.optional_extras.check_export_deps",
        lambda: ExportDepsReport(
            rows=(
                DepCheckRow("pypandoc (base)", False, "not installed"),
                DepCheckRow("pandoc", False, "not found"),
                DepCheckRow("weasyprint (export extra)", False, "not installed"),
                DepCheckRow("fpdf2 (base)", True, "import ok"),
                DepCheckRow("odfpy (base)", True, "import ok"),
            ),
            export_ready=False,
        ),
    )
    monkeypatch.setattr("smartrain.services.deps.optional_extras.ubuntu_weasyprint_apt_hint", lambda: None)

    with pytest.raises(click.exceptions.Exit) as exc:
        cli.cmd_deps_doctor(verbose=False)
    out = capsys.readouterr().out
    assert exc.value.exit_code == 1
    assert "Export dependencies doctor" in out
    assert "Reinstall smartrain" in out


def test_deps_install_dry_run_uses_export_default(capsys) -> None:
    with pytest.raises(click.exceptions.Exit) as exc:
        cli.cmd_deps_install(extra=[], all_extras=False, dry_run=True)
    out = capsys.readouterr().out
    assert exc.value.exit_code == 0
    assert "Would run:" in out
    assert "pip install" in out


def test_deps_install_unknown_extra_exits_2(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "smartrain.services.deps.optional_extras.install_optional_extras",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("Unknown extras: aaa")),
    )
    with pytest.raises(click.exceptions.Exit) as exc:
        cli.cmd_deps_install(extra=["aaa"], all_extras=False, dry_run=False)
    err = capsys.readouterr().err
    assert exc.value.exit_code == 2
    assert "Unknown extras" in err
