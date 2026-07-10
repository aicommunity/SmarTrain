from __future__ import annotations

from pathlib import Path

from smartrain.services.reporting import document_export as de


def test_pandoc_executable_prefers_env_path(monkeypatch, tmp_path: Path) -> None:
    pandoc = tmp_path / "pandoc"
    pandoc.write_text("", encoding="utf-8")
    monkeypatch.setenv("PANDOC", str(pandoc))
    assert de._pandoc_executable() == str(pandoc)


def test_try_pandoc_pdf_returns_false_when_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(de, "_pandoc_executable", lambda: None)
    assert de._try_pandoc_pdf(str(tmp_path), "en") is False


def test_try_pandoc_odt_success(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "en").mkdir(parents=True, exist_ok=True)
    (tmp_path / "en" / "index.md").write_text("# test", encoding="utf-8")
    monkeypatch.setattr(de, "_pandoc_executable", lambda: "pandoc")

    class DummyProc:
        returncode = 0
        stderr = ""
        stdout = ""

    monkeypatch.setattr(de.subprocess, "run", lambda *args, **kwargs: DummyProc())
    assert de._try_pandoc_odt(str(tmp_path), "en") is True
