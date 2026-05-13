from __future__ import annotations

from smartrain.providers import cli as providers_cli


def test_doctor_marks_runtime_issue_as_broken(monkeypatch, capsys) -> None:
    spec = providers_cli.list_provider_specs()[0]
    monkeypatch.setattr(
        providers_cli,
        "list_provider_records",
        lambda: [
            {
                "provider_id": spec.id,
                "repo_path": "/tmp/repo",
                "venv_path": "/tmp/venv",
                "install_state": "installed",
                "last_error": None,
            }
        ],
    )
    monkeypatch.setattr(
        providers_cli,
        "probe_provider_repo",
        lambda repo, s, venv: {
            "repo_found": True,
            "entrypoints_ok": True,
            "train_entry_ok": True,
            "infer_entry_ok": True,
            "requirements_ok": True,
            "venv_ready": True,
            "venv_python": "/tmp/venv/bin/python",
            "runtime_ok": False,
            "runtime_reason": "missing runtime dependency DCNv4",
        },
    )
    rc = providers_cli.main(["doctor", "--verbose"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "broken" in out
    assert "runtime=False" in out
    assert "runtime_reason: missing runtime dependency DCNv4" in out
