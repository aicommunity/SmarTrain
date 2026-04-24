from __future__ import annotations

from pathlib import Path

from smartrain import provider_global_index as pgi


def test_provider_global_index_roundtrip(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg))
    pgi.upsert_provider_record(
        {
            "provider_id": "leaf-yolo",
            "display_name": "LEAF-YOLO",
            "repo_path": "/tmp/leaf",
            "venv_path": "/tmp/leaf/venv",
            "install_root": "/tmp",
            "install_state": "installed",
            "detected_capabilities": {"train": True, "infer": True},
            "repo_ref": {"remote_url": "https://example", "branch": "main", "commit": "abc"},
            "installed_at": "2026-01-01T00:00:00+00:00",
            "last_validated_at": "2026-01-01T00:00:00+00:00",
            "last_error": None,
        }
    )
    recs = pgi.list_provider_records()
    assert len(recs) == 1
    assert recs[0]["provider_id"] == "leaf-yolo"
    loc = pgi.get_provider_location("leaf-yolo")
    assert loc is not None
    assert loc.repo_path == "/tmp/leaf"

