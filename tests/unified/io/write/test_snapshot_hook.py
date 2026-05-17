from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from smartrain.unified.io.read.run_adapter import RunAdapter
from smartrain.unified.io.write.snapshot_hook import maybe_dual_write_unified_snapshot


def test_maybe_dual_write_invokes_run_dual_write_when_env_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SMARTTRAIN_UNIFIED_WRITE", "1")
    monkeypatch.setenv("SMARTTRAIN_UNIFIED_DUAL_WRITE_MODE", "unified_only")
    run_dir = tmp_path / "runs" / "ds" / "run_hook"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "training_metadata.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_run_dual_write(**kwargs):
        captured["kwargs"] = kwargs
        return None

    payload = RunAdapter().read(str(run_dir))
    with patch("smartrain.unified.io.write.dual_write.run_dual_write", _fake_run_dual_write):
        with patch("smartrain.orchestrators.unified_gateway.load_target", return_value=payload):
            with patch("smartrain.unified.io.read.resolvers.infer_source_kind", return_value="run"):
                maybe_dual_write_unified_snapshot(str(run_dir), status_ok=True)
    assert "kwargs" in captured
    kw = captured["kwargs"]
    assert isinstance(kw, dict)
    assert kw.get("target_root") == str(run_dir)
    assert kw.get("legacy_writer") is None


def test_maybe_dual_write_noop_when_env_unset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SMARTTRAIN_UNIFIED_WRITE", raising=False)
    calls: list[object] = []

    def _fake_run_dual_write(**kwargs):
        calls.append(kwargs)

    with patch("smartrain.unified.io.write.dual_write.run_dual_write", _fake_run_dual_write):
        maybe_dual_write_unified_snapshot(str(tmp_path / "missing"), status_ok=True)
    assert calls == []


def test_maybe_dual_write_noop_when_not_ok(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SMARTTRAIN_UNIFIED_WRITE", "1")
    calls: list[object] = []

    def _fake_run_dual_write(**kwargs):
        calls.append(kwargs)

    with patch("smartrain.unified.io.write.dual_write.run_dual_write", _fake_run_dual_write):
        maybe_dual_write_unified_snapshot(str(tmp_path), status_ok=False)
    assert calls == []
