from __future__ import annotations

from pathlib import Path

from smartrain.external_providers import installer
from smartrain.external_providers.base import ExternalProviderSpec


def test_install_provider_succeeds_with_runtime_warning(monkeypatch, tmp_path: Path) -> None:
    spec = ExternalProviderSpec(
        id="mfel-yolo",
        display_name="MFEL-YOLO",
        repo_url="https://example/repo.git",
        branch="main",
        train_entry="train.py",
        infer_entry="val.py",
    )
    monkeypatch.setattr(installer, "get_provider_spec", lambda _pid: spec)
    saved: dict[str, object] = {}
    monkeypatch.setattr(installer, "upsert_provider_record", lambda rec: saved.setdefault("record", rec))
    monkeypatch.setattr(installer, "write_provider_state", lambda *a, **k: None)
    monkeypatch.setattr(installer, "_git_rev", lambda _p: "abc")

    def _fake_run(cmd: list[str], *, cwd: str, timeout_sec=None) -> None:
        if cmd[:2] == ["git", "clone"]:
            repo_dir = Path(cmd[-1])
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / ".git").mkdir(parents=True, exist_ok=True)
            (repo_dir / "train.py").write_text("print('train')\n", encoding="utf-8")
            (repo_dir / "val.py").write_text("print('val')\n", encoding="utf-8")
            (repo_dir / "requirements.txt").write_text("ultralytics\n", encoding="utf-8")

    monkeypatch.setattr(installer, "_run", _fake_run)
    monkeypatch.setattr(installer, "_probe_torch_state", lambda *_a, **_k: {"installed": False, "cuda": "", "version": ""})

    class _FakeBuilder:
        def __init__(self, with_pip: bool):
            self.with_pip = with_pip

        def create(self, path: str) -> None:
            v = Path(path) / "bin"
            v.mkdir(parents=True, exist_ok=True)
            (v / "python").write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    monkeypatch.setattr(installer.venv, "EnvBuilder", _FakeBuilder)
    monkeypatch.setattr(
        installer,
        "probe_provider_repo",
        lambda *a, **k: {
            "entrypoints_ok": True,
            "runtime_ok": False,
            "runtime_reason": "missing runtime dependency DCNv4",
        },
    )
    res = installer.install_provider("mfel-yolo", str(tmp_path))
    assert res.ok is True
    assert res.action == "installed"
    assert "runtime warning" in res.message
    rec = saved["record"]
    assert isinstance(rec, dict)
    assert rec["install_state"] == "installed"
    assert "DCNv4" in str(rec["last_error"])


def test_install_provider_auto_fetches_dcnv4_for_mfel(monkeypatch, tmp_path: Path) -> None:
    spec = ExternalProviderSpec(
        id="mfel-yolo",
        display_name="MFEL-YOLO",
        repo_url="https://example/repo.git",
        branch="main",
        train_entry="train.py",
        infer_entry="val.py",
    )
    monkeypatch.setattr(installer, "get_provider_spec", lambda _pid: spec)
    monkeypatch.setattr(installer, "upsert_provider_record", lambda rec: None)
    monkeypatch.setattr(installer, "write_provider_state", lambda *a, **k: None)
    monkeypatch.setattr(installer, "_git_rev", lambda _p: "abc")
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], *, cwd: str, timeout_sec=None) -> None:
        calls.append(list(cmd))
        if cmd[:5] == [cmd[0], "-m", "pip", "install", "--no-build-isolation"] and cmd[-1] == "DCNv4":
            raise RuntimeError("force fallback to git clone")
        if cmd[:2] == ["git", "clone"] and "example/repo.git" in cmd:
            repo_dir = Path(cmd[-1])
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / ".git").mkdir(parents=True, exist_ok=True)
            (repo_dir / "train.py").write_text("print('train')\n", encoding="utf-8")
            (repo_dir / "val.py").write_text("print('val')\n", encoding="utf-8")
            (repo_dir / "requirements.txt").write_text("ultralytics\n", encoding="utf-8")
        if cmd[:2] == ["git", "clone"] and "OpenGVLab/DCNv4.git" in cmd:
            dcn_dir = Path(cmd[-1]) / "DCNv4_op"
            dcn_dir.mkdir(parents=True, exist_ok=True)
            (dcn_dir / "setup.py").write_text("from setuptools import setup\nsetup(name='dcnv4')\n", encoding="utf-8")

    monkeypatch.setattr(installer, "_run", _fake_run)
    monkeypatch.setattr(installer, "_probe_torch_state", lambda *_a, **_k: {"installed": False, "cuda": "", "version": ""})

    class _FakeBuilder:
        def __init__(self, with_pip: bool):
            self.with_pip = with_pip

        def create(self, path: str) -> None:
            v = Path(path) / "bin"
            v.mkdir(parents=True, exist_ok=True)
            (v / "python").write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    monkeypatch.setattr(installer.venv, "EnvBuilder", _FakeBuilder)
    monkeypatch.setattr(
        installer,
        "probe_provider_repo",
        lambda *a, **k: {"entrypoints_ok": True, "runtime_ok": True, "runtime_reason": ""},
    )
    res = installer.install_provider("mfel-yolo", str(tmp_path))
    assert res.ok is True
    assert any("OpenGVLab/DCNv4.git" in " ".join(cmd) for cmd in calls)


def test_sync_torch_policy_installs_cu128_when_torch_absent(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(installer, "_probe_torch_state", lambda *_a, **_k: {"installed": False, "cuda": "", "version": ""})
    monkeypatch.setattr(installer, "_run", lambda cmd, **_k: calls.append(list(cmd)))

    action, message = installer._sync_torch_cuda_policy("python", cwd="/tmp")

    assert action == "installed"
    assert "cu128" in message
    assert calls
    cmd = calls[0]
    assert cmd[:5] == ["python", "-m", "pip", "install", "--index-url"]
    assert installer.TORCH_CUDA_DEFAULT_INDEX_URL in cmd
    assert "torch" in cmd
    assert "torchvision" in cmd
    assert "torchaudio" in cmd


def test_sync_torch_policy_skips_when_cuda13_already_installed(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        installer,
        "_probe_torch_state",
        lambda *_a, **_k: {"installed": True, "cuda": "13.0", "version": "2.8.0+cu130"},
    )
    monkeypatch.setattr(installer, "_run", lambda cmd, **_k: calls.append(list(cmd)))

    action, message = installer._sync_torch_cuda_policy("python", cwd="/tmp")

    assert action == "skipped"
    assert "13.0" in message
    assert calls == []


def test_sync_torch_policy_reinstalls_when_non_cuda13_torch_present(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        installer,
        "_probe_torch_state",
        lambda *_a, **_k: {"installed": True, "cuda": "12.4", "version": "2.6.0+cu124"},
    )
    monkeypatch.setattr(installer, "_run", lambda cmd, **_k: calls.append(list(cmd)))

    action, _ = installer._sync_torch_cuda_policy("python", cwd="/tmp")

    assert action == "installed"
    assert len(calls) == 1
