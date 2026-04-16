from __future__ import annotations

import json
from pathlib import Path

import pytest

from smartrain import model_training_module as mtm
from smartrain import train_resume as tr
from smartrain.workspace_paths import deploy_workspace


def test_diagnose_run_marks_resumable_when_last_checkpoint_exists(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run1"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "args.yaml").write_text("epochs: 100\n", encoding="utf-8")
    (run_dir / "train" / "weights" / "last.pt").write_text("bin", encoding="utf-8")

    diag = tr.diagnose_run(str(run_dir))
    assert diag.status == tr.RUN_STATUS_RESUMABLE_INCOMPLETE
    assert "resume_checkpoint_available" in diag.reasons


def test_diagnose_run_marks_completed_from_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run2"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"status": {"training": {"success": True}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    diag = tr.diagnose_run(str(run_dir))
    assert diag.status == tr.RUN_STATUS_COMPLETED


def test_diagnose_run_marks_incomplete_non_resumable_without_last(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run3"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "args.yaml").write_text("epochs: 300\n", encoding="utf-8")
    (run_dir / "train" / "results.csv").write_text("epoch,loss\n1,1.0\n", encoding="utf-8")

    diag = tr.diagnose_run(str(run_dir))
    assert diag.status == tr.RUN_STATUS_INCOMPLETE_NON_RESUMABLE
    assert "missing_last_checkpoint" in diag.reasons


def test_resume_noninteractive_requires_run_dir(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    code = mtm._run_resume_command(["--workspace", str(tmp_path), "-y"])
    assert code == 2


def test_resume_run_dir_fails_for_nonresumable_run(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds" / "run4"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "args.yaml").write_text("epochs: 30\n", encoding="utf-8")

    code = mtm._run_resume_command(
        ["--workspace", str(tmp_path), "--run-dir", str(run_dir), "-y"]
    )
    assert code == 2


def test_resume_run_dir_success_calls_resume_and_updates_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds" / "run5"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "args.yaml").write_text("epochs: 30\n", encoding="utf-8")
    (run_dir / "train" / "weights" / "last.pt").write_text("bin", encoding="utf-8")

    called: dict[str, bool] = {"resume": False, "metadata": False}

    def _fake_resume(path: str) -> None:
        assert path == str(run_dir)
        called["resume"] = True

    def _fake_meta(path: str, *, success: bool, error: str | None, diagnosis: tr.RunDiagnosis | None) -> None:
        assert path == str(run_dir)
        assert success is True
        assert error is None
        assert diagnosis is not None
        called["metadata"] = True

    monkeypatch.setattr(mtm, "resume_training_in_run", _fake_resume)
    monkeypatch.setattr(mtm, "update_resume_metadata", _fake_meta)

    code = mtm._run_resume_command(
        ["--workspace", str(tmp_path), "--run-dir", str(run_dir), "-y"]
    )
    assert code == 0
    assert called["resume"] is True
    assert called["metadata"] is True
