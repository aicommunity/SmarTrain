from __future__ import annotations

import json
from pathlib import Path

import pytest

from smartrain import model_training_module as mtm
from smartrain import run_discovery as rd
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
    (run_dir / "test").mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"status": {"training": {"success": True}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    diag = tr.diagnose_run(str(run_dir))
    assert diag.status == tr.RUN_STATUS_COMPLETED


def test_diagnose_run_marks_training_complete_test_pending_without_test_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run2_pending"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"status": {"training": {"success": True}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    diag = tr.diagnose_run(str(run_dir))
    assert diag.status == tr.RUN_STATUS_TRAINING_COMPLETE_TEST_PENDING
    assert "missing_test_dir" in diag.reasons


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


def test_run_discovery_finds_run_without_metadata_by_train_artifacts(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds" / "run_without_metadata"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "args.yaml").write_text("epochs: 20\n", encoding="utf-8")
    (run_dir / "train" / "weights" / "last.pt").write_text("bin", encoding="utf-8")

    runs = rd.find_run_directories(str(tmp_path / "runs"))
    assert str(run_dir) in runs
    assert rd.is_run_directory(str(run_dir)) is True


def test_resume_failed_before_epoch_stays_resumable_and_tracks_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds" / "run_failed_resume"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "args.yaml").write_text("epochs: 30\n", encoding="utf-8")
    (run_dir / "train" / "weights" / "last.pt").write_text("bin", encoding="utf-8")

    def _raise_resume(_: str) -> None:
        raise RuntimeError("ssh disconnected")

    monkeypatch.setattr(mtm, "resume_training_in_run", _raise_resume)
    code = mtm._run_resume_command(
        ["--workspace", str(tmp_path), "--run-dir", str(run_dir), "-y"]
    )
    assert code == 1

    diag = tr.diagnose_run(str(run_dir))
    assert diag.status == tr.RUN_STATUS_RESUMABLE_INCOMPLETE

    metadata = json.loads((run_dir / "training_metadata.json").read_text(encoding="utf-8"))
    attempts = metadata.get("resume_attempts")
    assert isinstance(attempts, list) and attempts
    assert attempts[-1]["success"] is False
    assert "ssh disconnected" in str(attempts[-1]["error"])


def test_resume_discover_runs_matches_run_discovery(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds" / "run_discover_consistency"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "results.csv").write_text("epoch,loss\n1,1.0\n", encoding="utf-8")

    resume_runs = tr.discover_runs(str(tmp_path))
    generic_runs = rd.find_run_directories(str(tmp_path / "runs"))
    assert str(run_dir) in resume_runs
    assert str(run_dir) in generic_runs


def test_resume_runs_test_stage_when_training_complete_but_test_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds" / "run_test_pending"
    run_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "data.yaml").write_text("path: .\ntrain: train/images\nval: val/images\n", encoding="utf-8")
    (run_dir / "_runtime_data_train.yaml").write_text(f"path: {tmp_path}\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"status": {"training": {"success": True}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    called = {"test": False, "meta": False}

    def _fake_test(model_dir: str, dataset_path: str, **_: object) -> tuple[None, None, dict]:
        assert model_dir == str(run_dir)
        assert dataset_path == str(tmp_path)
        called["test"] = True
        return None, None, {}

    def _fake_test_meta(path: str, *, success: bool, error: str | None, diagnosis: tr.RunDiagnosis | None) -> None:
        assert path == str(run_dir)
        assert success is True
        assert error is None
        assert diagnosis is not None
        called["meta"] = True

    monkeypatch.setattr(mtm, "test_yolo", _fake_test)
    monkeypatch.setattr(mtm, "update_resume_test_metadata", _fake_test_meta)
    monkeypatch.setattr(mtm, "_maybe_free_cuda_memory", lambda: None)

    code = mtm._run_resume_command(
        ["--workspace", str(tmp_path), "--run-dir", str(run_dir), "-y"]
    )
    assert code == 0
    assert called["test"] is True
    assert called["meta"] is True


def test_resolve_dataset_path_for_resume_falls_back_to_workspace_dataset_dir(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    dataset_name = "my_dataset"
    dataset_dir = tmp_path / "datasets" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data.yaml").write_text("path: .\ntrain: train/images\nval: val/images\n", encoding="utf-8")

    run_dir = tmp_path / "runs" / dataset_name / "run_from_other_machine"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"status": {"training": {"success": True}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    resolved = tr.resolve_dataset_path_for_resume(str(run_dir), str(tmp_path))
    assert resolved == str(dataset_dir)


def test_resolve_dataset_path_for_resume_uses_workspace_relative_metadata_path(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    dataset_dir = tmp_path / "datasets" / "ds_rel"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data.yaml").write_text("path: .\ntrain: train/images\nval: val/images\n", encoding="utf-8")

    run_dir = tmp_path / "runs" / "random_parent" / "run_rel"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_info": {"dataset": {"path_under_workspace": "datasets/ds_rel"}},
                "status": {"training": {"success": True}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    resolved = tr.resolve_dataset_path_for_resume(str(run_dir), str(tmp_path))
    assert resolved == str(dataset_dir)


def test_update_resume_metadata_hydrates_training_info_when_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "dataset_hydrate" / "run_hydrate"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "args.yaml").write_text("model: /old/path/yolo11x.pt\n", encoding="utf-8")
    (run_dir / "_runtime_data_train.yaml").write_text(f"path: {tmp_path}\n", encoding="utf-8")
    (tmp_path / "data.yaml").write_text("path: .\ntrain: train/images\nval: val/images\n", encoding="utf-8")

    tr.update_resume_metadata(str(run_dir), success=False, error="x", diagnosis=tr.diagnose_run(str(run_dir)))
    metadata = json.loads((run_dir / "training_metadata.json").read_text(encoding="utf-8"))

    assert metadata.get("training_info", {}).get("model") == "yolo11x"
    assert metadata.get("training_info", {}).get("dataset", {}).get("name") == "dataset_hydrate"
