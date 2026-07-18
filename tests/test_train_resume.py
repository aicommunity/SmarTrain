from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from smartrain.core.runtime import run_discovery as rd
from smartrain.workflows.training import train_entry
from smartrain.workflows.training import train_resume as tr
from smartrain.workflows.training import train_wiring
from smartrain.core.training.confidence_recommendation import write_not_available_recommendations
from smartrain.core.runtime.workspace_paths import deploy_workspace
from smartrain.workflows.testing.ultralytics_test_contract import ultralytics_pt_rich_files_required


def _touch_missing_ultralytics_pt_rich_files(test_dir: Path, *, task_type: str | None = None) -> None:
    """Stub files so has_complete_test_artifacts(run_dir, \"pt\") passes in unit tests."""
    for name in ultralytics_pt_rich_files_required(task_type):
        path = test_dir / name
        if path.is_file():
            continue
        path.write_bytes(b"stub")


def _write_resumable_last_pt(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 0, "optimizer": {"state": {}, "param_groups": []}}, path)


def _write_stripped_last_pt(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 299, "model": None}, path)


def test_diagnose_run_marks_resumable_when_last_checkpoint_exists(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run1"
    (run_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "args.yaml").write_text("epochs: 100\n", encoding="utf-8")
    _write_resumable_last_pt(run_dir / "train" / "weights" / "last.pt")

    diag = tr.diagnose_run(str(run_dir))
    assert diag.status == tr.RUN_STATUS_RESUMABLE_INCOMPLETE
    assert "resume_checkpoint_available" in diag.reasons


def test_diagnose_run_marks_completed_from_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run2"
    (run_dir / "test").mkdir(parents=True, exist_ok=True)
    (run_dir / "test" / "args.yaml").write_text("name: test\n", encoding="utf-8")
    (run_dir / "test" / "pr.csv").write_text("recall,precision\n0.0,1.0\n", encoding="utf-8")
    (run_dir / "test" / "pr_per_class.csv").write_text("class_name,ap\nobj,0.5\n", encoding="utf-8")
    _touch_missing_ultralytics_pt_rich_files(run_dir / "test")
    (run_dir / "test_metrics.csv").write_text("mAP50-95\n0.5\n", encoding="utf-8")
    write_not_available_recommendations(model_dir=str(run_dir), split="test", reason="stub")
    write_not_available_recommendations(model_dir=str(run_dir), split="val", reason="stub")
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
    assert "missing_test_artifacts" in diag.reasons


def test_diagnose_run_marks_pending_when_test_dir_exists_but_artifacts_incomplete(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run2_partial"
    (run_dir / "test").mkdir(parents=True, exist_ok=True)
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"status": {"training": {"success": True}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    diag = tr.diagnose_run(str(run_dir))
    assert diag.status == tr.RUN_STATUS_TRAINING_COMPLETE_TEST_PENDING
    assert "missing_metrics_csv" in diag.reasons


def test_diagnose_run_resumable_when_train_ultralytics_last_exists(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run_ultra_last"
    (run_dir / "train-ultralytics" / "weights").mkdir(parents=True)
    _write_resumable_last_pt(run_dir / "train-ultralytics" / "weights" / "last.pt")
    (run_dir / "train-ultralytics" / "args.yaml").write_text("epochs: 30\n", encoding="utf-8")

    diag = tr.diagnose_run(str(run_dir))
    assert diag.status == tr.RUN_STATUS_RESUMABLE_INCOMPLETE
    assert "resume_checkpoint_available" in diag.reasons


def test_diagnose_run_stripped_last_not_resumable_incomplete(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run_stripped"
    (run_dir / "train-ultralytics" / "weights").mkdir(parents=True)
    _write_stripped_last_pt(run_dir / "train-ultralytics" / "weights" / "last.pt")
    (run_dir / "train-ultralytics" / "args.yaml").write_text("epochs: 30\n", encoding="utf-8")

    diag = tr.diagnose_run(str(run_dir))
    assert diag.status == tr.RUN_STATUS_INCOMPLETE_NON_RESUMABLE
    assert "last_pt_present_but_stripped" in diag.reasons
    assert diag.has_last_pt is False


def test_diagnose_run_training_complete_pending_when_results_in_train_ultralytics_suffix_dir(
    tmp_path: Path,
) -> None:
    """Ultralytics may write metrics under train-ultralytics-2/ when the default name is taken."""
    run_dir = tmp_path / "runs" / "ds" / "run_suffix_csv"
    run_dir.mkdir(parents=True)
    (run_dir / "train-ultralytics" / "weights").mkdir(parents=True)
    _write_stripped_last_pt(run_dir / "train-ultralytics" / "weights" / "last.pt")
    (run_dir / "train-ultralytics-2" / "weights").mkdir(parents=True)
    (run_dir / "train-ultralytics-2" / "weights" / "best.pt").write_text("x", encoding="utf-8")
    (run_dir / "train-ultralytics-2" / "args.yaml").write_text("epochs: 3\n", encoding="utf-8")
    (run_dir / "train-ultralytics-2" / "results.csv").write_text(
        "epoch,mAP\n1,0.1\n2,0.2\n3,0.3\n", encoding="utf-8"
    )
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"status": {"training": {"success": False}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    diag = tr.diagnose_run(str(run_dir))
    assert diag.status == tr.RUN_STATUS_TRAINING_COMPLETE_TEST_PENDING
    assert "finalize_after_training_error_heuristic" in diag.reasons
    assert "results_csv_present" in diag.reasons


def test_canonicalize_resolves_suffix_train_results_csv(tmp_path: Path) -> None:
    from smartrain.core.runtime.run_artifacts import normalize_ultralytics_run_layout
    from smartrain.services.analyze.metrics_reader import results_csv_path

    run_dir = tmp_path / "runs" / "ds" / "run_suffix_canonicalize"
    run_dir.mkdir(parents=True)
    (run_dir / "train-ultralytics").mkdir()
    (run_dir / "train-ultralytics-2" / "weights").mkdir(parents=True)
    (run_dir / "train-ultralytics-2" / "results.csv").write_text("epoch,mAP\n1,0.1\n", encoding="utf-8")

    normalize_ultralytics_run_layout(str(run_dir))
    csv = results_csv_path(str(run_dir))
    assert csv is not None
    norm = csv.replace("\\", "/")
    assert norm.endswith("train-ultralytics/results.csv")


def test_diagnose_run_finalize_no_metadata_stripped_last_epochs_match(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run_no_meta_done"
    run_dir.mkdir(parents=True)
    (run_dir / "train-ultralytics" / "weights").mkdir(parents=True)
    _write_stripped_last_pt(run_dir / "train-ultralytics" / "weights" / "last.pt")
    (run_dir / "train-ultralytics" / "weights" / "best.pt").write_text("x", encoding="utf-8")
    (run_dir / "train-ultralytics" / "args.yaml").write_text("epochs: 3\n", encoding="utf-8")
    (run_dir / "train-ultralytics" / "results.csv").write_text(
        "epoch,mAP\n1,0.1\n2,0.2\n3,0.3\n", encoding="utf-8"
    )

    diag = tr.diagnose_run(str(run_dir))
    assert diag.status == tr.RUN_STATUS_TRAINING_COMPLETE_TEST_PENDING
    assert "finalize_stripped_no_metadata_epochs_ok" in diag.reasons


def test_diagnose_warns_archive_like_path_segment(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    run_dir = tmp_path / "nest" / "archive2.tar.gz" / "2026_run"
    run_dir.mkdir(parents=True)
    tr.diagnose_run(str(run_dir))
    assert "archive-like" in capsys.readouterr().out


def test_diagnose_run_finalize_pending_when_training_failed_but_weights_and_results_exist(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run_finalize_pending"
    run_dir.mkdir(parents=True)
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"status": {"training": {"success": False}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "train-ultralytics" / "weights").mkdir(parents=True)
    (run_dir / "train-ultralytics" / "weights" / "best.pt").write_text("bin", encoding="utf-8")
    (run_dir / "train-ultralytics" / "results.csv").write_text("epoch,mAP\n1,0.5\n", encoding="utf-8")

    diag = tr.diagnose_run(str(run_dir))
    assert diag.status == tr.RUN_STATUS_TRAINING_COMPLETE_TEST_PENDING
    assert "finalize_after_training_error_heuristic" in diag.reasons


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
    code = train_wiring.run_resume_command(["--workspace", str(tmp_path), "-y"])
    assert code == 2


def test_resume_run_dir_fails_for_nonresumable_run(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    run_dir = tmp_path / "runs" / "ds" / "run4"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "args.yaml").write_text("epochs: 30\n", encoding="utf-8")

    code = train_wiring.run_resume_command(
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
    _write_resumable_last_pt(run_dir / "train" / "weights" / "last.pt")

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

    monkeypatch.setattr(train_wiring, "_resume_training_in_run_cb", _fake_resume)
    monkeypatch.setattr(train_wiring, "_update_resume_metadata_cb", _fake_meta)

    code = train_wiring.run_resume_command(
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
    _write_resumable_last_pt(run_dir / "train" / "weights" / "last.pt")

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
    _write_resumable_last_pt(run_dir / "train" / "weights" / "last.pt")

    def _raise_resume(_: str) -> None:
        raise RuntimeError("ssh disconnected")

    monkeypatch.setattr(train_wiring, "_resume_training_in_run_cb", _raise_resume)
    code = train_wiring.run_resume_command(
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

    monkeypatch.setattr(train_wiring, "_resume_pt_test_runner", _fake_test)
    monkeypatch.setattr(train_wiring, "_update_resume_test_metadata_cb", _fake_test_meta)
    monkeypatch.setattr(train_wiring, "_maybe_free_cuda_memory_cb", lambda: None)

    code = train_wiring.run_resume_command(
        ["--workspace", str(tmp_path), "--run-dir", str(run_dir), "-y"]
    )
    assert code == 0
    assert called["test"] is True
    assert called["meta"] is True


def test_resume_test_backoff_retries_on_cuda_oom(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run_backoff_ok"
    run_dir.mkdir(parents=True, exist_ok=True)
    batches: list[int] = []
    freed = {"count": 0}

    def _fake_complete(*_args, **kwargs: object) -> bool:
        runner_kwargs = kwargs.get("pt_test_runner_kwargs")
        assert isinstance(runner_kwargs, dict)
        b = int(runner_kwargs.get("val_batch", -1))
        batches.append(b)
        if len(batches) == 1:
            raise RuntimeError("CUDA out of memory while testing")
        return True

    monkeypatch.setattr(train_wiring, "_complete_missing_test_artifacts_cb", _fake_complete)
    monkeypatch.setattr(
        train_wiring, "_maybe_free_cuda_memory_cb", lambda: freed.__setitem__("count", freed["count"] + 1)
    )

    train_wiring.complete_missing_test_with_backoff(
        str(run_dir),
        workspace_root=str(tmp_path),
        initial_batch=4,
        min_batch=1,
        backoff=2,
    )
    assert batches == [4, 2]
    assert freed["count"] == 1


def test_resume_test_backoff_fails_when_min_batch_oom(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run_backoff_fail"
    run_dir.mkdir(parents=True, exist_ok=True)
    batches: list[int] = []

    def _fake_complete(*_args, **kwargs: object) -> bool:
        runner_kwargs = kwargs.get("pt_test_runner_kwargs")
        assert isinstance(runner_kwargs, dict)
        batches.append(int(runner_kwargs.get("val_batch", -1)))
        raise RuntimeError("CUDA out of memory during validation")

    monkeypatch.setattr(train_wiring, "_complete_missing_test_artifacts_cb", _fake_complete)
    monkeypatch.setattr(train_wiring, "_maybe_free_cuda_memory_cb", lambda: None)

    with pytest.raises(RuntimeError, match="backoff exhausted"):
        train_wiring.complete_missing_test_with_backoff(
            str(run_dir),
            workspace_root=str(tmp_path),
            initial_batch=2,
            min_batch=1,
            backoff=2,
        )
    assert batches == [2, 1]


def test_resume_test_backoff_does_not_retry_non_oom(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ds" / "run_backoff_non_oom"
    run_dir.mkdir(parents=True, exist_ok=True)
    calls = {"count": 0}

    def _fake_complete(*_args, **_kwargs) -> bool:
        calls["count"] += 1
        raise RuntimeError("dataset yaml malformed")

    monkeypatch.setattr(train_wiring, "_complete_missing_test_artifacts_cb", _fake_complete)
    monkeypatch.setattr(train_wiring, "_maybe_free_cuda_memory_cb", lambda: None)

    with pytest.raises(RuntimeError, match="dataset yaml malformed"):
        train_wiring.complete_missing_test_with_backoff(
            str(run_dir),
            workspace_root=str(tmp_path),
            initial_batch=4,
            min_batch=1,
            backoff=2,
        )
    assert calls["count"] == 1


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


def test_resolve_dataset_path_for_resume_uses_train_args_data_yaml_path(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    dataset_dir = tmp_path / "datasets" / "from_args_yaml"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "data.yaml").write_text("path: .\ntrain: train/images\nval: val/images\n", encoding="utf-8")

    run_dir = tmp_path / "runs" / "some_ds" / "run_via_args"
    run_dir.mkdir(parents=True)
    (run_dir / "train-ultralytics").mkdir(parents=True)
    (run_dir / "train-ultralytics" / "args.yaml").write_text(
        f"data: {dataset_dir / 'data.yaml'}\n",
        encoding="utf-8",
    )

    resolved = tr.resolve_dataset_path_for_resume(str(run_dir), str(tmp_path))
    assert resolved == str(dataset_dir.resolve())


def test_resolve_dataset_path_for_resume_uses_train_args_data_relative_to_workspace(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    dataset_dir = tmp_path / "datasets" / "rel_via_args"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "data.yaml").write_text("path: .\ntrain: train/images\nval: val/images\n", encoding="utf-8")

    run_dir = tmp_path / "runs" / "some_ds" / "run_rel_args"
    run_dir.mkdir(parents=True)
    (run_dir / "train-ultralytics").mkdir(parents=True)
    (run_dir / "train-ultralytics" / "args.yaml").write_text(
        "data: datasets/rel_via_args\n",
        encoding="utf-8",
    )

    resolved = tr.resolve_dataset_path_for_resume(str(run_dir), str(tmp_path))
    assert resolved == str(dataset_dir.resolve())


def test_resolve_dataset_path_for_resume_uses_runtime_yaml_pointer_from_train_args(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    dataset_dir = tmp_path / "datasets" / "via_runtime_pointer"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "data.yaml").write_text("path: .\ntrain: train/images\nval: val/images\n", encoding="utf-8")

    run_dir = tmp_path / "runs" / "some_ds" / "run_runtime_pointer"
    run_dir.mkdir(parents=True)
    (run_dir / "train-ultralytics").mkdir(parents=True)
    runtime_yaml = run_dir / "tmp" / "_runtime_data_train.yaml"
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    runtime_yaml.write_text(
        f"path: {dataset_dir}\ntrain: train/images\nval: val/images\n",
        encoding="utf-8",
    )
    (run_dir / "train-ultralytics" / "args.yaml").write_text(
        f"data: {runtime_yaml}\n",
        encoding="utf-8",
    )

    resolved = tr.resolve_dataset_path_for_resume(str(run_dir), str(tmp_path))
    assert resolved == str(dataset_dir.resolve())


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
    assert resolved == str(dataset_dir.resolve())


def test_load_dataset_from_runtime_yaml_relative_path(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    dataset_dir = tmp_path / "datasets" / "foo"
    dataset_dir.mkdir(parents=True)
    run_dir = tmp_path / "runs" / "ds" / "r1"
    (run_dir / "tmp").mkdir(parents=True)
    (run_dir / "tmp" / "_runtime_data_train.yaml").write_text(
        "path: datasets/foo\ntrain: train/images\n",
        encoding="utf-8",
    )
    got = tr._load_dataset_from_runtime_yaml(str(run_dir), str(tmp_path))
    assert got == str(dataset_dir.resolve())


def test_load_dataset_from_runtime_yaml_legacy_abs_path(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))
    dataset_dir = tmp_path / "datasets" / "legacy"
    dataset_dir.mkdir(parents=True)
    run_dir = tmp_path / "runs" / "ds" / "r2"
    (run_dir / "tmp").mkdir(parents=True)
    (run_dir / "tmp" / "_runtime_data_train.yaml").write_text(
        f"path: {dataset_dir.resolve().as_posix()}\ntrain: train/images\n",
        encoding="utf-8",
    )
    got = tr._load_dataset_from_runtime_yaml(str(run_dir), str(tmp_path))
    assert Path(got).resolve() == dataset_dir.resolve()


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


def test_update_resume_metadata_hydrates_model_from_run_name_when_args_is_last(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / "runs"
        / "dataset_hydrate2"
        / "2026-04-24_19-13_yolo26x_300epochs-eb38791f"
    )
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "args.yaml").write_text(
        "model: /old/path/train/weights/last.pt\n", encoding="utf-8"
    )

    tr.update_resume_metadata(str(run_dir), success=False, error="x", diagnosis=tr.diagnose_run(str(run_dir)))
    metadata = json.loads((run_dir / "training_metadata.json").read_text(encoding="utf-8"))

    assert metadata.get("training_info", {}).get("model") == "yolo26x"


def test_update_resume_metadata_replaces_null_model(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "dataset_null" / "2026-04-24_19-13_yolo11x_300epochs-abcd"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "args.yaml").write_text("model: yolo11x.pt\n", encoding="utf-8")
    (run_dir / "training_metadata.json").write_text(
        json.dumps({"training_info": {"model": None, "dataset": {"name": "dataset_null"}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    tr.update_resume_metadata(str(run_dir), success=False, error="x", diagnosis=tr.diagnose_run(str(run_dir)))
    metadata = json.loads((run_dir / "training_metadata.json").read_text(encoding="utf-8"))
    assert metadata.get("training_info", {}).get("model") == "yolo11x"


def test_calc_confidence_non_interactive_processes_all_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deploy_workspace(str(tmp_path))
    run_a = tmp_path / "runs" / "ds" / "run_a"
    run_b = tmp_path / "runs" / "ds" / "run_b"
    (run_a / "train").mkdir(parents=True, exist_ok=True)
    (run_b / "train").mkdir(parents=True, exist_ok=True)
    (run_a / "training_metadata.json").write_text("{}", encoding="utf-8")
    (run_b / "training_metadata.json").write_text("{}", encoding="utf-8")

    called: list[tuple[str, int]] = []

    def _fake_ensure(run_dir: str, workspace_root: str, val_batch: int = 1) -> None:
        assert workspace_root == str(tmp_path)
        called.append((run_dir, val_batch))

    monkeypatch.setattr(train_wiring, "_ensure_resume_confidence_recommendations_cb", _fake_ensure)
    monkeypatch.setattr(
        "smartrain.core.training.confidence_recommendation.recommendations_complete",
        lambda _payload: True,
    )
    monkeypatch.setattr(
        "smartrain.core.training.confidence_recommendation.read_recommendation_file",
        lambda _path: {
            "objectives": {
                "A": {"global": {"threshold": 0.2}},
                "B": {"global": {"threshold": 0.2}},
                "C": {"global": {"threshold": 0.2}},
            }
        },
    )

    rc = train_entry.main(["calc-confidence", "--workspace", str(tmp_path), "-y"])
    assert rc == 0
    assert sorted(called) == sorted([(str(run_a), 1), (str(run_b), 1)])


def test_calc_confidence_non_interactive_with_run_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deploy_workspace(str(tmp_path))
    run_a = tmp_path / "runs" / "ds" / "run_a"
    (run_a / "train").mkdir(parents=True, exist_ok=True)
    (run_a / "training_metadata.json").write_text("{}", encoding="utf-8")

    called: list[tuple[str, int]] = []

    def _fake_ensure(run_dir: str, workspace_root: str, val_batch: int = 1) -> None:
        assert workspace_root == str(tmp_path)
        called.append((run_dir, val_batch))

    monkeypatch.setattr(train_wiring, "_ensure_resume_confidence_recommendations_cb", _fake_ensure)
    monkeypatch.setattr(
        "smartrain.core.training.confidence_recommendation.recommendations_complete",
        lambda _payload: True,
    )
    monkeypatch.setattr(
        "smartrain.core.training.confidence_recommendation.read_recommendation_file",
        lambda _path: {
            "objectives": {
                "A": {"global": {"threshold": 0.2}},
                "B": {"global": {"threshold": 0.2}},
                "C": {"global": {"threshold": 0.2}},
            }
        },
    )

    rc = train_entry.main(["calc-confidence", "--workspace", str(tmp_path), "--run-dir", "ds/run_a", "-y"])
    assert rc == 0
    assert called == [(str(run_a), 1)]


def test_calc_confidence_passes_val_batch_to_recompute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deploy_workspace(str(tmp_path))
    run_a = tmp_path / "runs" / "ds" / "run_a"
    (run_a / "train").mkdir(parents=True, exist_ok=True)
    (run_a / "training_metadata.json").write_text("{}", encoding="utf-8")

    called: list[tuple[str, str, int]] = []

    def _fake_ensure(run_dir: str, workspace_root: str, val_batch: int = 1) -> None:
        called.append((run_dir, workspace_root, val_batch))

    monkeypatch.setattr(train_wiring, "_ensure_resume_confidence_recommendations_cb", _fake_ensure)
    monkeypatch.setattr(
        "smartrain.core.training.confidence_recommendation.recommendations_complete",
        lambda _payload: True,
    )
    monkeypatch.setattr(
        "smartrain.core.training.confidence_recommendation.read_recommendation_file",
        lambda _path: {
            "objectives": {
                "A": {"global": {"threshold": 0.2}},
                "B": {"global": {"threshold": 0.2}},
                "C": {"global": {"threshold": 0.2}},
            }
        },
    )

    rc = train_entry.main(
        ["calc-confidence", "--workspace", str(tmp_path), "--run-dir", "ds/run_a", "--val-batch", "2", "-y"]
    )
    assert rc == 0
    assert called == [(str(run_a), str(tmp_path), 2)]
