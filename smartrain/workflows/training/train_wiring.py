"""Train resume/calc-confidence wiring (workflows layer; may import train_resume)."""

from __future__ import annotations

from smartrain.core.runtime.mpl_runtime import ensure_matplotlib_training_runtime
from smartrain.services.testing.model_test_service import complete_missing_test_artifacts
from smartrain.services.train_runtime_helpers import maybe_free_cuda_memory
from smartrain.services.training import train_cli_callbacks as _tcb
from smartrain.services.training import train_cli_parsers
from smartrain.services.training.train_resume_backoff_service import (
    complete_missing_test_with_backoff as _svc_complete_missing_test_with_backoff,
)
from smartrain.services.training.train_resume_cli_service import (
    run_calc_confidence_command as _svc_run_calc_confidence_command,
    run_resume_command as _svc_run_resume_command,
)
from smartrain.services.training.train_resume_confidence_service import ensure_resume_confidence_recommendations
from smartrain.services.training.train_resume_pt_test_runner import resume_ultralytics_pt_test_runner
from smartrain.services.training.train_runtime_ops import build_train_runtime_ops
from smartrain.workflows.training.train_resume import (
    RUN_STATUS_RESUMABLE_INCOMPLETE,
    RUN_STATUS_TRAINING_COMPLETE_TEST_PENDING,
    diagnose_run,
    list_incomplete_runs,
    resume_training_in_run,
    update_resume_metadata,
    update_resume_test_metadata,
)

parse_train_args_cb = train_cli_parsers.parse_train_args
run_interactive_train_setup_cb = _tcb.run_interactive_train_setup_cb
load_ultralytics_yaml_cb = _tcb.load_ultralytics_yaml_cb
resolve_cli_paths_with_profile_cb = _tcb.resolve_cli_paths_with_profile_cb
normalize_model_spec_cb = _tcb.normalize_model_spec_cb


def _resume_pt_test_runner(*args, **kwargs):
    return resume_ultralytics_pt_test_runner(*args, **kwargs)


def _ensure_resume_confidence_recommendations_cb(run_dir: str, workspace_root: str, val_batch: int = 1) -> None:
    return ensure_resume_confidence_recommendations(run_dir, workspace_root, val_batch=val_batch)


def _update_resume_test_metadata_cb(*args, **kwargs):
    return update_resume_test_metadata(*args, **kwargs)


def _maybe_free_cuda_memory_cb() -> None:
    return maybe_free_cuda_memory()


def _complete_missing_test_artifacts_cb(*args, **kwargs):
    return complete_missing_test_artifacts(*args, **kwargs)


def _list_incomplete_runs_cb(*args, **kwargs):
    return list_incomplete_runs(*args, **kwargs)


def _diagnose_run_cb(*args, **kwargs):
    return diagnose_run(*args, **kwargs)


def _resume_training_in_run_cb(*args, **kwargs):
    return resume_training_in_run(*args, **kwargs)


def _update_resume_metadata_cb(*args, **kwargs):
    return update_resume_metadata(*args, **kwargs)


def complete_missing_test_with_backoff(
    run_dir: str,
    *,
    workspace_root: str,
    initial_batch: int | None,
    min_batch: int,
    backoff: int,
) -> None:
    _svc_complete_missing_test_with_backoff(
        run_dir,
        workspace_root=workspace_root,
        initial_batch=initial_batch,
        min_batch=min_batch,
        backoff=backoff,
        complete_missing_test_artifacts_cb=_complete_missing_test_artifacts_cb,
        pt_test_runner_cb=_resume_pt_test_runner,
        update_metadata_cb=_update_resume_test_metadata_cb,
        maybe_free_cuda_memory_cb=_maybe_free_cuda_memory_cb,
    )


def run_calc_confidence_command(argv: list[str]) -> int:
    return _svc_run_calc_confidence_command(
        argv,
        ensure_resume_confidence_recommendations_cb=_ensure_resume_confidence_recommendations_cb,
    )


def run_train_after_setup_cb(**kwargs):
    from smartrain.services.train_service import run_train_after_setup

    return run_train_after_setup(**kwargs, runtime_ops=build_train_runtime_ops())


def run_resume_command(argv: list[str]) -> int:
    return _svc_run_resume_command(
        argv,
        run_status_resumable_incomplete=RUN_STATUS_RESUMABLE_INCOMPLETE,
        run_status_training_complete_test_pending=RUN_STATUS_TRAINING_COMPLETE_TEST_PENDING,
        list_incomplete_runs_cb=_list_incomplete_runs_cb,
        diagnose_run_cb=_diagnose_run_cb,
        resume_training_in_run_cb=_resume_training_in_run_cb,
        update_resume_metadata_cb=_update_resume_metadata_cb,
        update_resume_test_metadata_cb=_update_resume_test_metadata_cb,
        complete_missing_test_with_backoff_cb=complete_missing_test_with_backoff,
        ensure_resume_confidence_recommendations_cb=_ensure_resume_confidence_recommendations_cb,
        ensure_matplotlib_training_runtime_cb=ensure_matplotlib_training_runtime,
        maybe_free_cuda_memory_cb=_maybe_free_cuda_memory_cb,
    )
