"""Train resume/calc-confidence wiring (workflows layer; may import train_resume)."""

from __future__ import annotations

from smartrain.core.runtime.mpl_runtime import ensure_matplotlib_training_runtime
from smartrain.services.training.train_resume_backoff_service import (
    complete_missing_test_with_backoff as _svc_complete_missing_test_with_backoff,
)
from smartrain.services.training.train_resume_cli_service import (
    run_calc_confidence_command as _svc_run_calc_confidence_command,
    run_resume_command as _svc_run_resume_command,
)
from smartrain.services.training.train_runtime_ops import TrainRuntimeOps, build_train_runtime_ops

def _resume_pt_test_runner(*args, **kwargs):
    from smartrain.workflows.training import model_training_module as mtm

    return mtm._resume_ultralytics_pt_test_runner(*args, **kwargs)


def _ensure_resume_confidence_recommendations_cb(run_dir: str, workspace_root: str, val_batch: int = 1) -> None:
    from smartrain.workflows.training import model_training_module as mtm

    return mtm._ensure_resume_confidence_recommendations(run_dir, workspace_root, val_batch=val_batch)


def _update_resume_test_metadata_cb(*args, **kwargs):
    from smartrain.workflows.training import model_training_module as mtm

    return mtm.update_resume_test_metadata(*args, **kwargs)


def _maybe_free_cuda_memory_cb() -> None:
    from smartrain.workflows.training import model_training_module as mtm

    return mtm._maybe_free_cuda_memory()


def _complete_missing_test_artifacts_cb(*args, **kwargs):
    from smartrain.workflows.training import model_training_module as mtm

    return mtm.complete_missing_test_artifacts(*args, **kwargs)


def _list_incomplete_runs_cb(*args, **kwargs):
    from smartrain.workflows.training import model_training_module as mtm

    return mtm.list_incomplete_runs(*args, **kwargs)


def _diagnose_run_cb(*args, **kwargs):
    from smartrain.workflows.training import model_training_module as mtm

    return mtm.diagnose_run(*args, **kwargs)


def _resume_training_in_run_cb(*args, **kwargs):
    from smartrain.workflows.training import model_training_module as mtm

    return mtm.resume_training_in_run(*args, **kwargs)


def _update_resume_metadata_cb(*args, **kwargs):
    from smartrain.workflows.training import model_training_module as mtm

    return mtm.update_resume_metadata(*args, **kwargs)


from smartrain.workflows.training.train_resume import (
    RUN_STATUS_RESUMABLE_INCOMPLETE,
    RUN_STATUS_TRAINING_COMPLETE_TEST_PENDING,
)


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


def build_train_runtime_ops_from_mtm() -> TrainRuntimeOps:
    from smartrain.workflows.training import model_training_module as mtm

    base = build_train_runtime_ops()
    return TrainRuntimeOps(
        train_yolo=mtm.train_yolo,
        test_yolo=mtm.test_yolo,
        save_training_metadata=mtm.save_training_metadata,
        collect_system_profile=base.collect_system_profile,
        build_run_name=mtm._build_run_name,
        resolve_external_eval_source=base.resolve_external_eval_source,
        json_safe_train_summary=base.json_safe_train_summary,
        load_batch_from_training_metadata=base.load_batch_from_training_metadata,
        run_external_train=mtm.run_external_train,
        run_external_infer=mtm.run_external_infer,
    )


def parse_train_args_cb(argv: list[str]):
    from smartrain.workflows.training import model_training_module as mtm

    return mtm.parse_args(argv)


def run_interactive_train_setup_cb(args) -> bool:
    from smartrain.workflows.training import model_training_module as mtm

    return mtm._run_interactive_train_setup(args)


def resolve_cli_paths_with_profile_cb(args, u_cfg: dict):
    from smartrain.workflows.training import model_training_module as mtm

    return mtm._resolve_cli_paths_with_profile(args, u_cfg)


def load_ultralytics_yaml_cb(path: str | None):
    from smartrain.workflows.training import model_training_module as mtm

    return mtm._load_ultralytics_yaml(path)


def normalize_model_spec_cb(spec, *, add_pt_when_missing: bool = False) -> str:
    from smartrain.workflows.training import model_training_module as mtm

    return mtm._normalize_model_spec(spec, add_pt_when_missing=add_pt_when_missing)


def run_train_after_setup_cb(**kwargs):
    from smartrain.services.train_service import run_train_after_setup

    return run_train_after_setup(**kwargs, runtime_ops=build_train_runtime_ops_from_mtm())


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
