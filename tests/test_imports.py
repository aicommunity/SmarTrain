"""Importing package submodules."""

from __future__ import annotations


def test_import_package() -> None:
    import smartrain

    assert smartrain.__version__


def test_import_cli_modules() -> None:
    for mod in (
        "smartrain.workspace_paths",
        "smartrain.datasets_json_former",
        "smartrain.workflows.datasets.dataset_former",
        "smartrain.workflows.datasets.dataset_hash",
        "smartrain.workflows.datasets.dataset_roi_yolo",
        "smartrain.model_training_module",
        "smartrain.workflows.queue.training_queue",
        "smartrain.workflows.queue.training_queue_cli",
        "smartrain.workflows.registry.registry_cli",
        "smartrain.run_bundle_copy",
        "smartrain.results_analyzer",
        "smartrain.workflows.analyze.analyze_models",
        "smartrain.run_discovery",
        "smartrain.metrics_reader",
        "smartrain.workflows.analyze.compare_service",
        "smartrain.workflows.analyze.analyze_report",
        "smartrain.workflows.analyze.analyze_cache",
        "smartrain.workflows.analyze.plot_creator",
        "smartrain.cli_support.cli_argparse",
        "smartrain.workflows.datasets.cvat11_converter",
        "smartrain.workflows.datasets.cvat_cli",
        "smartrain.core.training.train_profile",
        "smartrain.core.training.train_backend_registry",
        "smartrain.core.training.train_model_catalog",
        "smartrain.core.training.train_model_resolver",
        "smartrain.core.training.external_model_ref",
        "smartrain.providers.cli",
        "smartrain.providers.core.global_index",
        "smartrain.providers.core.provider_install_state",
        "smartrain.external_providers.base",
        "smartrain.external_providers.registry",
        "smartrain.external_providers.probe",
        "smartrain.external_providers.runner",
        "smartrain.external_providers.installer",
        "smartrain.external_providers.adapters",
        "smartrain.core.runtime.environment_profile",
        "smartrain.workflows.inference.inference_perf",
        "smartrain.workflows.testing.unified_validator_core",
        "smartrain.workflows.testing.unified_metrics_adapter",
        "smartrain.external_providers.launchers.mfel_train_launcher",
        "smartrain.external_providers.launchers.mfel_infer_launcher",
        "smartrain.external_providers.launchers.mp_train_launcher",
        "smartrain.external_providers.launchers.mp_infer_launcher",
        "smartrain.providers.core.global_index",
        "smartrain.providers.cli",
        "smartrain.workflows.datasets.weighted_yolo_dataset",
        "smartrain.workflows.analyze.clearml_upload",
        "smartrain.workflows.inference.sahi_cli",
        "smartrain.workflows.inference.heatmap_cli",
        "smartrain.cli",
    ):
        __import__(mod)
