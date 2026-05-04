"""Importing package submodules."""

from __future__ import annotations


def test_import_package() -> None:
    import smartrain

    assert smartrain.__version__


def test_import_cli_modules() -> None:
    for mod in (
        "smartrain.workspace_paths",
        "smartrain.datasets_json_former",
        "smartrain.dataset_former",
        "smartrain.dataset_hash",
        "smartrain.dataset_roi_yolo",
        "smartrain.model_training_module",
        "smartrain.training_queue",
        "smartrain.training_queue_cli",
        "smartrain.registry_cli",
        "smartrain.run_bundle_copy",
        "smartrain.results_analyzer",
        "smartrain.analyze_models",
        "smartrain.run_discovery",
        "smartrain.metrics_reader",
        "smartrain.compare_service",
        "smartrain.analyze_report",
        "smartrain.analyze_cache",
        "smartrain.plot_creator",
        "smartrain.cli_argparse",
        "smartrain.cvat11_converter",
        "smartrain.cvat_cli",
        "smartrain.train_profile",
        "smartrain.train_backend_registry",
        "smartrain.train_model_catalog",
        "smartrain.train_model_resolver",
        "smartrain.external_model_ref",
        "smartrain.providers_cli",
        "smartrain.provider_global_index",
        "smartrain.provider_install_state",
        "smartrain.external_providers.base",
        "smartrain.external_providers.registry",
        "smartrain.external_providers.probe",
        "smartrain.external_providers.runner",
        "smartrain.external_providers.installer",
        "smartrain.external_providers.adapters",
        "smartrain.external_providers.launchers.mfel_train_launcher",
        "smartrain.external_providers.launchers.mfel_infer_launcher",
        "smartrain.external_providers.launchers.mp_train_launcher",
        "smartrain.external_providers.launchers.mp_infer_launcher",
        "smartrain.provider_global_index",
        "smartrain.providers_cli",
        "smartrain.weighted_yolo_dataset",
        "smartrain.clearml_upload",
        "smartrain.sahi_cli",
        "smartrain.heatmap_cli",
        "smartrain.cli",
    ):
        __import__(mod)
