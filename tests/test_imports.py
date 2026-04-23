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
        "smartrain.results_analyzer",
        "smartrain.plot_creator",
        "smartrain.cli_argparse",
        "smartrain.cvat11_converter",
        "smartrain.cvat_cli",
        "smartrain.train_profile",
        "smartrain.train_backend_registry",
        "smartrain.train_model_catalog",
        "smartrain.train_model_resolver",
        "smartrain.weighted_yolo_dataset",
        "smartrain.clearml_upload",
        "smartrain.sahi_cli",
        "smartrain.heatmap_cli",
        "smartrain.cli",
    ):
        __import__(mod)
