from __future__ import annotations

from typing import Any

from smartrain.services.training.train_yolo_execution_service import TrainYoloHooks


def build_train_yolo_hooks() -> TrainYoloHooks:
    def _setup_weighted_sampling_env() -> None:
        from smartrain.services.datasets.weighted_yolo_dataset import setup_weighted_sampling_env

        setup_weighted_sampling_env()

    def _register_weighted_sampling_callback(model: Any) -> None:
        from smartrain.services.datasets.weighted_yolo_dataset import register_weighted_sampling_callback

        register_weighted_sampling_callback(model)

    return TrainYoloHooks(
        setup_weighted_sampling_env=_setup_weighted_sampling_env,
        register_weighted_sampling_callback=_register_weighted_sampling_callback,
    )
