"""
Weighted image sampling by inverse class frequency (balance when unbalanced).
Monkey-patch ultralytics.data.build.YOLODataset - fragile when changing ultralytics version.
"""
from __future__ import annotations

import importlib.metadata
import warnings

import numpy as np
from ultralytics.data.dataset import YOLODataset


def warn_ultralytics_version() -> None:
    try:
        v = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        v = "unknown"
    warnings.warn(
        f"Weighted dataloader patches the internals of ultralytics (version {v}). "
        "After updating ultralytics, check compatibility.",
        UserWarning,
        stacklevel=2,
    )


class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mode = "train" in self.prefix
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts
        self.agg_func = np.mean
        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        self.counts = [0 for _ in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)
            for i in cls:
                self.counts[i] += 1
        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        weights = []
        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)
            if cls.size == 0:
                weights.append(1)
                continue
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        total_weight = sum(self.weights)
        return [w / total_weight for w in self.weights]

    def __getitem__(self, index):
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        index = int(np.random.choice(len(self.labels), p=self.probabilities))
        return self.transforms(self.get_image_and_label(index))


def _reassert_build_dataset(trainer) -> None:
    import ultralytics.data.build as build

    build.YOLODataset = YOLOWeightedDataset


def setup_weighted_sampling_env() -> None:
    """Call before YOLO() so that the dataset assembly picks up YOLOWeightedDataset."""
    warn_ultralytics_version()
    import ultralytics.data.build as build

    build.YOLODataset = YOLOWeightedDataset


def register_weighted_sampling_callback(model) -> None:
    """After creating YOLO - in case ultralytics resets the dataset class."""
    model.add_callback("on_pretrain_routine_start", _reassert_build_dataset)
