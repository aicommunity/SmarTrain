from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from smartrain.train_model_catalog import TrainModelCatalog


@dataclass(frozen=True)
class TrainModelResolution:
    normalized: str
    is_supported_alias: bool


class TrainModelResolver:
    """
    Resolves a model value from CLI/interactive input.

    Unknown values are allowed to keep compatibility with custom forks and local weights.
    """

    def __init__(self, catalog: TrainModelCatalog | None = None) -> None:
        self.catalog = catalog or TrainModelCatalog()
        self._supported_aliases = set(self.catalog.supported_aliases())

    def resolve(self, value: str | None, *, default_model: str, add_pt_when_missing: bool) -> TrainModelResolution:
        normalized = self._normalize(value, default_model=default_model, add_pt_when_missing=add_pt_when_missing)
        alias = self._strip_pt(normalized)
        return TrainModelResolution(
            normalized=normalized,
            is_supported_alias=alias in self._supported_aliases,
        )

    @staticmethod
    def _normalize(value: str | None, *, default_model: str, add_pt_when_missing: bool) -> str:
        model = str(value or "").strip() or default_model
        if (
            add_pt_when_missing
            and Path(model).suffix == ""
            and "/" not in model
            and "\\" not in model
            and model.lower().startswith("yolo")
        ):
            model = f"{model}.pt"
        return model

    @staticmethod
    def _strip_pt(value: str) -> str:
        v = value.strip()
        if v.lower().endswith(".pt"):
            return v[:-3]
        return v

