from __future__ import annotations

from dataclasses import dataclass

from smartrain.train_backend_registry import default_train_provider, get_train_backend_spec

@dataclass(frozen=True)
class TrainModelCatalog:
    """Catalog of model aliases supported by a training backend provider."""

    provider: str = default_train_provider()

    def supported_aliases(self) -> tuple[str, ...]:
        # Keep aliases copy-paste friendly for `--model`.
        return get_train_backend_spec(self.provider).supported_aliases

