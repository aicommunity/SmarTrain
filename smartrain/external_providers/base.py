from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ExternalProviderSpec:
    id: str
    display_name: str
    repo_url: str
    branch: str
    train_entry: str
    infer_entry: str
    requirements_entry: str | None = "requirements.txt"
    ready: bool = True
    note: str | None = None


INSTALL_STATES: Final[tuple[str, ...]] = ("installed", "failed", "removed", "stale")

