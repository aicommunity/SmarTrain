from __future__ import annotations

from dataclasses import dataclass

from smartrain.backends.contracts import BackendCapabilities


@dataclass(frozen=True)
class BackendEntry:
    capabilities: BackendCapabilities


class CapabilityRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, BackendEntry] = {}

    def register(self, capabilities: BackendCapabilities) -> None:
        key = capabilities.backend.strip().lower()
        self._entries[key] = BackendEntry(capabilities=capabilities)

    def resolve(self, *, task_type: str, model_format: str, require: str) -> BackendCapabilities:
        req = str(require or "").strip().lower()
        for entry in self._entries.values():
            caps = entry.capabilities
            if not caps.supports(task_type=task_type, model_format=model_format):
                continue
            if req == "train" and caps.can_train:
                return caps
            if req == "test" and caps.can_test:
                return caps
            if req == "infer" and caps.can_infer:
                return caps
        raise ValueError(f"No backend for task={task_type!r}, format={model_format!r}, require={require!r}")

