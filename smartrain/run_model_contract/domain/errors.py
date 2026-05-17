from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UnifiedErrorDetails:
    error_code: str
    entity: str
    field: str
    hint: str | None = None


class UnifiedError(Exception):
    def __init__(self, details: UnifiedErrorDetails, message: str) -> None:
        super().__init__(message)
        self.details = details


class UnifiedValidationError(UnifiedError):
    pass


class UnifiedCompatibilityError(UnifiedError):
    pass


class UnifiedMappingError(UnifiedError):
    pass

