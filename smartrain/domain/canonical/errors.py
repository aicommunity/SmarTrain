from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CanonicalErrorDetails:
    error_code: str
    entity: str
    field: str
    hint: str | None = None


class CanonicalError(Exception):
    def __init__(self, details: CanonicalErrorDetails, message: str) -> None:
        super().__init__(message)
        self.details = details


class CanonicalValidationError(CanonicalError):
    pass


class CanonicalCompatibilityError(CanonicalError):
    pass


class CanonicalMappingError(CanonicalError):
    pass

