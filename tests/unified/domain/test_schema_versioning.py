from __future__ import annotations

import pytest

from smartrain.unified.domain.errors import UnifiedCompatibilityError, UnifiedValidationError
from smartrain.unified.domain.validators import validate_schema_version


def test_schema_version_accepts_semver() -> None:
    validate_schema_version("2.1.0")


def test_schema_version_rejects_invalid_format() -> None:
    with pytest.raises(UnifiedValidationError):
        validate_schema_version("v2")


def test_schema_version_rejects_unsupported_major() -> None:
    with pytest.raises(UnifiedCompatibilityError):
        validate_schema_version("99.0.0")

