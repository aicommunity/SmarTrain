from __future__ import annotations

import pytest

from smartrain.domain.canonical.errors import CanonicalCompatibilityError, CanonicalValidationError
from smartrain.domain.canonical.validators import validate_schema_version


def test_schema_version_accepts_semver() -> None:
    validate_schema_version("2.1.0")


def test_schema_version_rejects_invalid_format() -> None:
    with pytest.raises(CanonicalValidationError):
        validate_schema_version("v2")


def test_schema_version_rejects_unsupported_major() -> None:
    with pytest.raises(CanonicalCompatibilityError):
        validate_schema_version("99.0.0")

