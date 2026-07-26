"""CLI facade: implementation in smartrain.services.models.model_convert_service."""

from __future__ import annotations

from typing import Any

from smartrain.services.models import model_convert_service as _impl

build_model_convert_arg_parser = _impl.build_model_convert_arg_parser
main = _impl.main


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


if __name__ == "__main__":
    main()
