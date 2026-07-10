"""Shared dataset catalog loading and interactive dataset selection for dataset CLI commands."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import WorkspaceLayout

from smartrain.services.datasets.dataset_cli_common import (
    load_dataset_catalog,
    sorted_class_names_for_dataset,
    sorted_class_names_union_from_catalog,
)

# Message text must stay stable for UX/tests.
EMPTY_DATASETS_INFO_MESSAGE = "[ERROR] datasets_info.json was not found or is empty."


def load_datasets_catalog(layout: WorkspaceLayout) -> dict[str, Any]:
    """Load ``datasets_info.json`` for the workspace layout (alias of ``load_dataset_catalog``)."""
    return load_dataset_catalog(layout)


def try_prompt_dataset_interactive(
    *,
    args: Any,
    argv: list[str],
    fill: Callable[[], None],
) -> bool:
    """
    If ``args.dataset`` is unset, interactive mode is allowed, and stdin is a TTY, run ``fill()``
    (which should mutate ``args``). Returns whether the interactive path ran.
    """
    if getattr(args, "dataset", None) is not None:
        return False
    if not is_interactive_allowed(argv) or not sys.stdin.isatty():
        return False
    fill()
    return True


__all__ = [
    "EMPTY_DATASETS_INFO_MESSAGE",
    "load_datasets_catalog",
    "sorted_class_names_for_dataset",
    "sorted_class_names_union_from_catalog",
    "try_prompt_dataset_interactive",
]
