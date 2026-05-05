from __future__ import annotations

import ast
from pathlib import Path


def test_train_service_mtm_usage_is_confined_to_composition_root() -> None:
    path = Path("smartrain/services/train_service.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    allowed_function = "run_train_after_setup"
    violations: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id != "mtm":
            continue
        # Find enclosing function for this usage.
        parent_func = None
        for parent in ast.walk(tree):
            if isinstance(parent, ast.FunctionDef) and any(child is node for child in ast.walk(parent)):
                parent_func = parent.name
                break
        if parent_func != allowed_function:
            violations.append(f"{node.attr}@{getattr(node, 'lineno', '?')}")

    assert not violations, (
        "Direct mtm.* usage is only allowed in composition root "
        f"'{allowed_function}', found: {', '.join(violations)}"
    )
