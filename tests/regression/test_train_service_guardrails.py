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


def test_train_service_does_not_use_private_mtm_getattr_fallbacks() -> None:
    path = Path("smartrain/services/train_service.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "getattr":
            continue
        if len(node.args) < 2:
            continue
        target, attr = node.args[0], node.args[1]
        if not isinstance(target, ast.Name) or target.id != "mtm":
            continue
        if not isinstance(attr, ast.Constant) or not isinstance(attr.value, str):
            continue
        if attr.value.startswith("_"):
            violations.append(f"{attr.value}@{getattr(node, 'lineno', '?')}")
    assert not violations, (
        "Private mtm getattr fallbacks are forbidden in train_service composition "
        f"root, found: {', '.join(violations)}"
    )


def test_service_to_workflows_imports_are_limited_by_transitional_allowlist() -> None:
    violations: list[str] = []
    service_files = sorted(Path("smartrain/services").glob("*.py"))
    for path in service_files:
        rel_path = str(path).replace("\\", "/")
        source = Path(rel_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        seen: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and isinstance(node.module, str):
                if node.module.startswith("smartrain.workflows"):
                    seen.add(node.module)
        if seen:
            violations.append(f"{rel_path}: {', '.join(sorted(seen))}")
    assert not violations, (
        "Direct services->workflows imports are forbidden; route through stable "
        f"adapter/facade modules instead: {'; '.join(violations)}"
    )
