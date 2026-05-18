from __future__ import annotations

import ast
from pathlib import Path


NO_HASATTR_TARGETS = (
    "smartrain/services/inference_service.py",
    "smartrain/services/train_service.py",
    "smartrain/external_providers/launchers/mp_infer_launcher.py",
    "smartrain/external_providers/launchers/mfel_infer_launcher.py",
    "smartrain/external_providers/launchers/mfel_val_launcher.py",
    "smartrain/external_providers/launchers/mfel_train_launcher.py",
)

TRAIN_SERVICE_PATH = Path("smartrain/services/train_service.py")
TRAIN_COMPOSITION_ROOT = "run_train_after_setup"


def _parse(path: str | Path) -> ast.AST:
    source = Path(path).read_text(encoding="utf-8")
    return ast.parse(source)


def _check_no_hasattr() -> list[str]:
    violations: list[str] = []
    for rel_path in NO_HASATTR_TARGETS:
        tree = _parse(rel_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hasattr":
                violations.append(f"{rel_path}:{getattr(node, 'lineno', '?')}")
    return violations


def _check_train_service_mtm_scope() -> list[str]:
    tree = _parse(TRAIN_SERVICE_PATH)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id != "mtm":
            continue
        parent_func = None
        for parent in ast.walk(tree):
            if isinstance(parent, ast.FunctionDef) and any(child is node for child in ast.walk(parent)):
                parent_func = parent.name
                break
        if parent_func != TRAIN_COMPOSITION_ROOT:
            violations.append(f"{TRAIN_SERVICE_PATH}:{getattr(node, 'lineno', '?')} ({node.attr})")
    return violations


def main() -> int:
    failed = False

    hasattr_violations = _check_no_hasattr()
    if hasattr_violations:
        failed = True
        print("[ERROR] Forbidden hasattr() in runtime-critical modules:")
        for item in hasattr_violations:
            print(f"  - {item}")

    mtm_violations = _check_train_service_mtm_scope()
    if mtm_violations:
        failed = True
        print("[ERROR] Direct mtm.* usage outside train composition root:")
        for item in mtm_violations:
            print(f"  - {item}")

    if failed:
        return 1

    print("[OK] Phase 8 guardrails passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
