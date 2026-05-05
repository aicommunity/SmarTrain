from __future__ import annotations

import ast
from pathlib import Path


TARGET_FILES = (
    "smartrain/services/inference_service.py",
    "smartrain/services/train_service.py",
    "smartrain/external_providers/launchers/mp_infer_launcher.py",
    "smartrain/external_providers/launchers/mfel_infer_launcher.py",
    "smartrain/external_providers/launchers/mfel_val_launcher.py",
    "smartrain/external_providers/launchers/mfel_train_launcher.py",
)


def test_runtime_critical_modules_do_not_use_hasattr() -> None:
    violations: list[str] = []
    for rel_path in TARGET_FILES:
        source = Path(rel_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "hasattr":
                continue
            violations.append(f"{rel_path}:{getattr(node, 'lineno', '?')}")

    assert not violations, "Runtime-critical code contains forbidden hasattr(): " + ", ".join(violations)
