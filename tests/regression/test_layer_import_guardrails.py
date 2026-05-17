from __future__ import annotations

import ast
from pathlib import Path

# (package_prefix, forbidden_import_prefix)
_FORBIDDEN_PREFIX_PAIRS: tuple[tuple[str, str], ...] = (
    ("smartrain/services", "smartrain.workflows"),
    ("smartrain/orchestrators", "smartrain.workflows"),
    ("smartrain/domain", "smartrain.workflows"),
    ("smartrain/domain", "smartrain.services"),
    ("smartrain/backends", "smartrain.workflows"),
    ("smartrain/unified", "smartrain.services"),
)

# (relative path under repo root, forbidden module prefix)
_ALLOWLIST: frozenset[tuple[str, str]] = frozenset()


def _py_files_under(prefix: str) -> list[Path]:
    root = Path(prefix.replace(".", "/"))
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.py"))


def _module_matches_forbidden(module: str, forbidden_prefix: str) -> bool:
    return module == forbidden_prefix or module.startswith(f"{forbidden_prefix}.")


def _forbidden_imports(path: Path, forbidden_prefix: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_matches_forbidden(alias.name, forbidden_prefix):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and isinstance(node.module, str):
            if _module_matches_forbidden(node.module, forbidden_prefix):
                found.add(node.module)
    return found


def _check_package(package_prefix: str, forbidden_prefix: str) -> list[str]:
    violations: list[str] = []
    for path in _py_files_under(package_prefix):
        rel = path.as_posix()
        for mod in sorted(_forbidden_imports(path, forbidden_prefix)):
            if not mod.startswith(forbidden_prefix):
                continue
            if (rel, mod) in _ALLOWLIST or any(
                mod.startswith(allowed) for rel_a, allowed in _ALLOWLIST if rel_a == rel
            ):
                continue
            violations.append(f"{rel}: {mod}")
    return violations


def test_layer_import_guardrails() -> None:
    all_violations: list[str] = []
    for pkg, forbidden in _FORBIDDEN_PREFIX_PAIRS:
        all_violations.extend(_check_package(pkg, forbidden))
    assert not all_violations, (
        "Forbidden cross-layer imports detected. "
        f"Allowlist: {sorted(_ALLOWLIST)!r}. Violations: {'; '.join(all_violations)}"
    )
