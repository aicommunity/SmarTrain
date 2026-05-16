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
)

# (relative path under repo root, forbidden module prefix)
_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {
        # Wave 5: remove when backends/implementations/ultralytics is wired
        ("smartrain/backends/ultralytics_adapter.py", "smartrain.workflows.inference"),
        ("smartrain/backends/external_provider_adapter.py", "smartrain.workflows.inference"),
        # Wave 3: gateway may use adapters instead of services.analyze (metrics_reader)
        ("smartrain/orchestrators/canonical_gateway.py", "smartrain.services.analyze"),
        # Wave 2 follow-up: decouple analyze services from workflows (testing/datasets)
        ("smartrain/services/analyze/metrics_reader.py", "smartrain.workflows.testing"),
        ("smartrain/services/analyze/report_writer.py", "smartrain.workflows.datasets"),
        ("smartrain/services/analyze/ultralytics_test_artifacts.py", "smartrain.workflows.testing"),
    }
)


def _py_files_under(prefix: str) -> list[Path]:
    root = Path(prefix.replace(".", "/"))
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.py"))


def _imports_from_workflows(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not isinstance(node.module, str):
            continue
        if node.module == "smartrain.workflows" or node.module.startswith("smartrain.workflows."):
            found.add(node.module)
    return found


def _check_package(package_prefix: str, forbidden_prefix: str) -> list[str]:
    violations: list[str] = []
    for path in _py_files_under(package_prefix):
        rel = path.as_posix()
        for mod in sorted(_imports_from_workflows(path)):
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
