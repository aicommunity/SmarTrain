"""Update plan model for workspace legacy → canonical migration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class UpdateCategory(str, Enum):
    LAYOUT = "layout"
    WEIGHTS = "weights"
    RELEASES = "releases"
    MANIFEST = "manifest"
    TESTS = "tests"
    METADATA = "metadata"
    YAML = "yaml"


class UpdateRisk(str, Enum):
    SAFE = "safe"
    ASK = "ask"
    SKIP = "skip"


class UpdateStatus(str, Enum):
    PENDING = "pending"
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"
    DRY_RUN = "dry_run"


ALL_CATEGORIES: tuple[UpdateCategory, ...] = tuple(UpdateCategory)


def parse_categories(raw: str | None) -> frozenset[UpdateCategory]:
    if not raw or not str(raw).strip():
        return frozenset(ALL_CATEGORIES)
    out: set[UpdateCategory] = set()
    for part in str(raw).split(","):
        token = part.strip().lower()
        if not token:
            continue
        out.add(UpdateCategory(token))
    return frozenset(out) if out else frozenset(ALL_CATEGORIES)


@dataclass
class UpdateStep:
    id: str
    category: UpdateCategory
    risk: UpdateRisk
    title: str
    detail: str = ""
    paths: list[str] = field(default_factory=list)
    action: str = ""
    status: UpdateStatus = UpdateStatus.PENDING
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["risk"] = self.risk.value
        d["status"] = self.status.value
        return d


@dataclass
class UpdatePlan:
    workspace_root: str
    steps: list[UpdateStep] = field(default_factory=list)
    residual: list[UpdateStep] = field(default_factory=list)

    def filtered(self, categories: frozenset[UpdateCategory]) -> UpdatePlan:
        return UpdatePlan(
            workspace_root=self.workspace_root,
            steps=[s for s in self.steps if s.category in categories],
            residual=[s for s in self.residual if s.category in categories],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_root": self.workspace_root,
            "steps": [s.to_dict() for s in self.steps],
            "residual": [s.to_dict() for s in self.residual],
            "counts": {
                "total": len(self.steps),
                "safe": sum(1 for s in self.steps if s.risk == UpdateRisk.SAFE),
                "ask": sum(1 for s in self.steps if s.risk == UpdateRisk.ASK),
                "skip": sum(1 for s in self.steps if s.risk == UpdateRisk.SKIP),
            },
        }
