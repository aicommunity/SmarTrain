# Clean Code Rules (Refactor Phase)

## Prohibited Patterns

- Runtime `hasattr` for core state initialization.
- Hidden dynamic attributes that bypass constructor/dataclass fields.
- Business logic mixed directly inside CLI parsing branches.

## Required Practices

- Explicit fields in constructors/dataclasses.
- Layered separation: parsing -> orchestration -> adapter.
- Shared helpers instead of copy-paste blocks.

## Review Checklist

- Is there any new duplicate helper code?
- Are new fields explicitly declared?
- Does CLI layer call only service/orchestrator APIs?
