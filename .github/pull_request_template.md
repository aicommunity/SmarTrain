## Summary
- What changed and why?
- Which plan/checklist item does this close?

## Refactor/Tech Debt Checklist
- [ ] I checked `docs/refactor/10-implementation-checklist.md` and updated statuses if needed.
- [ ] I checked `docs/refactor/09-tech-debt.md` and added/updated residual debt entries if needed.
- [ ] I reviewed `docs/refactor/07-clean-code-rules.md` and avoided prohibited patterns.

## Phase 8 Guardrails
- [ ] No business logic was added to CLI parsing branches.
- [ ] New fields are explicitly declared (no hidden dynamic attributes).
- [ ] No runtime `hasattr`-based core state initialization was added.
- [ ] No duplicate helper blocks were introduced.

## Tests
- [ ] Local tests were run.
- [ ] Added/updated regression tests for changed behavior.
