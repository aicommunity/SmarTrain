# Rollout Checklist

## Per-PR

- [ ] Design note updated
- [ ] Tests added/updated
- [ ] No undocumented contract change
- [ ] Migration note updated if needed

## Per-Wave Gate

- [ ] Regression smoke suite green
- [ ] Integration checks green
- [ ] Backward compatibility policy applied
- [ ] Owner approvals complete

## Canonical Cutover Gate

- [ ] Read parity confirmed (`run` vs `models`)
- [ ] Dual-write consistency acceptable
- [ ] Migration dry-run/apply reports reviewed
- [ ] Legacy fallback scope minimized
