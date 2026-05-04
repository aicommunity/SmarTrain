# Refactor Scope

## Goals

- Reduce duplicated logic in CLI and ML flows.
- Introduce isolated domain layers and explicit contracts.
- Prepare multi-backend and multi-task architecture.

## In Scope

- Commands: `train`, `test`, `inference`, `analyze`, `balance`, `augment`.
- CLI request/response normalization and interactive behavior.
- Backend abstraction and task abstraction.
- Canonical schema unification for `run` and `models`.

## Out of Scope

- Full implementation of new training backends in this phase.
- End-user feature expansion unrelated to architecture cleanup.

## Non-Functional Constraints

- Keep current behavior by default unless explicitly migrated.
- Prefer soft deprecations before hard removals.
- Every major change must have tests and migration notes.
